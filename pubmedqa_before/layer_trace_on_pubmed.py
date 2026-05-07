# layer_trace_on_filtered_unknown_scanplan_B.py
# Scheme B conflict injection: inject an EVIDENCE block that asserts the OPPOSITE of gold (conflict label).
# Keeps UNKNOWN-aware metrics + scan plan generation.
#
# IC prompt = base rule + (optional) Context + injected EVIDENCE(conflict_label) at position.
# NC prompt = internal knowledge only.
#
# Example (weighted scan plan):
# python layer_trace_on_filtered_unknown_scanplan_B.py \
#   --data /mnt/data/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json \
#   --fmt json \
#   --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
#   --position before_answer \
#   --patch_k 8 \
#   --patch_all_tokens \
#   --limit 50 \
#   --metrics abstain_adv,follow_conflict,yesno_margin \
#   --scan_plan_out result/scan_plan_weighted.json \
#   --scan_plan_metrics abstain_adv,yesno_margin \
#   --scan_plan_weights 0.5,0.5 \
#   --out result/trace_B.json \
#   --plot_prefix result/trace_B
#
# IMPORTANT: requires " yes"/" no"/" unknown" to be single tokens for next-token scoring.

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

SYSTEM_PROMPT = "You are a helpful medical QA assistant."
YES, NO, UNK = "yes", "no", "unknown"

BASE_RULE_IC = (
    "Answer the question using your best internal medical knowledge.\n"
    "You may be given an evidence block. The evidence may be correct or incorrect.\n"
    "If you cannot answer with high confidence, answer: unknown.\n"
    "Output one token only from: yes / no / unknown.\n\n"
)

EVIDENCE_TMPL = (
    "EVIDENCE (may be incorrect):\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)

# ---------------- IO ----------------
def load_json_or_jsonl(path: str, fmt: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        if fmt == "jsonl":
            rows = []
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
            return rows
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rows.append({"id": k, **v})
            else:
                rows.append({"id": k, "value": v})
        return rows
    raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")


def get_field(ex: Dict[str, Any], key: str) -> Any:
    cur: Any = ex
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_context(ctx: Any) -> str:
    if ctx is None:
        return ""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, list):
        return "\n".join([str(x) for x in ctx])
    if isinstance(ctx, dict):
        if "contexts" in ctx and isinstance(ctx["contexts"], list):
            return "\n".join([str(x) for x in ctx["contexts"]])
        return json.dumps(ctx, ensure_ascii=False)
    return str(ctx)


# ---------------- Prompting ----------------
def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def make_nc_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY your internal medical knowledge.\n"
        "If you cannot answer with high confidence, answer: unknown.\n"
        "Output one token only from: yes / no / unknown.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def invert_yes_no(y: str) -> str:
    y = y.strip().lower()
    if y == YES:
        return NO
    if y == NO:
        return YES
    raise ValueError(f"Cannot invert label: {y}")


def inject_block(base_prompt: str, block: str, position: str) -> str:
    """Insert a text block into base_prompt at prefix/before_question/before_answer."""
    block = block.strip() + "\n"
    if position == "prefix":
        return block + "\n" + base_prompt

    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            return block + "\n" + base_prompt
        return base_prompt.replace(marker, "\n" + block + marker, 1)

    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            return block + "\n" + base_prompt
        return base_prompt.replace(marker, "\n" + block + marker, 1)

    raise ValueError(f"Unknown position: {position}")


def make_ic_prompt_scheme_b(question: str, context: str, gold: str, position: str, include_context: bool) -> str:
    """
    Scheme B: evidence asserts the OPPOSITE of gold (conflict label).
    Optionally include original context, but the key injected evidence is the conflict block.
    """
    gold = gold.strip().lower()
    conflict = invert_yes_no(gold)
    evidence = EVIDENCE_TMPL.format(ans=conflict)

    parts = [BASE_RULE_IC]
    if include_context:
        parts.append(f"Context:\n{context}\n\n")
    parts.append(f"Question: {question}\nAnswer:")
    base = "".join(parts)

    return inject_block(base, evidence, position)


# ---------------- Model helpers ----------------
def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot find transformer layers on this model.")


def _check_single_token_choices(tok):
    yid = tok(" yes", add_special_tokens=False).input_ids
    nid = tok(" no", add_special_tokens=False).input_ids
    uid = tok(" unknown", add_special_tokens=False).input_ids
    if not (len(yid) == len(nid) == len(uid) == 1):
        raise SystemExit(
            f'ERROR: token lengths: " yes"={len(yid)}, " no"={len(nid)}, " unknown"={len(uid)}. '
            "This script requires single-token choices for next-token scoring."
        )
    return yid[0], nid[0], uid[0]


@torch.no_grad()
def score_nexttoken_logps(model, tok, formatted_prompt: str, ids: Tuple[int, int, int]) -> Dict[str, float]:
    """Return logP for yes/no/unknown at the next token."""
    yid, nid, uid = ids
    dev = next(model.parameters()).device
    enc = tok(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)
    return {
        YES: float(logp[0, yid]),
        NO: float(logp[0, nid]),
        UNK: float(logp[0, uid]),
    }


# ---------------- Effects (metrics) ----------------
def effect_yesno_margin(scores: Dict[str, float]) -> float:
    return scores[YES] - scores[NO]


def effect_abstain_adv(scores: Dict[str, float]) -> float:
    best_yn = max(scores[YES], scores[NO])
    return scores[UNK] - best_yn


def effect_follow_conflict(scores: Dict[str, float], gold: str) -> float:
    gold = gold.strip().lower()
    conflict = invert_yes_no(gold)
    return scores[conflict] - scores[gold]


# ---------------- Layer hooks ----------------
def capture_layer_outputs_lastk(model, k: int):
    layers = _get_layers(model)
    cache = [None] * len(layers)
    handles = []
    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                kk = min(k, hs.size(1))
                cache[ii] = hs[:, -kk:, :].detach()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))
    return cache, handles


def patch_layer_lastk(model, layer_idx: int, replacement_tail: torch.Tensor):
    layers = _get_layers(model)
    layer = layers[layer_idx]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if replacement_tail is None:
            return out
        kk = replacement_tail.size(1)
        if kk == 0:
            return out
        hs2 = hs.clone()
        hs2[:, -kk:, :] = replacement_tail.to(hs2.device, dtype=hs2.dtype)
        if isinstance(out, (tuple, list)):
            out = list(out)
            out[0] = hs2
            return tuple(out)
        return hs2

    return layer.register_forward_hook(hook)


def capture_layer_outputs_all(model):
    layers = _get_layers(model)
    cache = [None] * len(layers)
    handles = []
    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                cache[ii] = hs.detach()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))
    return cache, handles


def patch_layer_all(model, layer_idx: int, replacement_states: torch.Tensor):
    """
    Patch full hidden states for one layer.
    If NC/IC sequence lengths differ, patch the overlapping tail.
    """
    layers = _get_layers(model)
    layer = layers[layer_idx]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        if replacement_states is None:
            return out
        hs2 = hs.clone()
        rep = replacement_states.to(hs2.device, dtype=hs2.dtype)
        if rep.size(1) == hs2.size(1):
            hs2[:, :, :] = rep
        else:
            kk = min(rep.size(1), hs2.size(1))
            if kk > 0:
                hs2[:, -kk:, :] = rep[:, -kk:, :]
        if isinstance(out, (tuple, list)):
            out = list(out)
            out[0] = hs2
            return tuple(out)
        return hs2

    return layer.register_forward_hook(hook)


# ---------------- Plot ----------------
def plot_layer_scores(scores: List[float], out_path: str, title: str):
    plt.figure()
    plt.plot(range(len(scores)), scores)
    plt.xlabel("layer_idx")
    plt.ylabel("mean score (|effect| - |effect_p|)")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_metrics_arg(s: str) -> List[str]:
    if not s:
        return ["yesno_margin", "abstain_adv"]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    allowed = {"yesno_margin", "abstain_adv", "follow_conflict"}
    for p in parts:
        if p not in allowed:
            raise SystemExit(f"Unknown metric '{p}'. Allowed: {sorted(allowed)}")
    return parts


# ---------------- Scan plan (plateau heuristic) ----------------
def _find_plateau_start(
    scores: List[float],
    plateau_min_score_frac: float,
    plateau_eps: float,
    plateau_min_len: int,
) -> Optional[int]:
    if not scores:
        return None
    max_score = max(scores)
    if max_score <= 0:
        return None
    thr = plateau_min_score_frac * max_score
    n = len(scores)
    for s in range(0, n - plateau_min_len):
        if scores[s] < thr:
            continue
        ok = True
        for i in range(s, s + plateau_min_len - 1):
            if abs(scores[i + 1] - scores[i]) > plateau_eps:
                ok = False
                break
        if ok:
            return s
    return None


def build_scan_plan(
    scores: List[float],
    tail_k: int = 4,
    preplateau_width: int = 7,
    plateau_min_score_frac: float = 0.8,
    plateau_eps: float = 0.6,
    plateau_min_len: int = 6,
    topk_fallback: int = 8,
    rise_k: int = 6,
    rise_min_layer: int = 0,
    rise_max_layer: int = -1,
) -> Dict[str, Any]:
    n_layers = len(scores)
    plateau_start = _find_plateau_start(scores, plateau_min_score_frac, plateau_eps, plateau_min_len)
    tail_layers = list(range(max(0, n_layers - tail_k), n_layers))
    n_layers = len(scores)

    # ---- round0_rise: pick layers where the score rises fastest ----
    if rise_max_layer is None or rise_max_layer < 0:
        rise_max = max(0, n_layers - 1)  # diffs length is n_layers-1
    else:
        rise_max = min(rise_max_layer, n_layers - 1)

    rise_min = max(0, min(rise_min_layer, n_layers - 1))

    diffs = []
    for i in range(0, n_layers - 1):
        if i < rise_min or i >= rise_max:
            continue
        diffs.append((abs(scores[i + 1] - scores[i]), i))

    # take top-k by diff magnitude; use i+1 as the "rise layer"
    diffs.sort(key=lambda x: x[0], reverse=True)
    rise_layers = []
    for _, i in diffs[:max(0, rise_k)]:
        rise_layers.append(i + 1)

    # unique + sorted
    rise_layers = sorted(set(rise_layers))

    def dedup_rounds(plan: Dict[str, Any]) -> Dict[str, Any]:
        # Keep rounds disjoint by priority:
        # round0_rise > round1_preplateau/round1_topk_fallback > round2_tail
        round_keys = ["round0_rise", "round1_preplateau", "round1_topk_fallback", "round2_tail"]
        seen = set()
        for key in round_keys:
            vals = plan.get(key)
            if not vals:
                continue
            clean = []
            for x in vals:
                v = int(x)
                if v in seen:
                    continue
                seen.add(v)
                clean.append(v)
            plan[key] = clean
        return plan

    if plateau_start is not None:
        a = max(0, plateau_start - preplateau_width)
        b = min(n_layers - 1, plateau_start + 1)
        round1 = list(range(a, b + 1))
        return dedup_rounds({
            "method": "plateau",
            "plateau_start": plateau_start,
            "round0_rise": rise_layers,
            "round1_preplateau": round1,
            "round2_tail": tail_layers,
            "tail_k": tail_k,
            "preplateau_width": preplateau_width,
            "plateau_min_score_frac": plateau_min_score_frac,
            "plateau_eps": plateau_eps,
            "plateau_min_len": plateau_min_len,
        })

    idx_sorted = sorted(range(n_layers), key=lambda i: scores[i], reverse=True)[:topk_fallback]
    idx_sorted = sorted(idx_sorted)
    return dedup_rounds({
        "method": "topk_fallback",
        "plateau_start": None,
        "round0_rise": rise_layers,
        "round1_topk_fallback": idx_sorted,
        "round2_tail": tail_layers,
        "tail_k": tail_k,
        "topk_fallback": topk_fallback,
        "plateau_min_score_frac": plateau_min_score_frac,
        "plateau_eps": plateau_eps,
        "plateau_min_len": plateau_min_len,
    })


def parse_weights(weights_str: str, n: int) -> List[float]:
    if not weights_str:
        return [1.0 / n] * n
    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    if len(parts) != n:
        raise SystemExit(f"--scan_plan_weights expects {n} comma-separated weights, got {len(parts)}")
    w = [float(x) for x in parts]
    s = sum(w)
    if s == 0:
        raise SystemExit("--scan_plan_weights sum to 0")
    return [x / s for x in w]


def combine_scores(layer_score_mean: Dict[str, List[float]], metrics: List[str], weights: List[float]) -> List[float]:
    n_layers = len(next(iter(layer_score_mean.values())))
    combined = [0.0] * n_layers
    for m, w in zip(metrics, weights):
        s = layer_score_mean[m]
        if len(s) != n_layers:
            raise RuntimeError("Metric score length mismatch")
        for i in range(n_layers):
            combined[i] += w * s[i]
    return combined


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="filtered json/jsonl dataset")
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl"])
    ap.add_argument("--model", required=True, help="local HF model path")
    ap.add_argument("--position", default="before_answer",
                    choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--patch_k", type=int, default=8)
    ap.add_argument(
        "--patch_all_tokens",
        action="store_true",
        help="Patch full hidden states. If NC/IC lengths differ, patch overlapping tail tokens.",
    )
    ap.add_argument("--limit", type=int, default=50, help="max examples (patching is expensive)")
    ap.add_argument("--out", default="result/trace_B.json")
    ap.add_argument("--plot_prefix", default="result/trace_B",
                    help="prefix for plots; will append _<metric>.png")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device_map", default="auto")

    ap.add_argument("--metrics", default="abstain_adv,follow_conflict",
                    help="comma-separated: yesno_margin,abstain_adv,follow_conflict")

    ap.add_argument("--include_context", action="store_true",
                    help="If set, include original dataset CONTEXTS in IC prompt (in addition to evidence block).")

    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="id")

    # scan plan options
    ap.add_argument("--scan_plan_out", default="", help="if set, write scan plan JSON here")
    ap.add_argument("--scan_plan_metric", default="", help="single metric to build scan plan from")
    ap.add_argument("--scan_plan_metrics", default="", help="comma-separated metrics to combine for scan plan")
    ap.add_argument("--scan_plan_weights", default="", help="comma-separated weights (same count as scan_plan_metrics)")

    ap.add_argument("--tail_k", type=int, default=4)
    ap.add_argument("--preplateau_width", type=int, default=7)
    ap.add_argument("--plateau_min_score_frac", type=float, default=0.8)
    ap.add_argument("--plateau_eps", type=float, default=0.6)
    ap.add_argument("--plateau_min_len", type=int, default=6)
    ap.add_argument("--topk_fallback", type=int, default=8)
    ap.add_argument("--rise_k", type=int, default=6,
                help="How many rise-phase layers to include (by largest |score[i+1]-score[i]|).")
    ap.add_argument("--rise_min_layer", type=int, default=0,
                    help="Only consider rise candidates with i in [rise_min_layer, rise_max_layer).")
    ap.add_argument("--rise_max_layer", type=int, default=-1,
                help="Upper bound (exclusive) for rise candidate index i; -1 means n_layers-1.")


    args = ap.parse_args()
    metrics = parse_metrics_arg(args.metrics)
    patch_mode = "all_tokens" if args.patch_all_tokens else f"last_{args.patch_k}"

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
    )
    model.eval()

    ids = _check_single_token_choices(tok)

    data = load_json_or_jsonl(args.data, args.fmt)
    layers = _get_layers(model)
    n_layers = len(layers)

    sums = {m: torch.zeros(n_layers, dtype=torch.float64) for m in metrics}
    n_samples = 0
    per_example = []

    for ex in tqdm(data, desc="Layer tracing (NC -> IC, Scheme B)", unit="ex"):
        if args.limit > 0 and n_samples >= args.limit:
            break

        q = get_field(ex, args.q_field)
        ctx_raw = get_field(ex, args.c_field)
        gold = get_field(ex, args.y_field)
        if q is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            continue

        ctx = normalize_context(ctx_raw)
        sid = get_field(ex, args.id_field) if args.id_field else ex.get("id", None)

        nc_user = make_nc_prompt(str(q))
        ic_user = make_ic_prompt_scheme_b(
            question=str(q),
            context=ctx,
            gold=gold,
            position=args.position,
            include_context=args.include_context,
        )

        p_nc = format_chat(tok, nc_user, args.enable_thinking)
        p_ic = format_chat(tok, ic_user, args.enable_thinking)

        sc_nc = score_nexttoken_logps(model, tok, p_nc, ids)
        sc_ic = score_nexttoken_logps(model, tok, p_ic, ids)

        base_E: Dict[str, float] = {}
        con_E: Dict[str, float] = {}
        eff: Dict[str, float] = {}

        for m in metrics:
            if m == "yesno_margin":
                base_E[m] = effect_yesno_margin(sc_nc)
                con_E[m] = effect_yesno_margin(sc_ic)
            elif m == "abstain_adv":
                base_E[m] = effect_abstain_adv(sc_nc)
                con_E[m] = effect_abstain_adv(sc_ic)
            elif m == "follow_conflict":
                base_E[m] = effect_follow_conflict(sc_nc, gold)
                con_E[m] = effect_follow_conflict(sc_ic, gold)
            else:
                raise RuntimeError("unreachable")
            eff[m] = con_E[m] - base_E[m]

        # Cache NC per-layer states (all tokens or last-k tail)
        if args.patch_all_tokens:
            base_cache, h1 = capture_layer_outputs_all(model)
        else:
            base_cache, h1 = capture_layer_outputs_lastk(model, args.patch_k)
        _ = score_nexttoken_logps(model, tok, p_nc, ids)
        for hh in h1:
            hh.remove()

        layer_scores_this = {m: [] for m in metrics}

        for i in range(n_layers):
            if args.patch_all_tokens:
                ph = patch_layer_all(model, i, base_cache[i])
            else:
                ph = patch_layer_lastk(model, i, base_cache[i])
            sc_p = score_nexttoken_logps(model, tok, p_ic, ids)
            ph.remove()

            for m in metrics:
                if m == "yesno_margin":
                    Ep = effect_yesno_margin(sc_p)
                elif m == "abstain_adv":
                    Ep = effect_abstain_adv(sc_p)
                elif m == "follow_conflict":
                    Ep = effect_follow_conflict(sc_p, gold)
                else:
                    raise RuntimeError("unreachable")

                eff_p = Ep - base_E[m]
                score_i = abs(eff[m]) - abs(eff_p)
                sums[m][i] += score_i
                layer_scores_this[m].append(float(score_i))

        per_example.append({
            "id": sid,
            "gold": gold,
            "conflict": invert_yes_no(gold),
            "position": args.position,
            "patch_k": args.patch_k,
            "patch_all_tokens": bool(args.patch_all_tokens),
            "patch_mode": patch_mode,
            "include_context": bool(args.include_context),
            "scores_nc": sc_nc,
            "scores_ic": sc_ic,
            "effect": eff,
            "layer_score": layer_scores_this,
        })
        n_samples += 1

    layer_score_mean = {m: (sums[m] / max(n_samples, 1)).tolist() for m in metrics}
    top_layers = {
        m: sorted(range(n_layers), key=lambda i: layer_score_mean[m][i], reverse=True)[:10]
        for m in metrics
    }

    res = {
        "scheme": "B_conflict_evidence_injection",
        "n_samples": n_samples,
        "n_layers": n_layers,
        "position": args.position,
        "patch_k": args.patch_k,
        "patch_all_tokens": bool(args.patch_all_tokens),
        "patch_mode": patch_mode,
        "include_context": bool(args.include_context),
        "metrics": metrics,
        "top_layers_by_score": top_layers,
        "layer_score_mean": layer_score_mean,
        "per_example": per_example,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    for m in metrics:
        plot_layer_scores(
            layer_score_mean[m],
            f"{args.plot_prefix}_{m}.png",
            title=f"Layer trace NC->IC (Scheme B) | pos={args.position} | patch={patch_mode} | metric={m} | n={n_samples}",
        )

    # ---- scan plan generation ----
    if args.scan_plan_out:
        if args.scan_plan_metric and args.scan_plan_metrics:
            raise SystemExit("Use either --scan_plan_metric or --scan_plan_metrics, not both.")

        if args.scan_plan_metric:
            plan_metric = args.scan_plan_metric.strip()
            if plan_metric not in layer_score_mean:
                raise SystemExit(f"--scan_plan_metric '{plan_metric}' not in computed metrics {metrics}")
            plan_scores = layer_score_mean[plan_metric]
            plan_info = {"mode": "single_metric", "metric": plan_metric, "weights": {plan_metric: 1.0}}
        elif args.scan_plan_metrics:
            plan_metrics = [p.strip() for p in args.scan_plan_metrics.split(",") if p.strip()]
            for pm in plan_metrics:
                if pm not in layer_score_mean:
                    raise SystemExit(f"--scan_plan_metrics includes '{pm}' not in computed metrics {metrics}")
            w = parse_weights(args.scan_plan_weights, len(plan_metrics))
            plan_scores = combine_scores(layer_score_mean, plan_metrics, w)
            plan_info = {
                "mode": "weighted_metrics",
                "metrics": plan_metrics,
                "weights": {m: float(wi) for m, wi in zip(plan_metrics, w)},
            }
        else:
            pm = metrics[0]
            plan_scores = layer_score_mean[pm]
            plan_info = {"mode": "default_first_metric", "metric": pm, "weights": {pm: 1.0}}

        scan_plan = build_scan_plan(
            plan_scores,
            tail_k=args.tail_k,
            preplateau_width=args.preplateau_width,
            plateau_min_score_frac=args.plateau_min_score_frac,
            plateau_eps=args.plateau_eps,
            plateau_min_len=args.plateau_min_len,
            topk_fallback=args.topk_fallback,
            rise_k=args.rise_k,
            rise_min_layer=args.rise_min_layer,
            rise_max_layer=args.rise_max_layer,

        )

        scan_plan_out = {
            "scheme": "B_conflict_evidence_injection",
            "position": args.position,
            "patch_k": args.patch_k,
            "patch_all_tokens": bool(args.patch_all_tokens),
            "patch_mode": patch_mode,
            "include_context": bool(args.include_context),
            "n_samples": n_samples,
            "n_layers": n_layers,
            "plan_scores_source": plan_info,
            "scan_plan": scan_plan,
        }

        sp_path = Path(args.scan_plan_out)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.write_text(json.dumps(scan_plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved scan plan: {sp_path}")

    print(f"[OK] Saved trace: {out_path}")
    for m in metrics:
        print(f"[OK] Plot : {args.plot_prefix}_{m}.png  | top layers: {top_layers[m]}")

if __name__ == "__main__":
    main()
