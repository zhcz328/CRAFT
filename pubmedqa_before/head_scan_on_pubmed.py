# head_scan_on_pubmed.py
# Head scan for PubMed-style dataset (QUESTION/CONTEXTS/final_decision),
# using Scheme B conflict injection (invert(gold) evidence) consistent with layer_trace_on_pubmed.py.
#
# Head ablation method:
#   - Ablate an attention head by pre-softmax attention-logits masking (via attention_mask),
#     adapted from ablate_head_inf.py.
#
# Metrics (scalar E):
#   - yesno_margin    : logP(yes) - logP(no)
#   - abstain_adv     : logP(unknown) - max(logP(yes), logP(no))
#   - follow_conflict : logP(invert(gold)) - logP(gold)
#
# Weighted mode (Scheme C):
#   E = Σ_j w_j * E_metric_j
# Enable with:
#   --metrics abstain_adv,follow_conflict,yesno_margin --weights 0.4,0.4,0.2
# If --weights omitted, weights are uniform (and normalized).
#
# For each sample we compute E_nc, E_ic, and effect = E_ic - E_nc.
# For each (layer, head) ablation we recompute E_nc_ab, E_ic_ab and compute:
#   mean_abs_effect_reduction = mean( |effect| - |effect_ab| ), where effect_ab = E_ic_ab - E_nc  (NC baseline kept unablated)
#   mean_abs_base_change      = mean( |E_nc_ab - E_nc| )
#
# Plan mode consumes scan_plan.json (new schema) or legacy {"rounds":[...]}.

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

YES, NO, UNK = "yes", "no", "unknown"
SYSTEM_PROMPT = "You are a helpful medical QA assistant."

# ---- Prompt templates (match layer_trace_on_pubmed.py style) ----
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

def invert_yes_no(y: str) -> str:
    y = y.strip().lower()
    if y == YES: return NO
    if y == NO:  return YES
    raise ValueError(f"Cannot invert: {y}")

def normalize_context(ctx: Any) -> str:
    if ctx is None: return ""
    if isinstance(ctx, str): return ctx
    if isinstance(ctx, list): return "\n".join([str(x) for x in ctx])
    if isinstance(ctx, dict):
        if "contexts" in ctx and isinstance(ctx["contexts"], list):
            return "\n".join([str(x) for x in ctx["contexts"]])
        return json.dumps(ctx, ensure_ascii=False)
    return str(ctx)

def get_field(ex: Dict[str, Any], key: str) -> Any:
    cur: Any = ex
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

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

def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
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

def inject_block(base_prompt: str, block: str, position: str) -> str:
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
    gold = gold.strip().lower()
    conflict = invert_yes_no(gold)
    evidence = EVIDENCE_TMPL.format(ans=conflict)

    parts = [BASE_RULE_IC]
    if include_context:
        parts.append(f"Context:\n{context}\n\n")
    parts.append(f"Question: {question}\nAnswer:")
    base = "".join(parts)

    return inject_block(base, evidence, position)

# ---- token / scoring ----
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
    yid, nid, uid = ids
    dev = next(model.parameters()).device
    enc = tok(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)
    return {YES: float(logp[0, yid]), NO: float(logp[0, nid]), UNK: float(logp[0, uid])}

# ---- metrics (scalar E) ----
def E_yesno_margin(sc: Dict[str, float]) -> float:
    return sc[YES] - sc[NO]

def E_abstain_adv(sc: Dict[str, float]) -> float:
    return sc[UNK] - max(sc[YES], sc[NO])

def E_follow_conflict(sc: Dict[str, float], gold: str) -> float:
    gold = gold.strip().lower()
    conflict = invert_yes_no(gold)
    return sc[conflict] - sc[gold]

def compute_E_single(metric: str, sc: Dict[str, float], gold: Optional[str]) -> float:
    if metric == "yesno_margin":
        return E_yesno_margin(sc)
    if metric == "abstain_adv":
        return E_abstain_adv(sc)
    if metric == "follow_conflict":
        if gold is None:
            raise ValueError("follow_conflict needs gold.")
        return E_follow_conflict(sc, gold)
    raise ValueError(metric)

def parse_metrics_and_weights(metric: Optional[str], metrics: Optional[str], weights: Optional[str]):
    # Backward compat: if --metrics not provided, use [--metric] with weight 1.
    if metrics is None or str(metrics).strip() == "":
        m = (metric or "yesno_margin").strip()
        return [m], [1.0]

    ms = [x.strip() for x in str(metrics).split(",") if x.strip()]
    if not ms:
        m = (metric or "yesno_margin").strip()
        return [m], [1.0]

    if weights is None or str(weights).strip() == "":
        ws = [1.0 / len(ms)] * len(ms)
        return ms, ws

    ws = [float(x.strip()) for x in str(weights).split(",") if x.strip()]
    if len(ws) != len(ms):
        raise SystemExit(f"ERROR: --weights length {len(ws)} != --metrics length {len(ms)}")
    s = sum(ws)
    if s <= 0:
        raise SystemExit("ERROR: sum(weights) must be > 0")
    ws = [w / s for w in ws]  # normalize
    return ms, ws

def compute_E_weighted(ms: List[str], ws: List[float], sc: Dict[str, float], gold: Optional[str]) -> float:
    return sum(w * compute_E_single(m, sc, gold) for m, w in zip(ms, ws))

# --------------------------
# Head masking (ablation) via attention-logits masking (pre-softmax)
# Adapted from ablate_head_inf.py
# --------------------------

def _get_num_heads_and_hidden(model):
    cfg = getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    nheads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    return nheads, hidden

def _find_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Cannot locate transformer layers (model.model.layers / transformer.h / gpt_neox.layers).")

def _find_self_attn_modules(model):
    layers = _find_layers(model)
    layer2attn = {}
    for i, layer in enumerate(layers):
        mod = None
        if hasattr(layer, "self_attn"):
            mod = layer.self_attn
        elif hasattr(layer, "attn"):
            mod = layer.attn
        if mod is not None:
            layer2attn[i] = mod
    if not layer2attn:
        raise RuntimeError("Cannot find any self-attention modules in layers.")
    return layer2attn, len(layers)

def _apply_head_specific_mask(attn_mask, n_heads: int, heads_to_mask, keep_mode: str = "self"):
    if attn_mask is None:
        return None
    if not torch.is_tensor(attn_mask):
        return attn_mask
    if attn_mask.dim() != 4:
        return attn_mask

    B, Hm, T, S = attn_mask.shape

    if Hm == 1:
        mask = attn_mask.expand(B, n_heads, T, S).clone()
    elif Hm == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask

    neg = torch.finfo(mask.dtype).min

    for h in heads_to_mask:
        if not (0 <= h < n_heads):
            continue
        mask[:, h, :, :] = neg
        if keep_mode == "bos":
            mask[:, h, :, 0] = 0
        else:
            diag_len = min(T, S)
            idx = torch.arange(diag_len, device=mask.device)
            mask[:, h, idx, idx] = 0

    return mask

def install_head_mask_hooks(model, layer2heads: Dict[int, List[int]], keep_mode: str = "self"):
    layer2attn, _ = _find_self_attn_modules(model)
    n_heads_cfg, _ = _get_num_heads_and_hidden(model)
    if n_heads_cfg is None or n_heads_cfg <= 0:
        raise RuntimeError("Cannot read num_attention_heads from model.config; cannot apply head-specific logits mask.")

    handles = []
    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(h) for h in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, args, kwargs):
                attn_mask = None
                if kwargs is not None and "attention_mask" in kwargs:
                    attn_mask = kwargs["attention_mask"]
                elif len(args) >= 2:
                    attn_mask = args[1]

                new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)
                if new_mask is attn_mask:
                    return args, kwargs

                if kwargs is not None and "attention_mask" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = new_mask
                    return args, kwargs

                args = list(args)
                if len(args) >= 2:
                    args[1] = new_mask
                return tuple(args), kwargs
            return pre_hook

        try:
            handle = attn_mod.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=True)
        except TypeError:
            def make_pre_hook_no_kwargs(heads_local):
                def pre_hook_no_kwargs(module, inputs):
                    args = list(inputs)
                    if len(args) < 2:
                        return inputs
                    attn_mask = args[1]
                    new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)
                    args[1] = new_mask
                    return tuple(args)
                return pre_hook_no_kwargs
            handle = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))
        handles.append(handle)
    return handles

def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

# ---- scan plan loader (old + new schema) ----
def _dedup_sorted_layers(layers):
    return sorted({int(x) for x in (layers or [])})

def load_plan_layers(plan_path: str, round_name: Optional[str] = None):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    rounds = plan.get("rounds")
    if isinstance(rounds, list) and rounds:
        round_layers = [(r.get("name", f"round{idx}"), _dedup_sorted_layers(r.get("layers", [])))
                        for idx, r in enumerate(rounds)]
    else:
        sp = plan.get("scan_plan")
        if not isinstance(sp, dict):
            raise ValueError("plan file missing 'rounds' (old schema) and missing 'scan_plan' (new schema).")
        round_layers = []
        if sp.get("round0_rise"):
            round_layers.append(("round0_rise", _dedup_sorted_layers(sp.get("round0_rise"))))
        if sp.get("round1_preplateau"):
            round_layers.append(("round1_preplateau", _dedup_sorted_layers(sp.get("round1_preplateau"))))
        if sp.get("round1_topk_fallback"):
            round_layers.append(("round1_topk_fallback", _dedup_sorted_layers(sp.get("round1_topk_fallback"))))
        if sp.get("round2_tail"):
            round_layers.append(("round2_tail", _dedup_sorted_layers(sp.get("round2_tail"))))
        round_layers = [(n, ls) for (n, ls) in round_layers if ls]
        if not round_layers:
            raise ValueError("scan_plan contains no layer lists.")

    if round_name is None:
        return plan, round_layers
    for n, ls in round_layers:
        if n == round_name:
            return plan, [(n, ls)]
    raise ValueError(f"round '{round_name}' not found. Available: {[n for n,_ in round_layers]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json","jsonl"])
    ap.add_argument("--model", required=True)

    ap.add_argument("--position", default="before_answer", choices=["prefix","before_question","before_answer"])
    ap.add_argument("--include_context", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--keep_mode", default="self", choices=["self","bos"], help="mask keep mode to avoid NaNs")

    # Backward-compatible single metric
    ap.add_argument("--metric", default="yesno_margin", choices=["yesno_margin","abstain_adv","follow_conflict"],
                    help="Single metric mode (ignored if --metrics is provided).")
    # Weighted multi-metric
    ap.add_argument("--metrics", default=None,
                    help="Comma-separated metrics for weighted mode, e.g. 'abstain_adv,follow_conflict,yesno_margin'")
    ap.add_argument("--weights", default=None,
                    help="Comma-separated weights aligned with --metrics, e.g. '0.4,0.4,0.2'. If omitted, uniform weights.")

    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")

    ap.add_argument("--max_examples", type=int, default=50)

    ap.add_argument("--layers", default=None, help="manual: comma-separated layer idxs")
    ap.add_argument("--plan", default=None, help="scan_plan.json from layer_trace_on_pubmed.py")
    ap.add_argument("--round", default=None, help="only run this round name (e.g. round0_rise)")
    ap.add_argument("--plan_out_dir", default="result/headscan_pubmed")

    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    ms, ws = parse_metrics_and_weights(args.metric, args.metrics, args.weights)

    for m in ms:
        if m not in ("yesno_margin","abstain_adv","follow_conflict"):
            raise SystemExit(f"ERROR: unknown metric '{m}'. Allowed: yesno_margin, abstain_adv, follow_conflict")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation="eager",
    )
    model.eval()

    ids = _check_single_token_choices(tok)

    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if n_heads is None:
        raise RuntimeError("Missing num_attention_heads/n_head in config")

    data = load_json_or_jsonl(args.data, args.fmt)

    samples = []
    kept = 0
    for ex in tqdm(data, desc="Build samples", unit="ex"):
        if args.max_examples > 0 and kept >= args.max_examples:
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
        nc_user = make_nc_prompt(str(q))
        ic_user = make_ic_prompt_scheme_b(str(q), ctx, gold, args.position, args.include_context)

        p_nc = format_chat(tok, nc_user, args.enable_thinking)
        p_ic = format_chat(tok, ic_user, args.enable_thinking)

        sc_nc = score_nexttoken_logps(model, tok, p_nc, ids)
        sc_ic = score_nexttoken_logps(model, tok, p_ic, ids)

        E_nc = compute_E_weighted(ms, ws, sc_nc, gold)
        E_ic = compute_E_weighted(ms, ws, sc_ic, gold)

        samples.append((E_nc, E_ic, p_nc, p_ic, gold))
        kept += 1

    if not samples:
        raise SystemExit("No usable examples (need gold in {yes,no}).")

    def run_round(scan_layers: List[int], out_path: str):
        results = []
        for layer_idx in scan_layers:
            for h in tqdm(range(n_heads), desc=f"Scan L{layer_idx}", unit="head"):
                handles = install_head_mask_hooks(model, {int(layer_idx): [int(h)]}, keep_mode=args.keep_mode)

                eff_red_sum = 0.0
                base_change_sum = 0.0
                n = 0
                for E_nc, E_ic, p_nc, p_ic, gold in samples:
                    sc_ic_ab = score_nexttoken_logps(model, tok, p_ic, ids)
                    sc_nc_ab = score_nexttoken_logps(model, tok, p_nc, ids)

                    E_ic_ab = compute_E_weighted(ms, ws, sc_ic_ab, gold)
                    E_nc_ab = compute_E_weighted(ms, ws, sc_nc_ab, gold)

                    effect = E_ic - E_nc
                    effect_ab = E_ic_ab - E_nc

                    eff_red_sum += (abs(effect) - abs(effect_ab))
                    base_change_sum += abs(E_nc_ab - E_nc)
                    n += 1

                remove_hooks(handles)

                results.append({
                    "layer": int(layer_idx),
                    "head": int(h),
                    "mean_abs_effect_reduction": eff_red_sum / max(n, 1),
                    "mean_abs_base_change": base_change_sum / max(n, 1),
                })

        results.sort(key=lambda r: (-(r["mean_abs_effect_reduction"]), r["mean_abs_base_change"]))
        out = {
            "data": args.data,
            "fmt": args.fmt,
            "model": args.model,
            "position": args.position,
            "include_context": bool(args.include_context),
            "enable_thinking": bool(args.enable_thinking),
            "keep_mode": args.keep_mode,
            "metrics": ms,
            "weights": ws,
            "n_samples": len(samples),
            "scan_layers": scan_layers,
            "top20": results[:20],
            "saved": out_path,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    if args.plan:
        _, round_layers = load_plan_layers(args.plan, args.round)
        Path(args.plan_out_dir).mkdir(parents=True, exist_ok=True)

        all_rounds = []
        for rname, scan_layers in round_layers:
            out_path = str(Path(args.plan_out_dir) / f"head_scan_{rname}.json")
            run_round(scan_layers, out_path)
            all_rounds.append({"round": rname, "result_path": out_path})

        summary_path = str(Path(args.plan_out_dir) / "head_scan_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "data": args.data,
                "fmt": args.fmt,
                "model": args.model,
                "position": args.position,
                "include_context": bool(args.include_context),
                "enable_thinking": bool(args.enable_thinking),
                "keep_mode": args.keep_mode,
                "metrics": ms,
                "weights": ws,
                "max_examples": args.max_examples,
                "plan": args.plan,
                "rounds_ran": [x["round"] for x in all_rounds],
                "round_outputs": all_rounds,
                "saved": summary_path,
            }, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved per-round results under: {args.plan_out_dir}")
        print(f"[OK] Saved summary: {summary_path}")
        return

    if not args.layers:
        raise SystemExit("Either provide --plan, or use manual --layers like '32,33,34'.")

    scan_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_path = "head_scan_manual.json"
    run_round(scan_layers, out_path)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()
