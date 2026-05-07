# layer_trace.py
# Layer tracing with last-k or all-token patching + per-layer score plot
# score_i = |effect| - |effect_p|
#
# Usage example:
# python layer_trace.py \
#   --pairs kept_pairs_a12_b12.jsonl \
#   --model /path/to/Qwen3-4B \
#   --position before_question \
#   --patch_k 8 \
#   --max_pairs 57 \
#   --out layer_trace.json \
#   --plot_out layer_trace_scores.png

import json
import re
import argparse
from typing import List, Tuple
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib.pyplot as plt


YES, NO = "yes", "no"
SYSTEM = "You are a medical QA verifier. Answer ONLY with 'Yes' or 'No'. No other words."

EVID_TMPL = """EVIDENCE (may be incorrect):
- According to current clinical guidelines, this recommendation {align} with guidelines.
END EVIDENCE
"""
def detect_rising_layers(scores, peak=None, tolerate=1, min_len=3, max_keep=10):
    """
    返回“上升期层”索引列表（用于 round0）。
    思路：在 peak 左侧回溯，找到一段总体上升区间，允许 tolerate 次非上升；
          然后在该区间内按 delta=score[l]-score[l-1] 选最陡的 max_keep 个层。
    """
    n = len(scores)
    if n < 3:
        return []

    if peak is None:
        peak = max(range(n), key=lambda i: scores[i])

    # peak 太靠前就没啥“上升期”
    if peak <= 1:
        return []

    # 计算一阶差分
    delta = [scores[i] - scores[i-1] for i in range(1, n)]  # delta[i-1] = scores[i]-scores[i-1]

    # 从 peak 往左回溯，允许 tolerate 次 delta<=0
    bad = 0
    l = peak
    # 我们检查的是 l 的“增长”对应 delta[l-1]
    while l >= 2:
        if delta[l-1] <= 0:
            bad += 1
            if bad > tolerate:
                break
        l -= 1
    start = max(1, l)  # 至少从 1 开始（因为要用到 delta）
    end = peak

    # 区间长度太短就不要
    if end - start + 1 < min_len:
        return []

    # 在 [start..end] 内挑“最陡”的层（按 delta 排序）
    candidates = list(range(start, end + 1))
    candidates.sort(key=lambda idx: delta[idx-1], reverse=True)  # idx 对应 delta[idx-1]
    keep = candidates[:max_keep]
    return sorted(set(keep))


def read_pairs(path: str) -> List[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def make_evidence(label_yes: bool) -> str:
    align = "DOES align" if label_yes else "DOES NOT align"
    return EVID_TMPL.format(align=align).strip()


def inject_evidence(base_prompt: str, evidence: str, position: str) -> str:
    ev = evidence.strip() + "\n"
    if position == "prefix":
        return ev + "\n" + base_prompt

    if position == "before_question":
        m = list(re.finditer(r"(?im)^\s*question\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    if position == "before_answer":
        m = list(re.finditer(r"(?im)^\s*answer\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    raise ValueError(f"Unknown position: {position}")


def wrap_as_chat(tok, user_text: str, enable_thinking: bool) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        # older transformers without enable_thinking
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score_yes_no_fast(model, tok, prompt_text: str) -> Tuple[str, float]:
    """
    Returns (pred, delta) where:
      delta = logP(Yes) - logP(No) at the NEXT token distribution (logits[:, -1, :])
    """
    dev = next(model.parameters()).device
    enc = tok(prompt_text, return_tensors="pt")
    enc = {k: v.to(dev) for k, v in enc.items()}

    out = model(**enc)
    logits = out.logits[:, -1, :]  # next-token distribution
    logp = torch.log_softmax(logits, dim=-1)

    yid = tok(" Yes", add_special_tokens=False).input_ids
    nid = tok(" No", add_special_tokens=False).input_ids
    if len(yid) != 1 or len(nid) != 1:
        raise RuntimeError('" Yes"/" No" not single-token; need continuation scoring.')

    yes_lp = logp[0, yid[0]].item()
    no_lp = logp[0, nid[0]].item()
    delta = yes_lp - no_lp
    pred = YES if delta >= 0 else NO
    return pred, delta


def _get_layers(model):
    return model.model.layers if hasattr(model, "model") else model.layers


def capture_layer_outputs(model, k: int, patch_all_token: bool = False):
    """
    Cache each layer output.
    - patch_all_token=False: keep only the last-k tokens.
    - patch_all_token=True: keep the full layer output.
    """
    layers = _get_layers(model)
    n_layers = len(layers)
    cache = [None] * n_layers
    handles = []

    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out  # [B,T,H]
                if patch_all_token:
                    cache[ii] = hs.detach()
                else:
                    kk = min(k, hs.size(1))
                    cache[ii] = hs[:, -kk:, :].detach()  # [B,kk,H]
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    return cache, handles


def patch_layer_output(model, layer_idx: int, base_states: torch.Tensor):
    """
    Patch the overlapping tail of the current layer output using cached base states.
    This supports both:
    - cached last-k states
    - cached full-sequence states (all-token patch)

    When base/conflict sequence lengths differ, only the overlapping tail is patched.
    """
    layers = _get_layers(model)
    layer = layers[layer_idx]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out  # [B,T,H]
        kk = min(base_states.size(1), hs.size(1))
        hs2 = hs.clone()
        hs2[:, -kk:, :] = base_states[:, -kk:, :].to(hs2.device, dtype=hs2.dtype)

        if isinstance(out, (tuple, list)):
            out = list(out)
            out[0] = hs2
            return tuple(out)
        return hs2

    return layer.register_forward_hook(hook)


def plot_layer_scores(scores: List[float], plot_out: str, title: str = ""):
    x = list(range(len(scores)))
    plt.figure()
    plt.plot(x, scores)
    plt.xlabel("layer_idx")
    plt.ylabel("score_i = |effect| - |effect_p|")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=200)
    plt.close()


def _unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _append_round_dedup(rounds, seen_layers, name, layers, reason):
    uniq_layers = _unique_keep_order(layers)
    uniq_layers = [l for l in uniq_layers if l not in seen_layers]
    if not uniq_layers:
        return
    rounds.append({"name": name, "layers": uniq_layers, "reason": reason})
    seen_layers.update(uniq_layers)
def build_scan_plan(scores, plateau_eps=0.6, plateau_min_len=6,
                    preplateau_width=7, tail_k=4, plateau_min_score_frac=0.8, rising_tolerate=1, rising_min_len=3, rising_max_keep=10):
    """
    scores: list[float], length = n_layers
    Heuristic:
      1) find earliest plateau start s where diffs are small for >= plateau_min_len
      2) round1: [max(0, s-preplateau_width) .. min(n-1, s+1)]
      3) round2: last tail_k layers (n-tail_k .. n-1)
    Fallback: if plateau not found -> round1 = top-8 by score, round2 = last tail_k
    """
    n = len(scores)
    diffs = [abs(scores[i+1] - scores[i]) for i in range(n-1)]
    peak = max(range(n), key=lambda i: scores[i])
    rising = detect_rising_layers(
        scores,
        peak=peak,
        tolerate=rising_tolerate,
        min_len=rising_min_len,
        max_keep=rising_max_keep,
    )

    plateau_start = None
    max_score = max(scores)
    min_plateau_score = plateau_min_score_frac * max_score  # 或传参进函数

    plateau_start = None
    for s in range(n - plateau_min_len):
        if scores[s] < min_plateau_score:
            continue
        if all(d <= plateau_eps for d in diffs[s:s+plateau_min_len]):
            plateau_start = s
            break

    if plateau_start is None:
        top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:8]
        top = sorted(top)
        rounds = []
        seen_layers = set()
        if rising:
            _append_round_dedup(rounds, seen_layers, "round0_rising", rising, f"rising_before_peak={peak}")

        _append_round_dedup(rounds, seen_layers, "round1_top8_fallback", top, "no_plateau_found")
        _append_round_dedup(rounds, seen_layers, "round2_tail", list(range(max(0, n-tail_k), n)), "tail_layers")

        return {"n_layers": n, "plateau_start": None, "rounds": rounds}

    s = plateau_start
    r1_lo = max(0, s - preplateau_width)
    r1_hi = min(n-1, s + 1)  # include a couple layers into plateau edge
    round1 = list(range(r1_lo, r1_hi + 1))
    round2 = list(range(max(0, n - tail_k), n))

    rounds = []
    seen_layers = set()
    if rising:
        _append_round_dedup(rounds, seen_layers, "round0_rising", rising, f"rising_before_peak={peak}")

    _append_round_dedup(rounds, seen_layers, "round1_preplateau", round1, f"preplateau_window around start={s}")

    # 避免 round2 和 round1 全重叠
    if any(l not in set(round1) for l in round2):
        _append_round_dedup(rounds, seen_layers, "round2_tail", round2, "tail_layers")
    return {"n_layers": n, "plateau_start": s, "rounds": rounds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="kept_pairs jsonl with correct/wrong prompts")
    ap.add_argument("--model", required=True, help="local HF model path")
    ap.add_argument(
        "--device",
        default="auto",
        help="Model placement for HF loading. Use auto for sharded placement, or a single device like cuda:0 / cuda:1 / cpu.",
    )
    ap.add_argument("--position", default="before_question",
                    choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--patch_k", type=int, default=8, help="Patch last-k tokens per layer (default 8)")
    ap.add_argument(
        "--patch_all_token",
        action="store_true",
        help="Patch all tokens per layer; if lengths differ, patch the overlapping tail.",
    )
    ap.add_argument("--max_pairs", type=int, default=57, help="number of pairs to run (each pair -> 2 samples)")
    ap.add_argument("--out", default="result/layer_trace.json")
    ap.add_argument("--plot_out", default="result/layer_trace_scores.png")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--plan_out", default="result/scan_plan.json")
    ap.add_argument("--plateau_eps", type=float, default=0.6, help="plateau detection eps on score diff")
    ap.add_argument("--plateau_min_len", type=int, default=6, help="min consecutive layers for plateau")
    ap.add_argument("--preplateau_width", type=int, default=7, help="how many layers before plateau start to scan (round1)")
    ap.add_argument("--tail_k", type=int, default=4, help="how many last layers to scan (round2)")
    ap.add_argument("--plateau_min_score_frac", type=float, default=0.8,
                help="plateau region must have score >= frac * max_score (default 0.8)")
    ap.add_argument("--rising_tolerate", type=int, default=1, help="allow small pullbacks when detecting rising phase")
    ap.add_argument("--rising_min_len", type=int, default=3, help="min length of rising segment")
    ap.add_argument("--rising_max_keep", type=int, default=10, help="max layers kept for rising round0")

    args = ap.parse_args()

    if args.patch_k <= 0:
        raise ValueError("--patch_k must be positive")

    for out_path in [args.out, args.plot_out, args.plan_out]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device_map = "auto" if args.device == "auto" else {"": args.device}

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype
    )
    model.eval()

    pairs = read_pairs(args.pairs)[:args.max_pairs]
    layers = _get_layers(model)
    n_layers = len(layers)

    layer_scores_sum = torch.zeros(n_layers, dtype=torch.float64)
    n_samples = 0

    for rec in tqdm(pairs, desc="Layer tracing", unit="pair"):
        for side in ["correct", "wrong"]:
            base_prompt = rec[side]["prompt"]

            gold_yes = (side == "correct")
            conflict_yes = (not gold_yes)

            # Build prompts
            p_base = wrap_as_chat(tok, base_prompt, args.enable_thinking)

            conflict_user = inject_evidence(base_prompt, make_evidence(conflict_yes), args.position)
            p_con = wrap_as_chat(tok, conflict_user, args.enable_thinking)

            # Compute baseline/conflict deltas
            _, d_base = score_yes_no_fast(model, tok, p_base)
            _, d_con = score_yes_no_fast(model, tok, p_con)
            effect = d_con - d_base

            # Cache BASE per-layer states according to patch mode.
            base_cache, h1 = capture_layer_outputs(
                model,
                args.patch_k,
                patch_all_token=args.patch_all_token,
            )
            _ = score_yes_no_fast(model, tok, p_base)
            for hh in h1:
                hh.remove()

            # Patch each layer during CONFLICT run.
            for i in range(n_layers):
                ph = patch_layer_output(model, i, base_cache[i])
                _, d_patched = score_yes_no_fast(model, tok, p_con)
                ph.remove()

                effect_p = d_patched - d_base
                layer_scores_sum[i] += (abs(effect) - abs(effect_p))

            n_samples += 1

    layer_scores_mean = (layer_scores_sum / max(n_samples, 1)).tolist()
    top_layers = sorted(range(n_layers), key=lambda i: layer_scores_mean[i], reverse=True)[:10]

    res = {
        "n_samples": n_samples,
        "n_pairs": len(pairs),
        "position": args.position,
        "patch_k": args.patch_k,
        "patch_all_token": args.patch_all_token,
        "top_layers_by_score": top_layers,
        "layer_score_mean": layer_scores_mean,
        "saved": args.out,
        "plot_saved": args.plot_out,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    plot_layer_scores(
        layer_scores_mean,
        args.plot_out,
        title=(
            f"Layer tracing scores (position={args.position}, "
            f"{'all_tokens_tail_overlap' if args.patch_all_token else f'k={args.patch_k}'})"
        )
    )
    plan = build_scan_plan(
    layer_scores_mean,
    plateau_eps=args.plateau_eps,
    plateau_min_len=args.plateau_min_len,
    preplateau_width=args.preplateau_width,
    tail_k=args.tail_k,
    plateau_min_score_frac=args.plateau_min_score_frac,
    rising_tolerate=args.rising_tolerate,
    rising_min_len=args.rising_min_len,
    rising_max_keep=args.rising_max_keep,
    )

    plan.update({
        "position": args.position,
        "patch_k": args.patch_k,
        "patch_all_token": args.patch_all_token,
        "n_samples": n_samples,
        "note": "This plan is for head_scan.py",
    })

    with open(args.plan_out, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    print({"plan_saved": args.plan_out, **plan})

    print(res)


if __name__ == "__main__":
    main()
