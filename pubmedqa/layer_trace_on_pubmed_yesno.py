import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from yesno_utils import (
    NO,
    YES,
    get_field,
    get_prompt_profile,
    invert_yes_no,
    load_records,
    make_conflict_prompt,
    make_nc_prompt,
    normalize_context,
    score_yes_no,
    wrap_as_chat,
)


def _get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("Cannot find transformer layers on this model.")


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


def patch_layer_lastk(model, layer_idx: int, replacement_tail: torch.Tensor):
    layer = _get_layers(model)[layer_idx]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        kk = replacement_tail.size(1)
        hs2 = hs.clone()
        hs2[:, -kk:, :] = replacement_tail.to(hs2.device, dtype=hs2.dtype)
        if isinstance(out, (tuple, list)):
            out = list(out)
            out[0] = hs2
            return tuple(out)
        return hs2

    return layer.register_forward_hook(hook)


def patch_layer_all(model, layer_idx: int, replacement_states: torch.Tensor):
    layer = _get_layers(model)[layer_idx]

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        hs2 = hs.clone()
        rep = replacement_states.to(hs2.device, dtype=hs2.dtype)
        kk = min(rep.size(1), hs2.size(1))
        hs2[:, -kk:, :] = rep[:, -kk:, :]
        if isinstance(out, (tuple, list)):
            out = list(out)
            out[0] = hs2
            return tuple(out)
        return hs2

    return layer.register_forward_hook(hook)


def plot_layer_scores(scores, out_path, title):
    plt.figure()
    plt.plot(range(len(scores)), scores)
    plt.xlabel("layer_idx")
    plt.ylabel("mean score")
    plt.title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_metrics_arg(raw):
    if not raw:
        return ["yesno_margin", "follow_conflict"]
    allowed = {"yesno_margin", "follow_conflict"}
    metrics = [item.strip() for item in raw.split(",") if item.strip()]
    for metric in metrics:
        if metric not in allowed:
            raise SystemExit(f"Unknown metric '{metric}'. Allowed: {sorted(allowed)}")
    return metrics


def parse_weights(raw, n):
    if not raw:
        return [1.0 / n] * n
    parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != n:
        raise SystemExit(f"--scan_plan_weights expects {n} weights, got {len(parts)}")
    total = sum(parts)
    if total == 0:
        raise SystemExit("--scan_plan_weights sum to 0")
    return [value / total for value in parts]


def combine_scores(layer_score_mean, metrics, weights):
    n_layers = len(next(iter(layer_score_mean.values())))
    combined = [0.0] * n_layers
    for metric, weight in zip(metrics, weights):
        for idx in range(n_layers):
            combined[idx] += weight * layer_score_mean[metric][idx]
    return combined


def build_scan_plan(scores, tail_k=4, preplateau_width=7, topk_fallback=8):
    n_layers = len(scores)
    peak = max(range(n_layers), key=lambda idx: scores[idx])
    start = max(0, peak - preplateau_width)
    round1 = list(range(start, peak + 1))
    round2 = list(range(max(0, n_layers - tail_k), n_layers))
    topk = sorted(range(n_layers), key=lambda idx: scores[idx], reverse=True)[:topk_fallback]
    return {
        "method": "yesno_peak_window",
        "round1_preplateau": sorted(set(round1)),
        "round1_topk_fallback": sorted(set(topk)),
        "round2_tail": sorted(set(round2)),
        "tail_k": tail_k,
        "preplateau_width": preplateau_width,
        "topk_fallback": topk_fallback,
    }


def effect_yesno_margin(scores):
    return scores[YES] - scores[NO]


def effect_follow_conflict(scores, gold):
    return scores[invert_yes_no(gold)] - scores[gold]


def main():
    ap = argparse.ArgumentParser(description="Layer trace for PubMedQA yes/no NC->IC prompts.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet", "auto"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--patch_k", type=int, default=8)
    ap.add_argument("--patch_all_tokens", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="result_yesno/layer_trace/trace.json")
    ap.add_argument("--plot_prefix", default="result_yesno/layer_trace/trace")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--metrics", default="yesno_margin,follow_conflict")
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="pubid")
    ap.add_argument("--scan_plan_out", default="")
    ap.add_argument("--scan_plan_metrics", default="")
    ap.add_argument("--scan_plan_weights", default="")
    ap.add_argument("--tail_k", type=int, default=4)
    ap.add_argument("--preplateau_width", type=int, default=7)
    ap.add_argument("--topk_fallback", type=int, default=8)
    args = ap.parse_args()
    prompt_profile = get_prompt_profile(args.model)

    metrics = parse_metrics_arg(args.metrics)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
    )
    model.eval()

    data = load_records(args.data, fmt=args.fmt)
    n_layers = len(_get_layers(model))
    sums = {metric: torch.zeros(n_layers, dtype=torch.float64) for metric in metrics}
    per_example = []
    n_samples = 0
    patch_mode = "all_tokens" if args.patch_all_tokens else f"last_{args.patch_k}"

    for ex in tqdm(data, desc="Layer tracing yes/no", unit="ex"):
        if args.limit > 0 and n_samples >= args.limit:
            break
        question = get_field(ex, args.q_field)
        context_raw = get_field(ex, args.c_field)
        gold = get_field(ex, args.y_field)
        if question is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            continue

        context = normalize_context(context_raw)
        nc_user = make_nc_prompt(str(question), allow_unknown=False)
        ic_user = make_conflict_prompt(
            str(question),
            context,
            gold,
            args.position,
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )

        sc_nc = score_yes_no(model, tok, nc_user, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        sc_ic = score_yes_no(model, tok, ic_user, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)

        base_effects = {}
        conflict_effects = {}
        effect = {}
        for metric in metrics:
            if metric == "yesno_margin":
                base_effects[metric] = effect_yesno_margin(sc_nc)
                conflict_effects[metric] = effect_yesno_margin(sc_ic)
            else:
                base_effects[metric] = effect_follow_conflict(sc_nc, gold)
                conflict_effects[metric] = effect_follow_conflict(sc_ic, gold)
            effect[metric] = conflict_effects[metric] - base_effects[metric]

        if args.patch_all_tokens:
            base_cache, handles = capture_layer_outputs_all(model)
        else:
            base_cache, handles = capture_layer_outputs_lastk(model, args.patch_k)
        _ = score_yes_no(model, tok, nc_user, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        for handle in handles:
            handle.remove()

        layer_scores_this = {metric: [] for metric in metrics}
        for layer_idx in range(n_layers):
            if args.patch_all_tokens:
                patch_handle = patch_layer_all(model, layer_idx, base_cache[layer_idx])
            else:
                patch_handle = patch_layer_lastk(model, layer_idx, base_cache[layer_idx])
            patched_scores = score_yes_no(model, tok, ic_user, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
            patch_handle.remove()

            for metric in metrics:
                if metric == "yesno_margin":
                    patched_effect = effect_yesno_margin(patched_scores)
                else:
                    patched_effect = effect_follow_conflict(patched_scores, gold)
                effect_p = patched_effect - base_effects[metric]
                score = abs(effect[metric]) - abs(effect_p)
                sums[metric][layer_idx] += score
                layer_scores_this[metric].append(float(score))

        per_example.append(
            {
                "id": get_field(ex, args.id_field),
                "gold": gold,
                "conflict": invert_yes_no(gold),
                "position": args.position,
                "patch_mode": patch_mode,
                "scores_nc": sc_nc,
                "scores_ic": sc_ic,
                "effect": effect,
                "layer_score": layer_scores_this,
            }
        )
        n_samples += 1

    layer_score_mean = {metric: (sums[metric] / max(n_samples, 1)).tolist() for metric in metrics}
    top_layers = {
        metric: sorted(range(n_layers), key=lambda idx: layer_score_mean[metric][idx], reverse=True)[:10]
        for metric in metrics
    }
    payload = {
        "scheme": "pubmedqa_yesno_nc_to_conflict",
        "n_samples": n_samples,
        "n_layers": n_layers,
        "position": args.position,
        "patch_mode": patch_mode,
        "metrics": metrics,
        "layer_score_mean": layer_score_mean,
        "top_layers_by_score": top_layers,
        "per_example": per_example,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for metric in metrics:
        plot_layer_scores(
            layer_score_mean[metric],
            f"{args.plot_prefix}_{metric}.png",
            title=f"PubMedQA yes/no layer trace | pos={args.position} | patch={patch_mode} | metric={metric}",
        )

    if args.scan_plan_out:
        if args.scan_plan_metrics:
            plan_metrics = [item.strip() for item in args.scan_plan_metrics.split(",") if item.strip()]
        else:
            plan_metrics = [metrics[0]]
        weights = parse_weights(args.scan_plan_weights, len(plan_metrics))
        plan_scores = combine_scores(layer_score_mean, plan_metrics, weights)
        scan_plan = build_scan_plan(
            plan_scores,
            tail_k=args.tail_k,
            preplateau_width=args.preplateau_width,
            topk_fallback=args.topk_fallback,
        )
        write_payload = {
            "position": args.position,
            "patch_mode": patch_mode,
            "plan_scores_source": {
                "metrics": plan_metrics,
                "weights": {metric: weight for metric, weight in zip(plan_metrics, weights)},
            },
            "scan_plan": scan_plan,
        }
        scan_plan_path = Path(args.scan_plan_out)
        scan_plan_path.parent.mkdir(parents=True, exist_ok=True)
        scan_plan_path.write_text(json.dumps(write_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved scan plan: {scan_plan_path}")

    print(f"[OK] Saved trace: {out_path}")
    for metric in metrics:
        print(f"[OK] Plot: {args.plot_prefix}_{metric}.png | top layers: {top_layers[metric]}")


if __name__ == "__main__":
    main()
