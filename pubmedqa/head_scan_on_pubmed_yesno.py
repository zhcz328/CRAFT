import argparse
import json
from pathlib import Path

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
)


def _get_num_heads(model):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)


def _find_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate transformer layers.")


def _find_self_attn_modules(model):
    layers = _find_layers(model)
    layer2attn = {}
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer2attn[idx] = layer.self_attn
        elif hasattr(layer, "attn"):
            layer2attn[idx] = layer.attn
    if not layer2attn:
        raise RuntimeError("Cannot find self-attention modules.")
    return layer2attn


def _apply_head_specific_mask(attn_mask, n_heads, heads_to_mask, keep_mode):
    if attn_mask is None or (not torch.is_tensor(attn_mask)) or attn_mask.dim() != 4:
        return attn_mask
    bsz, hmask, tgt, src = attn_mask.shape
    if hmask == 1:
        mask = attn_mask.expand(bsz, n_heads, tgt, src).clone()
    elif hmask == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask
    neg = torch.finfo(mask.dtype).min
    for head_idx in heads_to_mask:
        mask[:, head_idx, :, :] = neg
        if keep_mode == "bos":
            mask[:, head_idx, :, 0] = 0
        else:
            diag_len = min(tgt, src)
            diag = torch.arange(diag_len, device=mask.device)
            mask[:, head_idx, diag, diag] = 0
    return mask


def install_head_mask_hooks(model, layer2heads, keep_mode):
    layer2attn = _find_self_attn_modules(model)
    n_heads = _get_num_heads(model)
    handles = []
    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(head) for head in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, args, kwargs):
                attn_mask = None
                if kwargs is not None and "attention_mask" in kwargs:
                    attn_mask = kwargs["attention_mask"]
                elif len(args) >= 2:
                    attn_mask = args[1]
                new_mask = _apply_head_specific_mask(attn_mask, n_heads, heads_local, keep_mode)
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
                    args[1] = _apply_head_specific_mask(args[1], n_heads, heads_local, keep_mode)
                    return tuple(args)
                return pre_hook_no_kwargs
            handle = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


def parse_metrics_and_weights(metrics_raw, weights_raw):
    metrics = [item.strip() for item in metrics_raw.split(",") if item.strip()]
    allowed = {"yesno_margin", "follow_conflict"}
    for metric in metrics:
        if metric not in allowed:
            raise SystemExit(f"Unknown metric '{metric}'. Allowed: {sorted(allowed)}")
    if not weights_raw:
        return metrics, [1.0 / len(metrics)] * len(metrics)
    weights = [float(item.strip()) for item in weights_raw.split(",") if item.strip()]
    if len(weights) != len(metrics):
        raise SystemExit("weights length must match metrics length")
    total = sum(weights)
    return metrics, [weight / total for weight in weights]


def compute_effect(metric, scores, gold):
    if metric == "yesno_margin":
        return scores[YES] - scores[NO]
    return scores[invert_yes_no(gold)] - scores[gold]


def load_plan_layers(plan_path, round_name=None):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    scan_plan = plan.get("scan_plan", {})
    round_layers = []
    seen = set()
    for key in ("round0_rise", "round1_preplateau", "round1_topk_fallback", "round2_tail"):
        layers = scan_plan.get(key) or []
        if layers:
            clean_layers = []
            for value in layers:
                layer = int(value)
                if layer in seen:
                    continue
                seen.add(layer)
                clean_layers.append(layer)
            if clean_layers:
                round_layers.append((key, clean_layers))
    if round_name is None:
        return plan, round_layers
    for key, layers in round_layers:
        if key == round_name:
            return plan, [(key, layers)]
    raise ValueError(f"round '{round_name}' not found")


def main():
    ap = argparse.ArgumentParser(description="Attention head scan for PubMedQA yes/no NC->IC prompts.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet", "auto"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--keep_mode", default="self", choices=["self", "bos"])
    ap.add_argument("--metrics", default="follow_conflict,yesno_margin")
    ap.add_argument("--weights", default="")
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--max_examples", type=int, default=50)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--round", default=None)
    ap.add_argument("--plan_out_dir", default="result_yesno/headscan_pubmed")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()
    prompt_profile = get_prompt_profile(args.model)

    metrics, weights = parse_metrics_and_weights(args.metrics, args.weights)
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
    n_heads = _get_num_heads(model)
    data = load_records(args.data, fmt=args.fmt)

    samples = []
    for ex in tqdm(data, desc="Build samples", unit="ex"):
        if args.max_examples > 0 and len(samples) >= args.max_examples:
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
        nc_prompt = make_nc_prompt(str(question), allow_unknown=False)
        ic_prompt = make_conflict_prompt(
            str(question),
            context,
            gold,
            args.position,
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )
        sc_nc = score_yes_no(model, tok, nc_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        sc_ic = score_yes_no(model, tok, ic_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        e_nc = sum(weight * compute_effect(metric, sc_nc, gold) for metric, weight in zip(metrics, weights))
        e_ic = sum(weight * compute_effect(metric, sc_ic, gold) for metric, weight in zip(metrics, weights))
        samples.append((e_nc, e_ic, nc_prompt, ic_prompt, gold))

    if not samples:
        raise SystemExit("No usable yes/no samples found.")

    def run_round(scan_layers, out_path):
        results = []
        for layer_idx in scan_layers:
            for head_idx in tqdm(range(n_heads), desc=f"Scan L{layer_idx}", unit="head"):
                handles = install_head_mask_hooks(model, {int(layer_idx): [int(head_idx)]}, keep_mode=args.keep_mode)
                effect_reduction_sum = 0.0
                base_change_sum = 0.0
                n = 0
                for e_nc, e_ic, nc_prompt, ic_prompt, gold in samples:
                    sc_ic_ab = score_yes_no(model, tok, ic_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
                    sc_nc_ab = score_yes_no(model, tok, nc_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
                    e_ic_ab = sum(weight * compute_effect(metric, sc_ic_ab, gold) for metric, weight in zip(metrics, weights))
                    e_nc_ab = sum(weight * compute_effect(metric, sc_nc_ab, gold) for metric, weight in zip(metrics, weights))
                    effect = e_ic - e_nc
                    effect_ab = e_ic_ab - e_nc
                    effect_reduction_sum += abs(effect) - abs(effect_ab)
                    base_change_sum += abs(e_nc_ab - e_nc)
                    n += 1
                remove_hooks(handles)
                results.append(
                    {
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "mean_abs_effect_reduction": effect_reduction_sum / max(1, n),
                        "mean_abs_base_change": base_change_sum / max(1, n),
                    }
                )

        results.sort(key=lambda row: (-(row["mean_abs_effect_reduction"]), row["mean_abs_base_change"]))
        payload = {
            "data": args.data,
            "model": args.model,
            "position": args.position,
            "metrics": metrics,
            "weights": weights,
            "n_samples": len(samples),
            "scan_layers": scan_layers,
            "top20": results[:20],
            "saved": out_path,
        }
        Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    if args.plan:
        _, round_layers = load_plan_layers(args.plan, args.round)
        out_dir = Path(args.plan_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        round_outputs = []
        for round_name, scan_layers in round_layers:
            out_path = out_dir / f"head_scan_{round_name}.json"
            run_round(scan_layers, out_path)
            round_outputs.append({"round": round_name, "result_path": str(out_path)})
        summary_path = out_dir / "head_scan_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "data": args.data,
                    "model": args.model,
                    "position": args.position,
                    "metrics": metrics,
                    "weights": weights,
                    "plan": args.plan,
                    "round_outputs": round_outputs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] Saved per-round results under: {out_dir}")
        return

    if not args.layers:
        raise SystemExit("Either provide --plan, or use manual --layers like '12,13,14'.")
    scan_layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    run_round(scan_layers, "head_scan_manual.json")


if __name__ == "__main__":
    main()
