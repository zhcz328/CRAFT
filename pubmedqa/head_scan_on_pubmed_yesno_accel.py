import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from head_scan_on_pubmed_yesno import (
    _get_num_heads,
    compute_effect,
    install_head_mask_hooks,
    load_plan_layers,
    parse_metrics_and_weights,
    remove_hooks,
)
from yesno_utils import (
    NO,
    YES,
    get_field,
    get_prompt_profile,
    invert_yes_no,
    load_records,
    logprob_continuation,
    make_conflict_prompt,
    make_nc_prompt,
    normalize_context,
    wrap_as_chat,
)


def _move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _tokenize_continuation(tok, formatted_prompt: str, continuation: str) -> torch.Tensor:
    enc_prompt = tok(formatted_prompt, add_special_tokens=False, return_tensors="pt")
    enc_full = tok(formatted_prompt + continuation, add_special_tokens=False, return_tensors="pt")
    prompt_len = int(enc_prompt["input_ids"].shape[1])
    return enc_full["input_ids"][0, prompt_len:]


@torch.inference_mode()
def _build_prompt_cache(model, tok, prompt: str, enable_thinking: bool, prompt_profile: str = "default"):
    formatted = wrap_as_chat(tok, prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
    enc = tok(formatted, add_special_tokens=False, return_tensors="pt")
    enc = _move_to_device(enc, model.device)
    out = model(**enc, use_cache=True)
    return {
        "formatted": formatted,
        "attention_mask": enc.get("attention_mask"),
        "past_key_values": out.past_key_values,
        "prompt_last_logits": out.logits[:, -1, :],
    }


@torch.inference_mode()
def _logprob_from_cache(model, tok, cache, continuation: str, enable_thinking: bool) -> float:
    cont_ids = _tokenize_continuation(tok, cache["formatted"], continuation)
    if cont_ids.numel() == 0:
        return float("-inf")

    cont_ids = cont_ids.to(model.device)
    first_log_probs = torch.log_softmax(cache["prompt_last_logits"], dim=-1)
    total = float(first_log_probs[0, int(cont_ids[0])])
    if cont_ids.numel() == 1:
        return total

    # The yes/no candidates are one token for the intended Qwen/Llama setups.
    # This path keeps the scorer correct if a tokenizer splits them.
    try:
        input_ids = cont_ids[:-1].unsqueeze(0)
        prompt_mask = cache.get("attention_mask")
        if prompt_mask is None:
            prompt_len = int(cache["prompt_last_logits"].shape[1])
            prompt_mask = torch.ones((1, prompt_len), dtype=torch.long, device=model.device)
        cont_mask = torch.ones_like(input_ids, device=model.device)
        attention_mask = torch.cat([prompt_mask, cont_mask], dim=1)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache["past_key_values"],
            use_cache=False,
        )
        log_probs = torch.log_softmax(out.logits, dim=-1)
        for i, tok_id in enumerate(cont_ids[1:]):
            total += float(log_probs[0, i, int(tok_id)])
        return total
    except Exception:
        # Conservative fallback for models whose cache API needs extra fields.
        prompt = cache["formatted"]
        if hasattr(tok, "apply_chat_template"):
            # logprob_continuation expects an unformatted user prompt, so this
            # fallback is intentionally rare. Returning -inf would bias results.
            raise
        return logprob_continuation(model, tok, prompt, continuation, enable_thinking)


@torch.inference_mode()
def score_yes_no_fast(
    model,
    tok,
    prompt: str,
    enable_thinking: bool,
    prompt_profile: str = "default",
) -> dict[str, float]:
    cache = _build_prompt_cache(model, tok, prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
    yes_lp = max(
        _logprob_from_cache(model, tok, cache, " yes", enable_thinking),
        _logprob_from_cache(model, tok, cache, "yes", enable_thinking),
    )
    no_lp = max(
        _logprob_from_cache(model, tok, cache, " no", enable_thinking),
        _logprob_from_cache(model, tok, cache, "no", enable_thinking),
    )
    return {YES: yes_lp, NO: no_lp}


def weighted_effect(metrics, weights, scores, gold):
    return sum(weight * compute_effect(metric, scores, gold) for metric, weight in zip(metrics, weights))


def build_samples(args, model, tok, metrics, weights):
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
            prompt_profile=args.prompt_profile,
        )
        sc_nc = score_yes_no_fast(
            model,
            tok,
            nc_prompt,
            enable_thinking=args.enable_thinking,
            prompt_profile=args.prompt_profile,
        )
        sc_ic = score_yes_no_fast(
            model,
            tok,
            ic_prompt,
            enable_thinking=args.enable_thinking,
            prompt_profile=args.prompt_profile,
        )
        e_nc = weighted_effect(metrics, weights, sc_nc, gold)
        e_ic = weighted_effect(metrics, weights, sc_ic, gold)
        samples.append(
            {
                "question": str(question),
                "gold": gold,
                "nc_prompt": nc_prompt,
                "ic_prompt": ic_prompt,
                "base_nc_scores": sc_nc,
                "base_ic_scores": sc_ic,
                "base_nc_effect": e_nc,
                "base_ic_effect": e_ic,
                "base_effect": e_ic - e_nc,
            }
        )
    if not samples:
        raise SystemExit("No usable yes/no samples found.")
    return samples


def _mean_head_effect(model, tok, samples, metrics, weights, include_nc, enable_thinking, prompt_profile):
    effect_reduction_sum = 0.0
    base_change_sum = 0.0
    metric_eff = {metric: 0.0 for metric in metrics}
    metric_base = {metric: 0.0 for metric in metrics}
    records = []

    for sample in samples:
        gold = sample["gold"]
        sc_ic_ab = score_yes_no_fast(
            model,
            tok,
            sample["ic_prompt"],
            enable_thinking=enable_thinking,
            prompt_profile=prompt_profile,
        )
        e_ic_ab = weighted_effect(metrics, weights, sc_ic_ab, gold)

        if include_nc:
            sc_nc_ab = score_yes_no_fast(
                model,
                tok,
                sample["nc_prompt"],
                enable_thinking=enable_thinking,
                prompt_profile=prompt_profile,
            )
            e_nc_ab = weighted_effect(metrics, weights, sc_nc_ab, gold)
            base_change_sum += abs(e_nc_ab - sample["base_nc_effect"])
        else:
            sc_nc_ab = sample["base_nc_scores"]

        effect_ab = e_ic_ab - sample["base_nc_effect"]
        effect_reduction_sum += abs(sample["base_effect"]) - abs(effect_ab)

        for metric in metrics:
            base_nc_m = compute_effect(metric, sample["base_nc_scores"], gold)
            base_ic_m = compute_effect(metric, sample["base_ic_scores"], gold)
            ab_ic_m = compute_effect(metric, sc_ic_ab, gold)
            if include_nc:
                ab_nc_m = compute_effect(metric, sc_nc_ab, gold)
            else:
                ab_nc_m = base_nc_m
            metric_eff[metric] += abs(base_ic_m - base_nc_m) - abs(ab_ic_m - base_nc_m)
            metric_base[metric] += abs(ab_nc_m - base_nc_m)

        records.append(
            {
                "question": sample["question"],
                "gold": gold,
                "wrong": invert_yes_no(gold),
                "base_nc": sample["base_nc_scores"],
                "base_ic": sample["base_ic_scores"],
                "ablated_nc": sc_nc_ab,
                "ablated_ic": sc_ic_ab,
                "base_effect_scalar": sample["base_effect"],
                "ablated_effect_scalar": effect_ab,
            }
        )

    n = max(1, len(samples))
    return {
        "mean_abs_effect_reduction": effect_reduction_sum / n,
        "mean_abs_base_change": base_change_sum / n,
        "metric_mean_abs_effect_reduction": {metric: value / n for metric, value in metric_eff.items()},
        "metric_mean_abs_base_change": {metric: value / n for metric, value in metric_base.items()},
        "records": records,
    }


def _coarse_samples(samples, limit: int, seed: int):
    if limit <= 0 or limit >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(samples)), limit))
    return [samples[i] for i in idxs]


def _scan_heads(model, tok, scan_layers, n_heads, samples, metrics, weights, args, include_nc, desc):
    rows = []
    for layer_idx in scan_layers:
        for head_idx in tqdm(range(n_heads), desc=f"{desc} L{layer_idx}", unit="head"):
            handles = install_head_mask_hooks(model, {int(layer_idx): [int(head_idx)]}, keep_mode=args.keep_mode)
            try:
                summary = _mean_head_effect(
                    model,
                    tok,
                    samples,
                    metrics,
                    weights,
                    include_nc=include_nc,
                    enable_thinking=args.enable_thinking,
                    prompt_profile=args.prompt_profile,
                )
            finally:
                remove_hooks(handles)
            item = {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "mean_abs_effect_reduction": summary["mean_abs_effect_reduction"],
                "mean_abs_base_change": summary["mean_abs_base_change"],
                "metric_mean_abs_effect_reduction": summary["metric_mean_abs_effect_reduction"],
                "metric_mean_abs_base_change": summary["metric_mean_abs_base_change"],
            }
            if args.save_records:
                item["records"] = summary["records"]
            rows.append(item)
    rows.sort(key=lambda row: (-(row["mean_abs_effect_reduction"]), row["mean_abs_base_change"], row["layer"], row["head"]))
    return rows


def run_round(args, model, tok, samples, metrics, weights, n_heads, round_name, scan_layers, out_path):
    coarse = _coarse_samples(samples, args.coarse_limit, args.coarse_seed)
    coarse_rows = _scan_heads(
        model,
        tok,
        scan_layers,
        n_heads,
        coarse,
        metrics,
        weights,
        args,
        include_nc=args.coarse_with_nc,
        desc=f"Coarse {round_name}",
    )

    if args.coarse_topk > 0:
        candidate_rows = coarse_rows[: min(args.coarse_topk, len(coarse_rows))]
    else:
        candidate_rows = coarse_rows
    candidate_heads = [(row["layer"], row["head"]) for row in candidate_rows]

    refine_rows = []
    for layer_idx, head_idx in tqdm(candidate_heads, desc=f"Refine {round_name}", unit="head"):
        handles = install_head_mask_hooks(model, {int(layer_idx): [int(head_idx)]}, keep_mode=args.keep_mode)
        try:
            summary = _mean_head_effect(
                model,
                tok,
                samples,
                metrics,
                weights,
                include_nc=True,
                enable_thinking=args.enable_thinking,
                prompt_profile=args.prompt_profile,
            )
        finally:
            remove_hooks(handles)
        refine_rows.append(
            {
                "layer": int(layer_idx),
                "head": int(head_idx),
                "mean_abs_effect_reduction": summary["mean_abs_effect_reduction"],
                "mean_abs_base_change": summary["mean_abs_base_change"],
                "metric_mean_abs_effect_reduction": summary["metric_mean_abs_effect_reduction"],
                "metric_mean_abs_base_change": summary["metric_mean_abs_base_change"],
                **({"records": summary["records"]} if args.save_records else {}),
            }
        )
    refine_rows.sort(key=lambda row: (-(row["mean_abs_effect_reduction"]), row["mean_abs_base_change"], row["layer"], row["head"]))

    payload = {
        "data": args.data,
        "model": args.model,
        "prompt_profile": args.prompt_profile,
        "position": args.position,
        "keep_mode": args.keep_mode,
        "metrics": metrics,
        "weights": weights,
        "n_samples": len(samples),
        "scan_layers": [int(x) for x in scan_layers],
        "round": round_name,
        "coarse": {
            "coarse_limit": args.coarse_limit,
            "coarse_seed": args.coarse_seed,
            "coarse_topk": args.coarse_topk,
            "coarse_with_nc": bool(args.coarse_with_nc),
            "n_coarse_samples": len(coarse),
            "n_all_heads": len(scan_layers) * n_heads,
            "n_refined_heads": len(refine_rows),
        },
        "coarse_top20": coarse_rows[:20],
        "top20": refine_rows[:20],
        "results": refine_rows,
        "saved": str(out_path),
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Accelerated attention head scan for PubMedQA yes/no NC->IC prompts.")
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
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--round", default=None)
    ap.add_argument("--plan_out_dir", default="result_yesno/headscan_pubmed")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--coarse_limit", type=int, default=128)
    ap.add_argument("--coarse_topk", type=int, default=36)
    ap.add_argument("--coarse_seed", type=int, default=42)
    ap.add_argument("--coarse_with_nc", action="store_true")
    ap.add_argument("--save_records", action="store_true")
    args = ap.parse_args()
    args.prompt_profile = get_prompt_profile(args.model)

    metrics, weights = parse_metrics_and_weights(args.metrics, args.weights)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation="eager",
    )
    model.eval()
    n_heads = _get_num_heads(model)
    if not n_heads:
        raise SystemExit("Cannot infer number of attention heads from model config.")

    samples = build_samples(args, model, tok, metrics, weights)

    if args.plan:
        _, round_layers = load_plan_layers(args.plan, args.round)
        out_dir = Path(args.plan_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        round_outputs = []
        for round_name, scan_layers in round_layers:
            out_path = out_dir / f"head_scan_{round_name}.json"
            run_round(args, model, tok, samples, metrics, weights, n_heads, round_name, scan_layers, out_path)
            round_outputs.append({"round": round_name, "result_path": str(out_path)})
        summary_path = out_dir / "head_scan_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "data": args.data,
                    "model": args.model,
                    "prompt_profile": args.prompt_profile,
                    "position": args.position,
                    "metrics": metrics,
                    "weights": weights,
                    "plan": args.plan,
                    "accelerated": True,
                    "coarse_limit": args.coarse_limit,
                    "coarse_topk": args.coarse_topk,
                    "round_outputs": round_outputs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] Saved accelerated per-round results under: {out_dir}")
        return

    if not args.layers:
        raise SystemExit("Either provide --plan, or use manual --layers like '12,13,14'.")
    scan_layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    run_round(args, model, tok, samples, metrics, weights, n_heads, "manual", scan_layers, "head_scan_manual.json")


if __name__ == "__main__":
    main()
