#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/root/logit_lens/Slake_vqa/image_conflict")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

import layer_trace_slake_mm_current_fixed_fastcache as base


def empty_label_counter():
    return {"gold": 0, "wrong": 0, "unknown": 0, "other": 0}


def update_label_counter(counter, pred_label):
    key = str(pred_label or "other")
    if key not in counter:
        key = "other"
    counter[key] += 1


def init_prediction_counters(n_layers):
    return {
        "base": {"nc": empty_label_counter(), "ctx": empty_label_counter()},
        "per_layer": [{"nc": empty_label_counter(), "ctx": empty_label_counter()} for _ in range(n_layers)],
    }


def accumulate_prediction_counters(prediction_counters, rec):
    update_label_counter(prediction_counters["base"]["nc"], rec["nc"].get("pred_label"))
    update_label_counter(prediction_counters["base"]["ctx"], rec["ctx"].get("pred_label"))
    for layer_idx_str, patched in rec["layers"].items():
        layer_idx = int(layer_idx_str)
        if "nc" in patched:
            update_label_counter(prediction_counters["per_layer"][layer_idx]["nc"], patched["nc"].get("pred_label"))
        if "ctx" in patched:
            update_label_counter(prediction_counters["per_layer"][layer_idx]["ctx"], patched["ctx"].get("pred_label"))


def rates_from_counter(counter, denom):
    denom = max(int(denom), 1)
    return {f"{label}_rate": float(count / denom) for label, count in counter.items()}


def build_prediction_rate_summary(prediction_counters, n_layers, used):
    summary = {"n_samples": int(used), "base": {}, "per_layer": [], "best_layers": {}}
    if used <= 0:
        return summary
    for scope in ("nc", "ctx"):
        summary["base"][scope] = rates_from_counter(prediction_counters["base"][scope], used)
    per_layer = []
    for layer_idx in range(n_layers):
        row = {"layer": int(layer_idx)}
        for scope in ("nc", "ctx"):
            layer_rates = rates_from_counter(prediction_counters["per_layer"][layer_idx][scope], used)
            base_rates = summary["base"][scope]
            for key, value in layer_rates.items():
                row[f"{scope}_{key}"] = value
                row[f"{scope}_{key}_delta"] = float(value - base_rates[key])
        per_layer.append(row)
    summary["per_layer"] = per_layer
    if per_layer:
        summary["best_layers"] = {
            "ctx_unknown_rate_delta": max(per_layer, key=lambda row: row["ctx_unknown_rate_delta"]),
            "ctx_gold_rate_delta": max(per_layer, key=lambda row: row["ctx_gold_rate_delta"]),
            "nc_unknown_rate_delta": max(per_layer, key=lambda row: row["nc_unknown_rate_delta"]),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Standalone layer trace with unknown-rate summary.")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trace_mode", default="conflict", choices=["conflict"])
    ap.add_argument("--position", default="image_conflict")
    ap.add_argument("--layer_scale", type=float, default=0.0)
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", default="follow_context")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_prefix", default="")
    ap.add_argument("--scan_plan_out", default="")
    ap.add_argument("--scan_plan_metric", default="hallucination_relief")
    args = ap.parse_args()

    args.model, _ = base.resolve_model_selection(model_name=args.model_name, model=args.model)
    args.position = base.normalize_position(args.position)
    args.trace_mode = base.normalize_trace_mode(args.trace_mode)
    if args.mask_scale <= 0:
        raise SystemExit("--mask_scale must be positive")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    allowed_metrics = {"gold_wrong_margin", "follow_conflict", "follow_context", "context_flip"}
    for m in metrics:
        if m not in allowed_metrics:
            raise SystemExit(f"Unknown metric '{m}'. Allowed: {sorted(allowed_metrics)}")

    df = pd.read_csv(args.data_csv)
    model, processor = base.load_model_and_processor(args.model, args.device, args.dtype)
    layers = base._get_layers(model)
    n_layers = len(layers)

    layer_deltas = {m: [0.0 for _ in range(n_layers)] for m in metrics}
    for extra_metric in ("follow_context", "nc_gold_margin_damage", "hallucination_relief"):
        layer_deltas.setdefault(extra_metric, [0.0 for _ in range(n_layers)])
    prediction_counters = init_prediction_counters(n_layers)

    rows = df.to_dict("records")
    if args.limit > 0:
        rows = rows[: args.limit]

    records = []
    used = 0
    skipped = 0
    for row_idx, row in tqdm(enumerate(rows), total=len(rows), desc="Tracing"):
        gold, wrong, unknown, _ = base.get_answer_candidates_from_row(row)
        question = str(row["question"])
        if not gold or not wrong or gold == wrong:
            skipped += 1
            continue
        try:
            img_path, det_path, mask_path, target_labels, nc_image, ic_image = base.load_nc_ic_images(
                row, args.image_root, max_side=672, mask_scale=args.mask_scale
            )
        except Exception:
            skipped += 1
            continue

        nc_prompt = base.shared_build_nc_prompt(question)
        ctx_prompt = base.shared_build_ic_prompt(question)
        nc_inputs = base.build_inputs_mm(processor, nc_prompt, nc_image, model.device)
        ctx_inputs = base.build_inputs_mm(processor, ctx_prompt, ic_image, model.device)
        nc_cache = base.build_prompt_cache_from_inputs(model, nc_inputs, need_hidden_states=False)
        ctx_cache = base.build_prompt_cache_from_inputs(model, ctx_inputs, need_hidden_states=False)
        nc_scores = base.score_answer_candidates_from_cache(model, processor, nc_cache, gold, wrong, unknown)
        ctx_scores = base.score_answer_candidates_from_cache(model, processor, ctx_cache, gold, wrong, unknown)
        if not base.is_hallucination_sample(nc_scores, ctx_scores):
            skipped += 1
            continue

        rec = {
            "row_idx": row_idx,
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "unknown": unknown,
            "image_path": str(img_path),
            "detection_path": str(det_path),
            "mask_path": str(mask_path),
            "ic_target_labels": target_labels,
            "trace_mode": args.trace_mode,
            "mask_scale": float(args.mask_scale),
            "nc": dict(nc_scores),
            "ctx": dict(ctx_scores),
            "layers": {},
        }
        for layer_idx in range(n_layers):
            nc_layer_cache = base.build_layer_scaled_prompt_cache_from_inputs(model, nc_inputs, layer_idx, args.layer_scale)
            ctx_layer_cache = base.build_layer_scaled_prompt_cache_from_inputs(model, ctx_inputs, layer_idx, args.layer_scale)
            rec["layers"][str(layer_idx)] = {
                "nc": base.score_answer_candidates_from_cache(model, processor, nc_layer_cache, gold, wrong, unknown),
                "ctx": base.score_answer_candidates_from_cache(model, processor, ctx_layer_cache, gold, wrong, unknown),
            }
        records.append(rec)
        base.accumulate_trace_record(layer_deltas, rec, metrics)
        accumulate_prediction_counters(prediction_counters, rec)
        used += 1

    if used > 0:
        for key in layer_deltas:
            layer_deltas[key] = [value / used for value in layer_deltas[key]]

    out = {
        "config": vars(args),
        "used": used,
        "skipped": skipped,
        "layer_scores": layer_deltas,
        "prediction_rate_summary": build_prediction_rate_summary(prediction_counters, n_layers, used),
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plot_prefix:
        base.plot_scores(layer_deltas, args.plot_prefix)

    if args.scan_plan_out:
        plan_metric = args.scan_plan_metric.strip() or "hallucination_relief"
        if plan_metric not in layer_deltas:
            raise SystemExit(f"scan plan metric {plan_metric} not found in computed metrics")
        scan_plan = base.build_scan_plan(layer_deltas[plan_metric])
        scan_out = {
            "n_samples": used,
            "n_layers": n_layers,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "mask_scale": float(args.mask_scale),
            "layer_scale": args.layer_scale,
            "plan_scores_source": {"mode": "single_metric", "metric": plan_metric, "weights": {plan_metric: 1.0}},
            "scan_plan": scan_plan,
        }
        plan_path = Path(args.scan_plan_out)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(scan_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"saved_to": str(out_path), "used": used, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
