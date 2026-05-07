#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Write a zero-valued ablation summary when no heads are selected.")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--selected_heads", required=True)
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--trace_mode", default="conflict")
    ap.add_argument("--position", default="image_conflict")
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--metrics", default="follow_context")
    ap.add_argument("--mask_scope", default="all")
    ap.add_argument("--keep_mode", default="self")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out = {
        "config": {
            "data_csv": args.data_csv,
            "model": args.model,
            "model_name": args.model_name,
            "selected_heads": args.selected_heads,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "mask_scale": float(args.mask_scale),
            "metrics": [m.strip() for m in str(args.metrics).split(",") if m.strip()],
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "dtype": args.dtype,
            "device": args.device,
            "auto_skip_reason": "no_selected_heads",
        },
        "selected_pairs": [],
        "n_selected_heads": 0,
        "n_samples": 0,
        "sample_filter": "none",
        "mask_scale": float(args.mask_scale),
        "mean_hallucination_relief": 0.0,
        "mean_ic_follow_context_gain": 0.0,
        "mean_nc_gold_margin_damage": 0.0,
        "pred_nc": {"base_acc": 0.0, "ab_acc": 0.0, "acc_delta": 0.0},
        "pred_ctx": {"base_acc": 0.0, "ab_acc": 0.0, "acc_delta": 0.0},
        "ctx_unknown": {"base_rate": 0.0, "ab_rate": 0.0, "rate_delta": 0.0},
        "per_metric": {},
        "records": [],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "reason": "no_selected_heads"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
