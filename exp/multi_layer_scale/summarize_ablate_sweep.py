#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def scale_tag_to_float(tag: str) -> float:
    return float(tag.replace("p", "."))


def extract_scale_from_path(path: Path) -> float:
    for part in reversed(path.parts):
        if part.startswith("layer_scale_"):
            return scale_tag_to_float(part.split("layer_scale_", 1)[1])
    raise ValueError(f"Cannot extract layer_scale from {path}")


def main():
    ap = argparse.ArgumentParser(description="Summarize val ablation results across layer_scale values.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("layer_scale_*/ablate/ablate_selected_heads_all_val.json"))
    if not files:
        raise SystemExit(f"No ablation result files found under {input_dir}")

    rows = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "layer_scale": extract_scale_from_path(path),
                "n_selected_heads": int(obj.get("n_selected_heads", 0)),
                "n_samples": int(obj.get("n_samples", 0)),
                "mean_hallucination_relief": float(obj.get("mean_hallucination_relief", 0.0)),
                "mean_ic_follow_context_gain": float(obj.get("mean_ic_follow_context_gain", 0.0)),
                "mean_nc_gold_margin_damage": float(obj.get("mean_nc_gold_margin_damage", 0.0)),
                "pred_nc_base_acc": float(obj.get("pred_nc", {}).get("base_acc", 0.0)),
                "pred_nc_ab_acc": float(obj.get("pred_nc", {}).get("ab_acc", 0.0)),
                "pred_nc_acc_delta": float(obj.get("pred_nc", {}).get("acc_delta", 0.0)),
                "pred_ctx_base_acc": float(obj.get("pred_ctx", {}).get("base_acc", 0.0)),
                "pred_ctx_ab_acc": float(obj.get("pred_ctx", {}).get("ab_acc", 0.0)),
                "pred_ctx_acc_delta": float(obj.get("pred_ctx", {}).get("acc_delta", 0.0)),
                "ctx_unknown_base_rate": float(obj.get("ctx_unknown", {}).get("base_rate", 0.0)),
                "ctx_unknown_ab_rate": float(obj.get("ctx_unknown", {}).get("ab_rate", 0.0)),
                "ctx_unknown_rate_delta": float(obj.get("ctx_unknown", {}).get("rate_delta", 0.0)),
                "selected_heads_json": str(obj.get("config", {}).get("selected_heads", "")),
                "source_json": str(path),
            }
        )

    rows.sort(key=lambda row: row["layer_scale"])
    out = {"n_runs": len(rows), "by_scale": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"out_json": args.out_json, "out_csv": args.out_csv, "n_runs": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
