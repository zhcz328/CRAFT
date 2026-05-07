#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def scale_tag_to_float(tag: str) -> float:
    return float(tag.replace("p", "."))


def extract_scale_from_name(path: Path) -> float:
    marker = "layerscale_"
    if marker not in path.stem:
        raise ValueError(f"Cannot extract layer_scale from {path.name}")
    raw = path.stem.split(marker, 1)[1]
    return scale_tag_to_float(raw)


def best_layer_by_score(values):
    if not values:
        return -1, 0.0
    idx = max(range(len(values)), key=lambda i: values[i])
    return int(idx), float(values[idx])


def main():
    ap = argparse.ArgumentParser(description="Summarize multi-layer-scale traces.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_layer_csv", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(
        path
        for path in input_dir.glob(f"layer_scale_*/trace_{args.split}/trace_{args.split}_layerscale_*.json")
        if not path.name.endswith("_scan_plan.json")
    )
    if not files:
        raise SystemExit(f"No trace files found in {input_dir} for split={args.split}")

    summary_rows = []
    layer_rows = []
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        scale = extract_scale_from_name(path)
        pred_summary = obj.get("prediction_rate_summary", {})
        base_ctx = pred_summary.get("base", {}).get("ctx", {})
        per_layer = pred_summary.get("per_layer", [])
        layer_scores = obj.get("layer_scores", {})
        hallucination_relief = layer_scores.get("hallucination_relief", [])
        follow_context = layer_scores.get("follow_context", [])
        nc_damage = layer_scores.get("nc_gold_margin_damage", [])
        best_unknown_row = max(
            per_layer,
            key=lambda row: row.get("ctx_unknown_rate_delta", float("-inf")),
            default={"layer": -1, "ctx_unknown_rate_delta": 0.0, "ctx_unknown_rate": 0.0},
        )
        best_relief_layer, best_relief_value = best_layer_by_score(hallucination_relief)
        best_follow_layer, best_follow_value = best_layer_by_score(follow_context)
        best_damage_layer, best_damage_value = best_layer_by_score(nc_damage)
        summary_rows.append(
            {
                "split": args.split,
                "layer_scale": scale,
                "used": int(obj.get("used", 0)),
                "skipped": int(obj.get("skipped", 0)),
                "base_ctx_unknown_rate": float(base_ctx.get("unknown_rate", 0.0)),
                "best_ctx_unknown_rate_delta_layer": int(best_unknown_row.get("layer", -1)),
                "best_ctx_unknown_rate_delta": float(best_unknown_row.get("ctx_unknown_rate_delta", 0.0)),
                "best_ctx_unknown_rate": float(best_unknown_row.get("ctx_unknown_rate", 0.0)),
                "best_hallucination_relief_layer": best_relief_layer,
                "best_hallucination_relief": best_relief_value,
                "best_follow_context_layer": best_follow_layer,
                "best_follow_context": best_follow_value,
                "best_nc_damage_layer": best_damage_layer,
                "best_nc_damage": best_damage_value,
                "source_json": str(path),
            }
        )
        for row in per_layer:
            layer_idx = int(row["layer"])
            layer_rows.append(
                {
                    "split": args.split,
                    "layer_scale": scale,
                    "layer": layer_idx,
                    "ctx_unknown_rate": float(row.get("ctx_unknown_rate", 0.0)),
                    "ctx_unknown_rate_delta": float(row.get("ctx_unknown_rate_delta", 0.0)),
                    "ctx_gold_rate": float(row.get("ctx_gold_rate", 0.0)),
                    "ctx_gold_rate_delta": float(row.get("ctx_gold_rate_delta", 0.0)),
                    "nc_unknown_rate": float(row.get("nc_unknown_rate", 0.0)),
                    "nc_unknown_rate_delta": float(row.get("nc_unknown_rate_delta", 0.0)),
                    "hallucination_relief": float(hallucination_relief[layer_idx]) if layer_idx < len(hallucination_relief) else 0.0,
                    "follow_context": float(follow_context[layer_idx]) if layer_idx < len(follow_context) else 0.0,
                    "nc_gold_margin_damage": float(nc_damage[layer_idx]) if layer_idx < len(nc_damage) else 0.0,
                }
            )

    summary_rows.sort(key=lambda row: row["layer_scale"])
    layer_rows.sort(key=lambda row: (row["layer_scale"], row["layer"]))
    out_json = {
        "split": args.split,
        "n_runs": len(summary_rows),
        "layer_scales": [row["layer_scale"] for row in summary_rows],
        "by_scale": summary_rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(args.out_layer_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    print(json.dumps({"summary_json": args.out_json, "summary_csv": args.out_csv, "layer_csv": args.out_layer_csv}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
