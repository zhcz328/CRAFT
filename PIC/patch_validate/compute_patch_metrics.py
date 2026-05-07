from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    LOCAL_K_VARIANTS,
    MANIFEST_ROOT,
    METRIC_ROOT,
    compute_consensus_curve,
    compute_robustness,
    compute_smoothness,
    dump_json,
    ensure_result_dirs,
    load_curve,
    load_json,
    variant_sort_key,
    write_markdown_table,
)


def build_metrics(manifest_path: Path, robustness_method: str) -> Dict[str, Any]:
    ensure_result_dirs()
    manifest = load_json(manifest_path)
    experiments = manifest["experiments"]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in experiments:
        grouped.setdefault(entry["model_key"], []).append(entry)

    output_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "robustness_method": robustness_method,
        "models": {},
        "missing": [],
    }

    for model_key, entries in grouped.items():
        entries = sorted(entries, key=lambda item: variant_sort_key(item["variant"]))
        present_curves: Dict[str, List[float]] = {}
        model_meta = {
            "display_name": entries[0]["display_name"],
            "modality": entries[0]["modality"],
            "position": entries[0]["position"],
            "variants": {},
            "missing_variants": [],
        }

        for entry in entries:
            trace_path = Path(entry["trace_json"])
            if not trace_path.exists():
                model_meta["missing_variants"].append(entry["variant"])
                summary["missing"].append(
                    {
                        "model_key": model_key,
                        "variant": entry["variant"],
                        "trace_json": str(trace_path),
                    }
                )
                continue
            present_curves[entry["variant"]] = load_curve(trace_path, entry.get("curve_metric"))

        local_curves = [
            present_curves[variant]
            for variant in LOCAL_K_VARIANTS
            if variant in present_curves
        ]
        consensus_curve = compute_consensus_curve(local_curves)

        for entry in entries:
            variant = entry["variant"]
            if variant not in present_curves:
                continue
            curve = present_curves[variant]
            smoothness = compute_smoothness(curve)
            robustness = (
                compute_robustness(curve, consensus_curve, method=robustness_method)
                if consensus_curve.size
                else None
            )
            peak_layer = max(range(len(curve)), key=lambda idx: curve[idx]) if curve else None
            peak_score = curve[peak_layer] if peak_layer is not None else None

            row = {
                "model_key": model_key,
                "display_name": entry["display_name"],
                "modality": entry["modality"],
                "position": entry["position"],
                "variant": variant,
                "patch_k": entry["patch_k"],
                "patch_all_token": entry["patch_all_token"],
                "smoothness": smoothness,
                "robustness": robustness,
                "peak_layer": peak_layer,
                "peak_score": peak_score,
                "trace_json": entry["trace_json"],
                "curve_metric": entry.get("curve_metric"),
            }
            output_rows.append(row)
            model_meta["variants"][variant] = row

        if consensus_curve.size:
            model_meta["local_consensus_curve"] = [float(x) for x in consensus_curve.tolist()]
        summary["models"][model_key] = model_meta

    csv_headers = [
        "model_key",
        "display_name",
        "modality",
        "position",
        "variant",
        "patch_k",
        "patch_all_token",
        "smoothness",
        "robustness",
        "peak_layer",
        "peak_score",
        "trace_json",
    ]
    csv_lines = [",".join(csv_headers)]
    for row in output_rows:
        csv_lines.append(
            ",".join(
                [
                    str(row["model_key"]),
                    str(row["display_name"]),
                    str(row["modality"]),
                    str(row["position"]),
                    str(row["variant"]),
                    str(row["patch_k"]),
                    str(row["patch_all_token"]),
                    "" if row["smoothness"] is None else f"{row['smoothness']:.8f}",
                    "" if row["robustness"] is None else f"{row['robustness']:.8f}",
                    "" if row["peak_layer"] is None else str(row["peak_layer"]),
                    "" if row["peak_score"] is None else f"{row['peak_score']:.8f}",
                    str(row["trace_json"]),
                ]
            )
        )

    (METRIC_ROOT / "patch_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    dump_json(METRIC_ROOT / "patch_metrics.json", summary)

    md_rows = []
    for row in output_rows:
        md_rows.append(
            [
                row["display_name"],
                row["modality"],
                row["variant"],
                f"{row['smoothness']:.4f}",
                "NA" if row["robustness"] is None else f"{row['robustness']:.4f}",
                row["peak_layer"],
                "NA" if row["peak_score"] is None else f"{row['peak_score']:.4f}",
            ]
        )
    write_markdown_table(
        METRIC_ROOT / "patch_metrics.md",
        ["Model", "Modality", "Variant", "Smoothness", "Robustness", "Peak Layer", "Peak Score"],
        md_rows,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute smoothness and robustness for patch curves.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "patch_trace_manifest.json",
        help="Path to the patch trace manifest produced by generate_run_manifest.py",
    )
    parser.add_argument(
        "--robustness-method",
        choices=["consensus_l1", "pearson_to_consensus"],
        default="consensus_l1",
        help="Robustness definition. consensus_l1 is the default appendix-friendly stability score.",
    )
    args = parser.parse_args()
    summary = build_metrics(args.manifest, args.robustness_method)
    print(
        f"Saved metrics for {len(summary['models'])} models to "
        f"{METRIC_ROOT / 'patch_metrics.csv'} and {METRIC_ROOT / 'patch_metrics.json'}"
    )


if __name__ == "__main__":
    main()

