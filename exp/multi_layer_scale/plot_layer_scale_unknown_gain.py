#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "outputs" / "layer_scale_sweep" / "layer_scale_sweep_val_ablate_summary.csv"
DEFAULT_INPUT_JSON = BASE_DIR / "outputs" / "layer_scale_sweep" / "layer_scale_sweep_val_ablate_summary.json"
DEFAULT_OUT_PREFIX = BASE_DIR / "outputs" / "layer_scale_sweep" / "layer_scale_vs_unknown_gain"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot layer_scale vs unknown gain from multi_layer_scale outputs."
    )
    ap.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    ap.add_argument("--input_json", type=Path, default=DEFAULT_INPUT_JSON)
    ap.add_argument("--out_prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    ap.add_argument(
        "--use_percent",
        action="store_true",
        help="Plot unknown gain in percentage points instead of raw rate delta.",
    )
    return ap.parse_args()


def load_rows(csv_path: Path, json_path: Path) -> list[dict]:
    rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        rows = payload.get("by_scale", [])
    else:
        raise FileNotFoundError(f"No input summary found: {csv_path} or {json_path}")

    normalized = []
    for row in rows:
        normalized.append(
            {
                "layer_scale": float(row["layer_scale"]),
                "ctx_unknown_base_rate": float(row["ctx_unknown_base_rate"]),
                "ctx_unknown_ab_rate": float(row["ctx_unknown_ab_rate"]),
                "ctx_unknown_rate_delta": float(row["ctx_unknown_rate_delta"]),
            }
        )
    return sorted(normalized, key=lambda row: row["layer_scale"])


def make_plot(rows: list[dict], out_prefix: Path, use_percent: bool) -> tuple[Path, Path]:
    x_values = [row["layer_scale"] for row in rows]
    y_values = [row["ctx_unknown_rate_delta"] for row in rows]
    if use_percent:
        y_values = [value * 100.0 for value in y_values]
        ylabel = "unknown gain (percentage points)"
    else:
        ylabel = "unknown gain"

    plt.figure(figsize=(8, 5), dpi=220)
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        color="#1f5aa6",
    )
    plt.axhline(0.0, color="#7a7a7a", linewidth=1.0, linestyle="--", alpha=0.8)
    plt.xlabel("layer_scale")
    plt.ylabel(ylabel)
    plt.title("Layer Scale vs Unknown Gain")
    plt.grid(True, alpha=0.28)
    plt.xticks(x_values)
    plt.tight_layout()

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv, args.input_json)
    png_path, pdf_path = make_plot(rows, args.out_prefix, args.use_percent)

    print(f"loaded_rows={len(rows)}")
    print(f"input_csv={args.input_csv}")
    print(f"input_json={args.input_json}")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")
    for row in rows:
        print(
            "layer_scale={layer_scale} unknown_base={base} unknown_ab={ab} unknown_gain={delta}".format(
                layer_scale=row["layer_scale"],
                base=row["ctx_unknown_base_rate"],
                ab=row["ctx_unknown_ab_rate"],
                delta=row["ctx_unknown_rate_delta"],
            )
        )


if __name__ == "__main__":
    main()
