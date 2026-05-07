#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "mask_scale_sweep_summary.csv"
DEFAULT_JSON = DATA_DIR / "mask_scale_sweep_summary.json"
DEFAULT_PNG = BASE_DIR / "mask_scale_vs_ic_unknown_rate.png"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot mask scale vs IC unknown rate from precomputed summary data."
    )
    ap.add_argument("--summary_csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--summary_json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out_png", type=Path, default=DEFAULT_PNG)
    return ap.parse_args()


def load_summary(csv_path: Path, json_path: Path) -> list[dict]:
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif json_path.exists():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(
            f"Neither summary CSV nor JSON exists: {csv_path} / {json_path}"
        )

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                **row,
                "mask_scale": float(row["mask_scale"]),
                "ic_unknown_rate": float(row["ic_unknown_rate"]),
            }
        )

    required = {"mask_scale", "ic_unknown_rate"}
    missing = required - set(normalized_rows[0].keys()) if normalized_rows else required
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return sorted(normalized_rows, key=lambda row: row["mask_scale"])


def make_plot(rows: list[dict], out_path: Path) -> None:
    x_values = [row["mask_scale"] for row in rows]
    y_values = [row["ic_unknown_rate"] for row in rows]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2,
        color="#2F5DA8",
    )
    plt.xlabel("mask scale")
    plt.ylabel("ic_unknown_rate")
    plt.title("Mask Scale vs IC Unknown Rate")
    plt.grid(True, alpha=0.3)
    plt.xticks(x_values, rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_summary(args.summary_csv, args.summary_json)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    make_plot(rows, args.out_png)

    print(f"loaded_rows={len(rows)}")
    print(f"summary_csv={args.summary_csv}")
    print(f"summary_json={args.summary_json}")
    print(f"saved_plot={args.out_png}")
    for row in rows:
        print(
            "mask_scale={mask_scale} ic_unknown_rate={ic_unknown_rate}".format(
                mask_scale=row["mask_scale"],
                ic_unknown_rate=row["ic_unknown_rate"],
            )
        )


if __name__ == "__main__":
    main()
