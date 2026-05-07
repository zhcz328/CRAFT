#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


BASE_DIR = Path("/root/logit_lens/PIC/tuned_lens/image_lens/small")
CSV_PATH = BASE_DIR / "text_lens_hulu_med_4b_curve_data.csv"
OUT_PNG = BASE_DIR / "text_lens_hulu_med_4b_reproduced.png"
OUT_PDF = BASE_DIR / "text_lens_hulu_med_4b_reproduced.pdf"

COLORS = {
    "red_follow_conflict": "#ff1f1f",
    "green_resist": "#0a8a12",
    "blue_dashed_after_ablation": "#1d2dff",
}

LINESTYLES = {
    "red_follow_conflict": "-",
    "green_resist": "-",
    "blue_dashed_after_ablation": "--",
}

LINEWIDTHS = {
    "red_follow_conflict": 4.2,
    "green_resist": 4.2,
    "blue_dashed_after_ablation": 4.0,
}

ALPHAS = {
    "red_follow_conflict": 0.24,
    "green_resist": 0.24,
    "blue_dashed_after_ablation": 0.20,
}


def plus_tick_formatter(x: float, _: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    return f"{x:g}"


def read_curves(path: Path) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(row["line_label"], []).append(row)

    out: dict[str, dict[str, list[float]]] = {}
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["layer"]))
        out[label] = {
            "layers": [int(row["layer"]) for row in rows],
            "values": [float(row["plotted_value"]) for row in rows],
            "low": [float(row["band_low"]) for row in rows],
            "high": [float(row["band_high"]) for row in rows],
        }
    return out


def curve_bounds(curves: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
    values = []
    for payload in curves.values():
        values.extend(payload["low"])
        values.extend(payload["high"])
    if not values:
        return -1.0, 1.0
    y_min = min(values)
    y_max = max(values)
    pad = max(0.25, 0.12 * (y_max - y_min))
    return y_min - pad, y_max + pad


def configure_axis(ax, x_max: int, y_min: float, y_max: float) -> None:
    ax.set_facecolor("#fffdf8")
    ax.axhspan(0.0, y_max, facecolor="#dff3de", alpha=0.50, zorder=0)
    ax.axhspan(y_min, 0.0, facecolor="#ffdede", alpha=0.52, zorder=0)
    ax.axhline(0.0, color="black", linestyle=(0, (6, 3)), linewidth=2.3, zorder=2)

    ax.set_xlim(1, x_max)
    ax.set_ylim(y_min, y_max)

    xticks = [1]
    for tick in [6, 12, 18, 24, 30, 36]:
        if tick < x_max:
            xticks.append(tick)
    if x_max not in xticks:
        xticks.append(x_max)
    ax.set_xticks(sorted(set(xticks)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_formatter(FuncFormatter(plus_tick_formatter))

    ax.grid(True, color="#a7a7a7", linewidth=1.0, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=32, width=1.6, length=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def plot_one(ax, label: str, payload: dict[str, list[float]]) -> None:
    ax.fill_between(
        payload["layers"],
        payload["low"],
        payload["high"],
        color=COLORS[label],
        alpha=ALPHAS[label],
        linewidth=0,
        zorder=3,
    )
    ax.plot(
        payload["layers"],
        payload["values"],
        color=COLORS[label],
        linestyle=LINESTYLES[label],
        linewidth=LINEWIDTHS[label],
        zorder=4,
    )


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    curves = read_curves(CSV_PATH)
    y_min, y_max = curve_bounds(curves)
    x_max = max(max(payload["layers"]) for payload in curves.values())

    fig, ax = plt.subplots(figsize=(13.2, 10.6))
    configure_axis(ax, x_max=x_max, y_min=y_min, y_max=y_max)

    for label in [
        "red_follow_conflict",
        "green_resist",
        "blue_dashed_after_ablation",
    ]:
        plot_one(ax, label, curves[label])

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
