#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


ROOT = Path("/root/logit_lens/PIC/probe")
FIG8_DIR = ROOT / "fig8"
CSV_PATH = ROOT / "hulumed4b_probe_ablation_panel.csv"
PNG_PATH = FIG8_DIR / "hulumed_fig8.png"
PDF_PATH = FIG8_DIR / "hulumed_fig8.pdf"
TURN_REGION_HALF_WIDTH = 2
PLATEAU_TOLERANCE = 0.02
PLATEAU_MIN_RUN = 4
PLATEAU_MAX_SPREAD = 0.02
RISING_LOOKBACK = 8

MODEL_NAME = "HuluMed-4B"
SERIES_KEYS = (
    "probe_a_auroc",
    "probe_b_macro_f1",
    "probe_b_macro_f1_post_ablation",
)
def pick_font_family() -> list[str]:
    candidates = [
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "Songti SC",
        "SimSun",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "DejaVu Serif",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    chosen = [name for name in candidates if name in installed]
    return chosen or ["DejaVu Serif"]


def load_series() -> dict[str, list[tuple[int, float]]]:
    series_map: dict[str, list[tuple[int, float]]] = {name: [] for name in SERIES_KEYS}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["model"] != MODEL_NAME:
                continue
            series_name = row["series"]
            if series_name not in series_map:
                continue
            series_map[series_name].append((int(row["layer"]), float(row["value"])))

    for name, curve in series_map.items():
        if not curve:
            raise ValueError(f"Missing curve for {name} in {CSV_PATH}")
        curve.sort(key=lambda item: item[0])
    return series_map


def argmax(curve: list[tuple[int, float]]) -> tuple[int, float]:
    return max(curve, key=lambda item: item[1])


def find_pre_plateau_turning_layer(
    curve: list[tuple[int, float]],
    tolerance: float = PLATEAU_TOLERANCE,
    min_run: int = PLATEAU_MIN_RUN,
    max_spread: float = PLATEAU_MAX_SPREAD,
) -> tuple[int, int]:
    if not curve:
        raise ValueError("Curve is empty")

    values = [value for _, value in curve]
    peak_value = max(values)
    plateau_threshold = peak_value - tolerance

    plateau_start_idx = len(curve) - 1
    window = max(1, min_run)
    for start_idx in range(len(curve) - window + 1):
        window_values = values[start_idx : start_idx + window]
        if all(value >= plateau_threshold for value in window_values) and (
            max(window_values) - min(window_values) <= max_spread
        ):
            plateau_start_idx = start_idx
            break

    turn_idx = max(0, plateau_start_idx - 1)
    return curve[turn_idx][0], curve[plateau_start_idx][0]


def find_rising_region(
    curve: list[tuple[int, float]],
    plateau_start_layer: int,
    lookback: int = RISING_LOOKBACK,
) -> tuple[int, int]:
    layer_to_idx = {layer: idx for idx, (layer, _) in enumerate(curve)}
    plateau_idx = layer_to_idx[plateau_start_layer]
    start_idx = max(0, plateau_idx - lookback)
    search_slice = curve[start_idx : plateau_idx + 1]
    rise_start_layer, _ = min(search_slice, key=lambda item: item[1])
    return rise_start_layer, plateau_start_layer


def main() -> None:
    FIG8_DIR.mkdir(parents=True, exist_ok=True)
    font_family = pick_font_family()
    series_labels = {
        "probe_a_auroc": "Probe-A: Conflict Detection",
        "probe_b_macro_f1": "Probe-B: Follow-Conflict Prediction",
        "probe_b_macro_f1_post_ablation": "Probe-B (post-ablation)",
    }
    baseline_label = "Random Baseline"
    conflict_label = "Conflict Encoding"
    shift_label = "Preference Shift"
    delta_label = "Delta $\\ell$: {delta} layers"
    peak_ablation_label = "Peak={value:.2f} (post-ablation)"

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 16,
            "axes.labelsize": 24,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
            "axes.unicode_minus": False,
        }
    )

    series = load_series()
    probe_a = series["probe_a_auroc"]
    probe_b = series["probe_b_macro_f1"]
    probe_b_ab = series["probe_b_macro_f1_post_ablation"]

    x_a = [x for x, _ in probe_a]
    y_a = [y for _, y in probe_a]
    x_b = [x for x, _ in probe_b]
    y_b = [y for _, y in probe_b]
    x_b_ab = [x for x, _ in probe_b_ab]
    y_b_ab = [y for _, y in probe_b_ab]

    probe_a_peak_layer, probe_a_peak_value = argmax(probe_a)
    probe_b_peak_layer, probe_b_peak_value = argmax(probe_b)
    probe_b_ab_peak_layer, probe_b_ab_peak_value = argmax(probe_b_ab)
    probe_a_turn_layer, probe_a_plateau_start = find_pre_plateau_turning_layer(probe_a)
    probe_b_turn_layer, probe_b_plateau_start = find_pre_plateau_turning_layer(probe_b)
    probe_a_rise_start, probe_a_rise_end = find_rising_region(probe_a, probe_a_plateau_start)
    probe_b_rise_start, probe_b_rise_end = find_rising_region(probe_b, probe_b_plateau_start)
    delta_layers = probe_b_rise_start - probe_a_rise_start
    probe_a_region = (
        max(1, probe_a_rise_start) - 0.5,
        min(36, probe_a_rise_end) + 0.5,
    )
    probe_b_region = (
        max(1, probe_b_rise_start) - 0.5,
        min(36, probe_b_rise_end) + 0.5,
    )
    probe_a_region_center = (probe_a_region[0] + probe_a_region[1]) / 2
    probe_b_region_center = (probe_b_region[0] + probe_b_region[1]) / 2

    fig, ax = plt.subplots(figsize=(11.5, 10.4))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.96)

    ax.axvspan(probe_a_region[0], probe_a_region[1], color="#5b9bd5", alpha=0.22, zorder=0)
    ax.axvspan(probe_b_region[0], probe_b_region[1], color="#f4a259", alpha=0.28, zorder=0)

    ax.plot(x_a, y_a, color="#2f73b7", linewidth=3.0, label=series_labels["probe_a_auroc"], zorder=3)
    ax.plot(x_b, y_b, color="#ff7f0e", linewidth=3.0, label=series_labels["probe_b_macro_f1"], zorder=3)
    ax.plot(
        x_b_ab,
        y_b_ab,
        color="#ff8c1a",
        linewidth=2.8,
        linestyle="--",
        label=series_labels["probe_b_macro_f1_post_ablation"],
        zorder=3,
    )
    ax.axhline(0.5, color="0.55", linewidth=2.0, linestyle="--", label=baseline_label, zorder=1)

    ax.set_xlim(1, 36)
    ax.set_ylim(0.10, 1.02)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("AUROC / Macro-F1")
    ax.set_xticks(range(0, 37, 2))
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, axis="y", color="0.82", linewidth=1.0)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        fontsize=14,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=2.5,
    )
    legend.get_frame().set_edgecolor("0.75")

    ax.text(
        11.8,
        0.855,
        conflict_label,
        ha="center",
        va="center",
        fontsize=19,
    )
    ax.text(
        28.2,
        0.18,
        shift_label,
        ha="center",
        va="center",
        fontsize=19,
    )

    arrow_y = 0.445
    ax.annotate(
        "",
        xy=(16.0, arrow_y),
        xytext=(19.0, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.8),
    )
    ax.text(
        19.8,
        0.465,
        delta_label.format(delta=abs(delta_layers)),
        ha="center",
        va="bottom",
        fontsize=19,
    )

    ax.annotate(
        f"Peak={probe_b_peak_value:.2f}",
        xy=(probe_b_peak_layer, probe_b_peak_value),
        xytext=(27.4, 0.905),
        fontsize=18,
        arrowprops=dict(arrowstyle="-", color="black", lw=1.4),
    )
    ax.annotate(
        peak_ablation_label.format(value=probe_b_ab_peak_value),
        xy=(probe_b_ab_peak_layer, probe_b_ab_peak_value),
        xytext=(24.7, 0.772),
        fontsize=15,
        arrowprops=dict(arrowstyle="-", color="black", lw=1.4),
    )
    ax.text(28.9, 0.474, baseline_label, fontsize=16, va="bottom", ha="left")

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")
    print(
        "Peaks:",
        {
            "probe_a": (probe_a_peak_layer, round(probe_a_peak_value, 4)),
            "probe_b": (probe_b_peak_layer, round(probe_b_peak_value, 4)),
            "probe_b_post_ablation": (probe_b_ab_peak_layer, round(probe_b_ab_peak_value, 4)),
            "probe_a_pre_plateau_turn": probe_a_turn_layer,
            "probe_a_plateau_start": probe_a_plateau_start,
            "probe_a_rise_start": probe_a_rise_start,
            "probe_a_rise_end": probe_a_rise_end,
            "probe_b_pre_plateau_turn": probe_b_turn_layer,
            "probe_b_plateau_start": probe_b_plateau_start,
            "probe_b_rise_start": probe_b_rise_start,
            "probe_b_rise_end": probe_b_rise_end,
        },
    )


if __name__ == "__main__":
    main()
