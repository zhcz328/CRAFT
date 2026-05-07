#!/usr/bin/env python3
"""Draw bubble charts for image-conflict layer-trace hallucination relief scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "image_layer_trace"
NUMERIC_FONT_SCALE = 2.5

PANELS = [
    {
        "title": "Hulu-med-4B",
        "short_name": "hulumed4b",
        "source": ROOT / "Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/trace_image_conflict.json",
    },
    {
        "title": "InternVL3.5-4B",
        "short_name": "internvl35_4b",
        "source": ROOT / "Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/trace_image_conflict.json",
    },
]

TEAL_WHITE_CORAL = mcolors.LinearSegmentedColormap.from_list(
    "teal_white_coral",
    ["#349BA5", "#ffffff", "#b54432"],
    N=256,
)


def load_metric_series(path: Path, metric: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layer_scores = payload.get("layer_scores", {})
    if metric not in layer_scores:
        raise KeyError(f"{metric!r} not found in {path}")
    values = np.asarray(layer_scores[metric], dtype=float)
    return {
        "values": values,
        "used": int(payload.get("used", 0)),
        "skipped": int(payload.get("skipped", 0)),
        "config": payload.get("config", {}),
    }


def bubble_sizes(values: np.ndarray, min_size: float = 120.0, max_size: float = 1500.0) -> np.ndarray:
    abs_values = np.abs(values)
    max_abs = float(np.max(abs_values)) if len(abs_values) else 0.0
    if max_abs <= 1e-12:
        return np.full_like(abs_values, min_size, dtype=float)
    scaled = np.sqrt(abs_values / max_abs)
    return min_size + scaled * (max_size - min_size)


def color_positions(
    values: np.ndarray,
    min_value: float,
    positive_cap: float,
    positive_floor: float = 0.42,
    negative_floor: float = 0.18,
) -> np.ndarray:
    positions = np.zeros_like(values, dtype=float)
    neg_scale = max(abs(min(min_value, 0.0)), 1e-6)
    pos_scale = max(positive_cap, 1e-6)

    neg_mask = values < 0
    if np.any(neg_mask):
        clipped = np.clip(np.abs(values[neg_mask]) / neg_scale, 0.0, 1.0)
        stretched = np.power(clipped, 0.65)
        positions[neg_mask] = -(negative_floor + (1.0 - negative_floor) * stretched)

    pos_mask = values > 0
    if np.any(pos_mask):
        clipped = np.clip(values[pos_mask] / pos_scale, 0.0, 1.0)
        positions[pos_mask] = positive_floor + (1.0 - positive_floor) * clipped

    return np.clip(positions, -1.0, 1.0)


def save_standalone_legend(
    out_dir: Path,
    cmap: mcolors.Colormap,
    max_neg: float,
    positive_cap: float,
) -> Dict[str, str]:
    fig_leg = plt.figure(figsize=(5.0, 1.2), dpi=300)
    ax_leg = fig_leg.add_axes([0.11, 0.42, 0.78, 0.15])
    norm = mcolors.TwoSlopeNorm(vmin=-max_neg, vcenter=0.0, vmax=positive_cap)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig_leg.colorbar(sm, cax=ax_leg, orientation="horizontal")
    cbar.set_ticks([-max_neg, -0.5 * max_neg, 0.0, 0.5 * positive_cap, positive_cap])
    cbar.set_ticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor("#666666")
    cbar.ax.tick_params(length=2.2, width=0.7, pad=2, labelsize=7.2 * NUMERIC_FONT_SCALE)
    fig_leg.text(0.50, 0.82, "Relative Hallucination Relief", ha="center", va="center", fontsize=16)

    png_path = out_dir / "image_conflict_hallucination_relief_bubble_legend_only.png"
    pdf_path = out_dir / "image_conflict_hallucination_relief_bubble_legend_only.pdf"
    fig_leg.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig_leg.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_leg)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def draw() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11 * NUMERIC_FONT_SCALE,
            "ytick.labelsize": 12 * NUMERIC_FONT_SCALE,
        }
    )

    metric = "hallucination_relief"
    series: List[Dict[str, Any]] = []
    all_values: List[np.ndarray] = []
    max_layers = 0

    for panel in PANELS:
        loaded = load_metric_series(panel["source"], metric)
        record = {**panel, **loaded}
        series.append(record)
        all_values.append(record["values"])
        max_layers = max(max_layers, len(record["values"]))

    merged = np.concatenate(all_values) if all_values else np.array([0.0], dtype=float)
    min_value = float(np.min(merged)) if len(merged) else -1.0
    max_value = float(np.max(merged)) if len(merged) else 1.0
    max_neg = max(abs(min(min_value, 0.0)), 1e-6)
    positive_values = merged[merged > 0]
    if len(positive_values):
        positive_cap = max(float(np.percentile(positive_values, 60)), max_value * 0.18, 1e-6)
    else:
        positive_cap = 1.0

    fig, ax = plt.subplots(figsize=(4.6, 10.2))
    cmap = TEAL_WHITE_CORAL
    color_norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    manifest: Dict[str, Any] = {"metric": metric, "panels": []}

    for row_idx, record in enumerate(series):
        values = record["values"]
        y_pos = np.arange(len(values), dtype=float)
        color_pos = color_positions(values, min_value=min_value, positive_cap=positive_cap)
        x_pos = row_idx
        xs = np.full(len(values), x_pos, dtype=float)
        ys = y_pos
        sizes = bubble_sizes(values)

        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=color_pos,
            cmap=cmap,
            norm=color_norm,
            edgecolors="#2f2f2f",
            linewidths=0.4,
            alpha=0.96,
        )

        manifest["panels"].append(
            {
                "title": record["title"],
                "source": str(record["source"]),
                "used": record["used"],
                "skipped": record["skipped"],
                "n_layers": len(values),
                "value_min": float(np.min(values)),
                "value_max": float(np.max(values)),
                "value_mean": float(np.mean(values)),
            }
        )

    for x_line in np.arange(-0.5, len(series), 1.0):
        ax.axvline(x_line, color="#d9d9d9", linewidth=0.8, zorder=0)

    ax.set_xlim(-0.7, len(series) - 0.3)
    ax.set_ylim(-1.25, max_layers - 0.2)
    ax.set_xticks(np.arange(len(series)))
    ax.set_xticklabels([])
    yticks = list(range(0, max_layers, 5))
    if (max_layers - 1) not in yticks:
        yticks.append(max_layers - 1)
    ax.set_yticks(sorted(yticks))
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_color("#6a6a6a")
        spine.set_linewidth(1.45)

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.05, top=0.985)

    png_path = OUT_DIR / "image_conflict_hallucination_relief_bubble.png"
    pdf_path = OUT_DIR / "image_conflict_hallucination_relief_bubble.pdf"
    source_path = OUT_DIR / "image_conflict_hallucination_relief_bubble_sources.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    legend_outputs = save_standalone_legend(OUT_DIR, cmap, max_neg=max_neg, positive_cap=positive_cap)

    manifest["outputs"] = {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "sources": str(source_path),
        "legend_png": legend_outputs["png"],
        "legend_pdf": legend_outputs["pdf"],
    }
    manifest["color_scale"] = {
        "vmin": -max_neg,
        "vcenter": 0.0,
        "vmax": positive_cap,
        "display_tick_mode": "separate_signed_normalized",
        "display_ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "positive_cap_rule": "60th_percentile_of_positive_values_with_lower_bound_0.18_times_positive_max",
        "cmap": "teal_white_coral",
    }
    source_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = draw()
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
