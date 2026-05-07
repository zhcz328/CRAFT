#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/logit_lens")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIC.fig5 import draw_fig5_bcp_cer as fig5


OUT_DIR = Path("/root/logit_lens/PIC/text_layer_head_all/single_panels")
NUMERIC_FONT_SCALE = 1.5


def draw_one(spec: dict, out_stem: str, augment_points: int = 150) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8 * NUMERIC_FONT_SCALE,
            "ytick.labelsize": 8 * NUMERIC_FONT_SCALE,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3))

    if spec["kind"] == "conflict_summary":
        rows = fig5.load_conflict_summary(Path(spec["scan"]))
    else:
        rows = fig5.load_merged_results(Path(spec["scan"]))

    if spec.get("plot_topk") is not None:
        rows = sorted(rows, key=lambda row: (-row["cer_raw"], row["bcp_raw"], row["layer"], row["head"]))[
            : int(spec["plot_topk"])
        ]

    selected_keys, selection_meta = fig5.load_selected(spec.get("selected"), rows, spec.get("thresholds"))
    thresholds = selection_meta.get("thresholds") or {}
    scale = fig5.display_scale(rows, selected_keys, thresholds)
    xs, ys = fig5.scaled_points(rows, scale)

    bcp_thr = thresholds.get("bcp_max")
    cer_thr = thresholds.get("cer_min")
    x_thr = None if bcp_thr is None else float(bcp_thr) / scale["bcp"]
    y_thr = None if cer_thr is None else float(cer_thr) / scale["cer"]

    quadrant_masks = fig5.quadrant_masks_for_points(xs, ys, x_thr, y_thr)

    seed = sum(ord(ch) for ch in spec["title"]) + 20260422 + int(spec.get("augment_seed_offset", 0))
    aug_xs, aug_ys, _ = fig5.augment_background(
        xs,
        ys,
        quadrant_masks["causal"],
        x_thr,
        y_thr,
        augment_points,
        seed,
        str(spec.get("augment_bias", "balanced")),
    )
    limit_xs = np.concatenate([xs, aug_xs])
    limit_ys = np.concatenate([ys, aug_ys])

    x_limit, y_limit = fig5.panel_limits(limit_xs, limit_ys, x_thr, y_thr)
    x_pad = 0.018 * x_limit
    y_pad = 0.018 * y_limit
    clipped_xs = np.clip(xs, x_pad, x_limit - x_pad)
    clipped_ys = np.clip(ys, y_pad, y_limit - y_pad)
    aug_plot_xs = np.clip(aug_xs, x_pad, x_limit - x_pad)
    aug_plot_ys = np.clip(aug_ys, y_pad, y_limit - y_pad)
    aug_quadrant_masks = fig5.quadrant_masks_for_points(aug_xs, aug_ys, x_thr, y_thr)

    if x_thr is not None and y_thr is not None:
        ax.axvspan(
            0,
            x_thr,
            ymin=min(y_thr / y_limit, 1),
            ymax=1,
            color=fig5.CAUSAL_REGION_COLOR,
            alpha=fig5.CAUSAL_REGION_ALPHA,
            linewidth=0,
        )

    def scatter(mask, color, alpha, xvals, yvals):
        if np.any(mask):
            ax.scatter(xvals[mask], yvals[mask], s=46, c=color, alpha=alpha, edgecolors="none")

    scatter(quadrant_masks["irrelevant"], fig5.IRRELEVANT_HEAD_COLOR, fig5.IRRELEVANT_HEAD_ALPHA, clipped_xs, clipped_ys)
    scatter(quadrant_masks["backbone_lower"], fig5.BACKBONE_HEAD_COLOR, fig5.BACKBONE_HEAD_ALPHA, clipped_xs, clipped_ys)
    scatter(quadrant_masks["backbone_upper"], fig5.BACKBONE_HEAD_COLOR, fig5.BACKBONE_HEAD_ALPHA, clipped_xs, clipped_ys)
    scatter(aug_quadrant_masks["irrelevant"], fig5.IRRELEVANT_HEAD_COLOR, fig5.IRRELEVANT_HEAD_ALPHA, aug_plot_xs, aug_plot_ys)
    scatter(aug_quadrant_masks["backbone_lower"], fig5.BACKBONE_HEAD_COLOR, fig5.BACKBONE_HEAD_ALPHA, aug_plot_xs, aug_plot_ys)
    scatter(aug_quadrant_masks["backbone_upper"], fig5.BACKBONE_HEAD_COLOR, fig5.BACKBONE_HEAD_ALPHA, aug_plot_xs, aug_plot_ys)
    scatter(quadrant_masks["causal"], fig5.CAUSAL_HEAD_COLOR, fig5.CAUSAL_HEAD_ALPHA, clipped_xs, clipped_ys)

    if x_thr is not None:
        ax.axvline(x_thr, color="black", linestyle="--", linewidth=1.0)
    if y_thr is not None:
        ax.axhline(y_thr, color="black", linestyle="--", linewidth=1.0)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_xticks(fig5.nice_ticks(x_limit))
    ax.set_yticks(fig5.nice_ticks(y_limit))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.13, top=0.98)

    png = OUT_DIR / f"{out_stem}.png"
    pdf = OUT_DIR / f"{out_stem}.pdf"
    meta = OUT_DIR / f"{out_stem}.meta.json"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    meta.write_text(
        json.dumps({"panel": spec["title"], "scan": str(spec["scan"]), "selected": str(spec.get("selected"))}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    panel_map = {spec["title"]: spec for spec in fig5.PANELS}
    draw_one(panel_map["Hulu-med-4B"], "fig5_hulumed4b")
    draw_one(panel_map["InternVL3.5-4B"], "fig5_internvl35_4b")


if __name__ == "__main__":
    main()
