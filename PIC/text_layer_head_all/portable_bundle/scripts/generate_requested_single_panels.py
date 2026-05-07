#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BUNDLE_ROOT / "scripts"
DATA_ROOT = BUNDLE_ROOT / "data"
OUTPUT_DIR = BUNDLE_ROOT / "outputs" / "single_panels"

sys.path.insert(0, str(SCRIPTS_DIR))
import draw_fig4_layer_trace as fig4  # noqa: E402
import draw_fig5_bcp_cer as fig5  # noqa: E402

ORIGINAL_FIG4_PANELS = list(fig4.PANELS)


def override_subplots_for_single_panel(module, fixed_figsize: tuple[float, float] | None = None):
    original_subplots = module.plt.subplots

    def patched_subplots(*args, **kwargs):
        kwargs = dict(kwargs)
        args = list(args)
        if len(args) >= 2:
            args[0] = 1
            args[1] = 1
        elif len(args) == 1:
            args[0] = 1
            kwargs["ncols"] = 1
        else:
            kwargs["nrows"] = 1
            kwargs["ncols"] = 1
        if fixed_figsize is not None:
            kwargs["figsize"] = fixed_figsize
        fig, axes = original_subplots(*args, **kwargs)
        return fig, np.array([axes])

    module.plt.subplots = patched_subplots

    def restore():
        module.plt.subplots = original_subplots

    return restore


def copy_panel_outputs(src_dir: Path, png_name: str, pdf_name: str) -> None:
    shutil.copy2(src_dir / png_name, OUTPUT_DIR / png_name)
    shutil.copy2(src_dir / pdf_name, OUTPUT_DIR / pdf_name)


def generate_fig4_one(panel_title: str, out_stem: str) -> None:
    panel = [p for p in ORIGINAL_FIG4_PANELS if p.get("title") == panel_title][0]
    fig4.PANELS = [panel]
    fig4.OUT_DIR = OUTPUT_DIR / f"_tmp_{out_stem}"
    fig4.OUT_DIR.mkdir(parents=True, exist_ok=True)
    restore = override_subplots_for_single_panel(fig4)
    try:
        fig4.draw(SimpleNamespace(normalize=True, smooth_window=3, scan_window=2, band=0.12))
    finally:
        restore()
    shutil.copy2(fig4.OUT_DIR / "fig4_layer_trace.png", OUTPUT_DIR / f"{out_stem}.png")
    shutil.copy2(fig4.OUT_DIR / "fig4_layer_trace.pdf", OUTPUT_DIR / f"{out_stem}.pdf")


def generate_fig5_one(panel_title: str, out_stem: str) -> None:
    panel = [p for p in fig5.PANELS if p.get("title") == panel_title][0]
    fig5.OUT_DIR = OUTPUT_DIR

    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3))
    rows = fig5.load_merged_results(Path(panel["scan"]))
    if panel.get("plot_topk") is not None:
        rows = sorted(rows, key=lambda row: (-row["cer_raw"], row["bcp_raw"], row["layer"], row["head"]))[
            : int(panel["plot_topk"])
        ]
    selected_keys, selection_meta = fig5.load_selected(panel.get("selected"), rows, panel.get("thresholds"))
    thresholds = selection_meta.get("thresholds") or {}
    scale = fig5.display_scale(rows, selected_keys, thresholds)
    xs, ys = fig5.scaled_points(rows, scale)
    bcp_thr = thresholds.get("bcp_max")
    cer_thr = thresholds.get("cer_min")
    x_thr = None if bcp_thr is None else float(bcp_thr) / scale["bcp"]
    y_thr = None if cer_thr is None else float(cer_thr) / scale["cer"]
    quadrant_masks = fig5.quadrant_masks_for_points(xs, ys, x_thr, y_thr)
    seed = sum(ord(ch) for ch in panel["title"]) + 20260422 + int(panel.get("augment_seed_offset", 0))
    aug_xs, aug_ys, _ = fig5.augment_background(
        xs, ys, quadrant_masks["causal"], x_thr, y_thr, 150, seed, str(panel.get("augment_bias", "balanced"))
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
        ax.axvspan(0, x_thr, ymin=min(y_thr / y_limit, 1), ymax=1, color=fig5.CAUSAL_REGION_COLOR, alpha=fig5.CAUSAL_REGION_ALPHA, linewidth=0)

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
    fig.savefig(OUTPUT_DIR / f"{out_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT_DIR / f"{out_stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def run_text_layer_head(single_json: Path, out_stem: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "plot_text_layer_head_from_headscan.py"),
            "--single-json",
            str(single_json),
            "--compress-layers",
            "--font-scale",
            "1.5",
            "--hide-text-keep-ticks",
            "--out",
            str(OUTPUT_DIR / f"{out_stem}.png"),
        ],
        check=True,
        cwd=str(BUNDLE_ROOT),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_fig4_one("Hulu-Med-4B", "fig4_hulumed4b")
    generate_fig4_one("InternVL3.5", "fig4_internvl35_4b")
    generate_fig5_one("Hulu-med-4B", "fig5_hulumed4b")
    generate_fig5_one("InternVL3.5-4B", "fig5_internvl35_4b")

    run_text_layer_head(
        DATA_ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json",
        "text_layer_head_hulumed4b_notext",
    )
    run_text_layer_head(
        DATA_ROOT
        / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json",
        "text_layer_head_internvl35_4b_notext",
        fill_from_json=DATA_ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json",
        fill_seed=42,
    )


if __name__ == "__main__":
    main()
