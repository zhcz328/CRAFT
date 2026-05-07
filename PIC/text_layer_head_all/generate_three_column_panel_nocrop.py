#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "text_layer_head_all"


def load_curve(path: Path, metric: Optional[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("layer_score_mean"), list):
        vals = payload["layer_score_mean"]
    elif isinstance(payload.get("layer_score_mean"), dict):
        key = metric or "follow_conflict"
        vals = payload["layer_score_mean"][key]
    elif isinstance(payload.get("layer_scores"), dict):
        key = metric or "follow_conflict"
        vals = payload["layer_scores"][key]
    else:
        raise ValueError(f"Unsupported trace json format: {path}")
    return np.asarray([float(v) for v in vals], dtype=float)


def moving_average(y: np.ndarray, window: int = 3) -> np.ndarray:
    if len(y) < 2 or window <= 1:
        return y
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def draw_fig4_panel(ax: plt.Axes, title: str, vqa_path: Path, slake_path: Path) -> None:
    y_vqa = moving_average(load_curve(vqa_path, "follow_conflict"), 3)
    y_slake = moving_average(load_curve(slake_path, "follow_conflict"), 3)
    x_vqa = np.arange(len(y_vqa))
    x_slake = np.arange(len(y_slake))

    ax.plot(x_vqa, y_vqa, color="#2ca02c", linewidth=2.0, marker="o", markersize=3.5, label="VQA-RAD")
    ax.plot(x_slake, y_slake, color="#ff7f0e", linewidth=2.0, marker="o", markersize=3.5, label="Slake-VQA")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Layer ID", fontsize=10)
    ax.set_ylabel("Average value", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False, fontsize=9)


def load_headscan_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def load_selected_set(path: Path) -> set:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected") or payload.get("selected_heads") or []
    return {(int(r["layer"]), int(r["head"])) for r in selected}


def draw_fig5_panel(ax: plt.Axes, title: str, scan_path: Path, selected_path: Path) -> None:
    rows = load_headscan_rows(scan_path)
    selected = load_selected_set(selected_path)
    if not rows:
        ax.set_title(f"{title} (no data)")
        ax.axis("off")
        return

    bcp = np.asarray([max(float(r.get("mean_abs_base_change", 0.0)), 0.0) for r in rows], dtype=float)
    cer = np.asarray([max(float(r.get("mean_abs_effect_reduction", 0.0)), 0.0) for r in rows], dtype=float)
    keys = [(int(r["layer"]), int(r["head"])) for r in rows]
    sel_mask = np.asarray([k in selected for k in keys], dtype=bool)

    # Normalize display to stable ranges.
    bcp_ref = max(float(np.percentile(bcp[bcp > 0], 98)) if np.any(bcp > 0) else 1.0, 1e-6)
    cer_ref = max(float(np.percentile(cer[cer > 0], 98)) if np.any(cer > 0) else 1.0, 1e-6)
    x = bcp / bcp_ref * 0.28
    y = cer / cer_ref * 0.46

    ax.scatter(x[~sel_mask], y[~sel_mask], s=10, c="#d6d6d6", alpha=0.52, edgecolors="none")
    ax.scatter(x[sel_mask], y[sel_mask], s=16, c="#ff3b30", alpha=0.72, edgecolors="none")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("BCP", fontsize=10)
    ax.set_ylabel("CER", fontsize=10)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regenerate 3-column panel without cropping source images.")
    ap.add_argument("--out-prefix", default="fig4_fig5_textlayerhead_3col_nocrop")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    plt.rcParams.update({"font.size": 10})

    # Column 1: regenerate Fig4 (Hulu-med-4B, InternVL3.5-4B)
    draw_fig4_panel(
        axes[0, 0],
        "Fig4 Hulu-med-4B",
        ROOT / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/trace_conflict.json",
        ROOT / "Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_question/trace_conflict.json",
    )
    draw_fig4_panel(
        axes[1, 0],
        "Fig4 InternVL3.5-4B",
        ROOT / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/trace_conflict.json",
        ROOT / "Slake_vqa/text_conflict/internvl35_4b/result_before_question_slake/trace_conflict.json",
    )

    # Column 2: regenerate Fig5 (Hulu-med-4B, InternVL3.5-4B)
    draw_fig5_panel(
        axes[0, 1],
        "Fig5 Hulu-med-4B",
        ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json",
        ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json",
    )
    draw_fig5_panel(
        axes[1, 1],
        "Fig5 InternVL3.5-4B",
        ROOT
        / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json",
        ROOT / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json",
    )

    # Column 3: text_layer_head (already generated, no crop)
    hulu_img = Image.open(ROOT / "PIC/text_layer_head/text_layer_head_hulumed4b.png").convert("RGB")
    intern_img = Image.open(ROOT / "PIC/text_layer_head/text_layer_head_internvl35_4b_filled.png").convert("RGB")
    axes[0, 2].imshow(hulu_img)
    axes[1, 2].imshow(intern_img)
    axes[0, 2].set_title("Text-layer-head Hulu-med-4B", fontsize=12, fontweight="bold")
    axes[1, 2].set_title("Text-layer-head InternVL3.5-4B", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")

    fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.04, wspace=0.16, hspace=0.22)

    out_png = OUT_DIR / f"{args.out_prefix}.png"
    out_pdf = OUT_DIR / f"{args.out_prefix}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

