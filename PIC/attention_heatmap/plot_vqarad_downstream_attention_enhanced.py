#!/usr/bin/env python3
"""Create a visually enhanced downstream-attention comparison figure.

This script reloads the best sample chosen in
`val_row0330_downstream_attention_before_after_diff_meta.json`, recomputes
before/after prompt self-attention, and renders a stronger visualization:

1. Before / After:
   - diagonal suppressed for display
   - robust percentile clipping
   - stronger PowerNorm
2. Difference:
   - relative change: (after - before) / (before + eps)
   - robust clipping
   - top changed cells highlighted
"""

from __future__ import annotations

import csv
import gc
import importlib.machinery
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path("/root/logit_lens")
PROJECT_ROOT = ROOT / "VQA_RAD" / "Hulu-med"
OUT_DIR = ROOT / "PIC" / "attention_heatmap"
META_PATH = OUT_DIR / "val_row0330_downstream_attention_before_after_diff_meta.json"
MODEL_NAME = "/root/autodl-tmp/Hulu-Med-4B"
POSITION = "before_question"
DTYPE = "bf16"
DEVICE = "cuda"
MAX_IMAGE_SIDE = 672


def install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return
    stub = types.ModuleType("pandas")
    stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)

    def isna(value):
        try:
            return value != value
        except Exception:
            return value is None

    stub.isna = isna
    stub.notna = lambda value: not isna(value)
    stub.DataFrame = object
    stub.Series = object
    sys.modules["pandas"] = stub


install_pandas_stub()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import head_scan_vqarad_mm_fastcache as hs  # type: ignore  # noqa: E402
import plot_cross_attention_nc_ic as vis  # type: ignore  # noqa: E402


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_row(path: Path, row_idx: int) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[int(row_idx)]


def lower_triangle_mask(shape: Tuple[int, int], include_diagonal: bool = True) -> np.ndarray:
    k = 0 if include_diagonal else -1
    return np.tril(np.ones(shape, dtype=bool), k=k)


def lower_triangle_values(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray:
    return matrix[lower_triangle_mask(matrix.shape, include_diagonal=include_diagonal)]


def masked_upper(matrix: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_array(matrix, mask=np.triu(np.ones(matrix.shape, dtype=bool), k=1))


def suppress_diagonal_display(matrix: np.ndarray, width: int = 1) -> np.ndarray:
    out = matrix.copy()
    n = min(out.shape[0], out.shape[1])
    for offset in range(width):
        idx = np.arange(0, n - offset)
        out[idx + offset, idx] = np.nan
    return out


@torch.no_grad()
def capture_prompt_attentions(model, inputs) -> List[torch.Tensor]:
    target_dtype = hs.infer_vision_input_dtype(model)
    fixed_inputs = hs.move_to_device(inputs, next(model.parameters()).device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=False, output_attentions=True, return_dict=True)
    attentions = [layer[0].detach().float().cpu() for layer in out.attentions]
    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attentions


def extract_window_matrix(attns: Sequence[torch.Tensor], layer_idx: int, head_idx: int, window: Tuple[int, int]) -> np.ndarray:
    start, end = window
    return np.asarray(attns[layer_idx][head_idx, start:end, start:end].numpy(), dtype=np.float32)


def build_inputs(model, processor, meta: dict):
    sample = meta["sample"]
    question = str(sample["question"])
    wrong = str(sample["wrong"]).strip()
    image_path = Path(sample["image_path"])
    image = hs.resize_image_max_side(Image.open(image_path), max_side=MAX_IMAGE_SIDE)
    prompt = hs.build_ctx_prompt(question, wrong, position=POSITION)
    marked_prompt = vis.build_ctx_prompt_marked(question, wrong, position=POSITION)
    pack = vis.build_inputs_mm_with_spans(
        processor,
        prompt,
        image,
        next(model.parameters()).device,
        model,
        marked_prompt=marked_prompt,
    )
    return pack


def plot_enhanced(
    selected_heads: Sequence[dict],
    before_attns: Sequence[torch.Tensor],
    after_attns: Sequence[torch.Tensor],
    window: Tuple[int, int],
    out_png: Path,
    out_pdf: Path,
    title: str,
) -> None:
    before_mats = [extract_window_matrix(before_attns, x["layer"], x["head"], window) for x in selected_heads]
    after_mats = [extract_window_matrix(after_attns, x["layer"], x["head"], window) for x in selected_heads]
    raw_diff_mats = [after - before for before, after in zip(before_mats, after_mats)]
    rel_diff_mats = [(after - before) / np.maximum(before, 1e-4) for before, after in zip(before_mats, after_mats)]

    disp_before = [suppress_diagonal_display(x, width=2) for x in before_mats]
    disp_after = [suppress_diagonal_display(x, width=2) for x in after_mats]

    non_diag_vals = np.concatenate([
        lower_triangle_values(x, include_diagonal=False) for x in before_mats + after_mats
    ])
    seq_vmin = float(np.percentile(non_diag_vals, 12))
    seq_vmax = float(np.percentile(non_diag_vals, 96.5))
    seq_norm = PowerNorm(gamma=0.45, vmin=seq_vmin, vmax=seq_vmax)

    rel_vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in rel_diff_mats])
    rel_abs = float(np.percentile(np.abs(rel_vals), 97.5))
    rel_abs = max(rel_abs, 1e-3)
    rel_norm = TwoSlopeNorm(vmin=-rel_abs, vcenter=0.0, vmax=rel_abs)

    seq_cmap = plt.get_cmap("magma").copy()
    seq_cmap.set_bad("#b0b0b0")
    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad("#b0b0b0")

    fig, axes = plt.subplots(3, len(selected_heads), figsize=(4.7 * len(selected_heads), 12.5), constrained_layout=True)
    if len(selected_heads) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    row_labels = ["Before Ablation", "After Ablation", "Relative Difference"]
    plot_mats = [disp_before, disp_after, rel_diff_mats]
    plot_norms = [seq_norm, seq_norm, rel_norm]
    plot_cmaps = [seq_cmap, seq_cmap, diff_cmap]

    n_tokens = window[1] - window[0]
    tick_positions = list(range(0, n_tokens, max(1, math.ceil(n_tokens / 6))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    for col_idx, item in enumerate(selected_heads):
        highlight_scores = np.abs(lower_triangle_values(raw_diff_mats[col_idx], include_diagonal=False))
        thresh = float(np.percentile(highlight_scores, 97.5)) if highlight_scores.size else 0.0
        raw_abs = np.abs(raw_diff_mats[col_idx])
        coords = np.argwhere(np.tril(raw_abs >= thresh, k=-1)) if thresh > 0 else np.zeros((0, 2), dtype=int)
        coords = coords[:12]
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(masked_upper(plot_mats[row_idx][col_idx]), cmap=plot_cmaps[row_idx], norm=plot_norms[row_idx], interpolation="nearest")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=8, length=2)
            if row_idx == 0:
                ax.set_title(f"Layer {item['layer']} Head {item['head']}", fontsize=13, pad=8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=12)
            if thresh > 0:
                edge = "#111111" if row_idx < 2 else "#000000"
                lw = 1.0 if row_idx < 2 else 1.1
                for y, x in coords:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=edge, linewidth=lw))
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = read_json(META_PATH)
    selected_heads = meta["selection"]["selected_downstream_heads"]
    window = (int(meta["plot_window"]["start"]), int(meta["plot_window"]["end"]))
    ablated = meta["selection"]["ablated_conflict_heads"]

    layer2heads: Dict[int, List[int]] = {}
    for item in ablated:
        layer2heads.setdefault(int(item["layer"]), []).append(int(item["head"]))

    model, processor = hs.load_model_and_processor(MODEL_NAME, DEVICE, DTYPE)
    pack = build_inputs(model, processor, meta)
    before_attns = capture_prompt_attentions(model, pack["inputs"])
    handles = hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_attns = capture_prompt_attentions(model, pack["inputs"])
    finally:
        hs.remove_hooks(handles)

    out_png = OUT_DIR / "val_row0330_downstream_attention_before_after_diff_enhanced.png"
    out_pdf = OUT_DIR / "val_row0330_downstream_attention_before_after_diff_enhanced.pdf"
    title = "VQA-RAD Hulu-Med-4B Downstream Attention Routing (Enhanced Contrast, val row 330)"
    plot_enhanced(selected_heads, before_attns, after_attns, window, out_png, out_pdf, title)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
