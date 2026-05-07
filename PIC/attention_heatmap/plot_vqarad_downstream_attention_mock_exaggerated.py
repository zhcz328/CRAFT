#!/usr/bin/env python3
"""Build a display-only exaggerated before/after attention figure.

This is NOT a result figure. It amplifies the real before/after delta so the
user can visually inspect where the differences mainly occur.
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
AMPLIFY = 10.0
FONT_SCALE = 1.5
N_HEADS = 5
WINDOW_PAD = 2
MAX_IMAGE_DISPLAY_BINS = 18


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


def lower_triangle_values(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray:
    k = 0 if include_diagonal else -1
    mask = np.tril(np.ones(matrix.shape, dtype=bool), k=k)
    return matrix[mask]


def masked_upper(matrix: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_array(matrix, mask=np.triu(np.ones(matrix.shape, dtype=bool), k=1))


def suppress_diagonal_display(matrix: np.ndarray, width: int = 2) -> np.ndarray:
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
    return vis.build_inputs_mm_with_spans(
        processor,
        prompt,
        image,
        next(model.parameters()).device,
        model,
        marked_prompt=marked_prompt,
    )


def choose_union_window(pack: dict, pad: int = 2) -> Tuple[int, int]:
    prompt_len = len(pack["input_ids"])
    image_span = tuple(int(x) for x in pack.get("image_span", (0, 0)))
    evidence_span = tuple(int(x) for x in pack.get("evidence_span", (0, 0)))
    anchors = [image_span[0], evidence_span[0], image_span[1], evidence_span[1]]
    start = max(0, min(anchors) - pad)
    end = min(prompt_len, max(anchors) + pad)
    if end <= start:
        return (0, min(prompt_len, 32))
    return (start, end)


def remap_span_to_window(span: Tuple[int, int], window: Tuple[int, int]) -> Tuple[int, int] | None:
    start, end = span
    w0, w1 = window
    lo = max(start, w0)
    hi = min(end, w1)
    if hi <= lo:
        return None
    return (lo - w0, hi - w0)


def build_display_bins(
    window_len: int,
    image_local: Tuple[int, int] | None,
    max_image_bins: int,
) -> Tuple[List[List[int]], Tuple[int, int] | None]:
    if image_local is None:
        bins = [[i] for i in range(window_len)]
        return bins, None

    img0, img1 = image_local
    image_positions = list(range(img0, img1))
    if len(image_positions) <= max_image_bins:
        bins = [[i] for i in range(window_len)]
        return bins, image_local

    bins: List[List[int]] = []
    for i in range(img0):
        bins.append([i])

    chunk_edges = np.linspace(0, len(image_positions), num=max_image_bins + 1, dtype=int)
    for left, right in zip(chunk_edges[:-1], chunk_edges[1:]):
        group = image_positions[left:right]
        if group:
            bins.append(group)

    for i in range(img1, window_len):
        bins.append([i])

    new_img0 = img0
    new_img1 = img0 + max_image_bins
    return bins, (new_img0, new_img1)


def compress_matrix_with_bins(matrix: np.ndarray, bins: Sequence[Sequence[int]]) -> np.ndarray:
    n = len(bins)
    out = np.zeros((n, n), dtype=np.float32)
    for i, row_group in enumerate(bins):
        for j, col_group in enumerate(bins):
            if j > i:
                out[i, j] = 0.0
                continue
            block = matrix[np.ix_(list(row_group), list(col_group))]
            out[i, j] = float(np.mean(block)) if block.size else 0.0
    return out


def renorm_rows_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    n = out.shape[0]
    for r in range(n):
        vals = out[r, : r + 1]
        vals = np.clip(vals, 0.0, None)
        s = float(vals.sum())
        if s > 0:
            out[r, : r + 1] = vals / s
        else:
            out[r, : r + 1] = matrix[r, : r + 1]
        out[r, r + 1 :] = 0.0
    return out


def exaggerate_pair(before: np.ndarray, after: np.ndarray, amplify: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = after - before
    before_mock = before - 0.5 * (amplify - 1.0) * delta
    after_mock = after + 0.5 * (amplify - 1.0) * delta
    before_mock = renorm_rows_lower_triangle(before_mock)
    after_mock = renorm_rows_lower_triangle(after_mock)
    return before_mock, after_mock, after_mock - before_mock


def add_span_markers(ax, image_span_local: Tuple[int, int] | None, evidence_span_local: Tuple[int, int] | None) -> None:
    if image_span_local is not None:
        x0, x1 = image_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#2f6df6", alpha=0.22, lw=1.4, ec="#1d4ed8")
        ax.text(
            (x0 + x1 - 1) / 2.0,
            -1.25,
            "Image tokens",
            color="#1d4ed8",
            fontsize=8.5 * FONT_SCALE,
            ha="center",
            va="bottom",
            clip_on=False,
        )
    if evidence_span_local is not None:
        x0, x1 = evidence_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#d62728", alpha=0.22, lw=1.4, ec="#b91c1c")
        ax.text(
            (x0 + x1 - 1) / 2.0,
            -3.05,
            "Conflict text",
            color="#b91c1c",
            fontsize=8.5 * FONT_SCALE,
            ha="center",
            va="bottom",
            clip_on=False,
        )


def plot_mock(
    selected_heads: Sequence[dict],
    before_attns: Sequence[torch.Tensor],
    after_attns: Sequence[torch.Tensor],
    window: Tuple[int, int],
    image_span: Tuple[int, int],
    evidence_span: Tuple[int, int],
) -> None:
    real_before = [extract_window_matrix(before_attns, x["layer"], x["head"], window) for x in selected_heads]
    real_after = [extract_window_matrix(after_attns, x["layer"], x["head"], window) for x in selected_heads]
    image_local = remap_span_to_window(image_span, window)
    evidence_local = remap_span_to_window(evidence_span, window)
    bins, image_local_compressed = build_display_bins(window[1] - window[0], image_local, MAX_IMAGE_DISPLAY_BINS)
    evidence_local_compressed = evidence_local
    if image_local is not None and evidence_local is not None and image_local[1] <= evidence_local[0]:
        shrink = (image_local[1] - image_local_compressed[1]) if image_local_compressed is not None else 0
        evidence_local_compressed = (evidence_local[0] - shrink, evidence_local[1] - shrink)

    mock_before = []
    mock_after = []
    mock_diff = []
    for b, a in zip(real_before, real_after):
        mb, ma, md = exaggerate_pair(b, a, AMPLIFY)
        mock_before.append(compress_matrix_with_bins(mb, bins))
        mock_after.append(compress_matrix_with_bins(ma, bins))
        mock_diff.append(compress_matrix_with_bins(md, bins))

    vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in mock_before + mock_after])
    vmin = float(np.percentile(vals, 10))
    vmax = float(np.percentile(vals, 96))
    seq_norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

    diff_vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in mock_diff])
    diff_abs = float(np.percentile(np.abs(diff_vals), 97))
    diff_abs = max(diff_abs, 1e-3)
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)

    seq_cmap = plt.get_cmap("coolwarm").copy()
    seq_cmap.set_bad("#b0b0b0")
    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad("#b0b0b0")

    fig, axes = plt.subplots(
        3,
        len(selected_heads),
        figsize=(6.4 * len(selected_heads), 16.8),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, hspace=0.08, wspace=0.08)
    rows = ["Mock Before", "Mock After", "Mock Difference"]
    n_tokens = len(bins)
    tick_positions = list(range(0, n_tokens, max(1, math.ceil(n_tokens / 6))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    for col_idx, item in enumerate(selected_heads):
        mats = [
            suppress_diagonal_display(mock_before[col_idx]),
            suppress_diagonal_display(mock_after[col_idx]),
            mock_diff[col_idx],
        ]
        norms = [seq_norm, seq_norm, diff_norm]
        cmaps = [seq_cmap, seq_cmap, diff_cmap]
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(masked_upper(mats[row_idx]), cmap=cmaps[row_idx], norm=norms[row_idx], interpolation="nearest")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=8 * FONT_SCALE, length=3)
            if row_idx == 0:
                ax.set_title(
                    f"Layer {item['layer']} Head {item['head']}",
                    fontsize=13 * FONT_SCALE,
                    pad=28,
                )
            if col_idx == 0:
                ax.set_ylabel(rows[row_idx], fontsize=12 * FONT_SCALE)
            add_span_markers(ax, image_local_compressed, evidence_local_compressed)
            cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
            cbar.ax.tick_params(labelsize=8 * FONT_SCALE, length=3)

    out_png = OUT_DIR / "val_row0330_downstream_attention_mock_exaggerated_top5.png"
    out_pdf = OUT_DIR / "val_row0330_downstream_attention_mock_exaggerated_top5.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


def main() -> None:
    meta = read_json(META_PATH)
    selected_heads = meta["selection"]["top_ranked_candidates"][:N_HEADS]
    ablated = meta["selection"]["ablated_conflict_heads"]

    layer2heads: Dict[int, List[int]] = {}
    for item in ablated:
        layer2heads.setdefault(int(item["layer"]), []).append(int(item["head"]))

    model, processor = hs.load_model_and_processor(MODEL_NAME, DEVICE, DTYPE)
    pack = build_inputs(model, processor, meta)
    window = choose_union_window(pack, pad=WINDOW_PAD)
    image_span = tuple(int(x) for x in pack.get("image_span", (0, 0)))
    evidence_span = tuple(int(x) for x in pack.get("evidence_span", (0, 0)))
    before_attns = capture_prompt_attentions(model, pack["inputs"])
    handles = hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_attns = capture_prompt_attentions(model, pack["inputs"])
    finally:
        hs.remove_hooks(handles)
    plot_mock(selected_heads, before_attns, after_attns, window, image_span, evidence_span)


if __name__ == "__main__":
    main()
