#!/usr/bin/env python3
"""Display-only mixed-layer mock figure without the Difference row."""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


ROOT = Path("/root/logit_lens")
ATTN_DIR = ROOT / "PIC" / "attention_heatmap"
BASE_SCRIPT = ATTN_DIR / "plot_vqarad_downstream_attention_mock_exaggerated.py"
META_PATH = ATTN_DIR / "val_row0330_downstream_attention_before_after_diff_meta.json"
MIXED_META_PATH = ATTN_DIR / "val_row0330_downstream_attention_mock_exaggerated_mixedlayers_top5_meta.json"
OUT_STEM = "val_row0330_downstream_attention_mock_exaggerated_mixedlayers_top5_no_difference"
TEXT_BOOST = 2.25


def add_aligned_span_markers(base, ax, image_span_local, evidence_span_local) -> None:
    if image_span_local is not None:
        x0, x1 = image_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#2f6df6", alpha=0.22, lw=1.4, ec="#1d4ed8")
        ax.text(
            (x0 + x1 - 1) / 2.0,
            1.035,
            "Image tokens",
            color="#1d4ed8",
            fontsize=8.5 * base.FONT_SCALE * TEXT_BOOST,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )
    if evidence_span_local is not None:
        x0, x1 = evidence_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#d62728", alpha=0.22, lw=1.4, ec="#b91c1c")
        ax.text(
            (x0 + x1 - 1) / 2.0,
            1.035,
            "Conflict text",
            color="#b91c1c",
            fontsize=8.5 * base.FONT_SCALE * TEXT_BOOST,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mock_base_nodiff", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main() -> None:
    base = load_module(BASE_SCRIPT)
    meta = base.read_json(META_PATH)
    mixed_meta = json.loads(MIXED_META_PATH.read_text(encoding="utf-8"))
    selected_heads = mixed_meta["selected_heads"]

    ablated = meta["selection"]["ablated_conflict_heads"]
    layer2heads: Dict[int, List[int]] = {}
    for item in ablated:
        layer2heads.setdefault(int(item["layer"]), []).append(int(item["head"]))

    model, processor = base.hs.load_model_and_processor(base.MODEL_NAME, base.DEVICE, base.DTYPE)
    pack = base.build_inputs(model, processor, meta)
    window = base.choose_union_window(pack, pad=base.WINDOW_PAD)
    image_span = tuple(int(x) for x in pack.get("image_span", (0, 0)))
    evidence_span = tuple(int(x) for x in pack.get("evidence_span", (0, 0)))

    before_attns = base.capture_prompt_attentions(model, pack["inputs"])
    handles = base.hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_attns = base.capture_prompt_attentions(model, pack["inputs"])
    finally:
        base.hs.remove_hooks(handles)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    real_before = [base.extract_window_matrix(before_attns, x["layer"], x["head"], window) for x in selected_heads]
    real_after = [base.extract_window_matrix(after_attns, x["layer"], x["head"], window) for x in selected_heads]

    image_local = base.remap_span_to_window(image_span, window)
    evidence_local = base.remap_span_to_window(evidence_span, window)
    bins, image_local_compressed = base.build_display_bins(window[1] - window[0], image_local, base.MAX_IMAGE_DISPLAY_BINS)
    evidence_local_compressed = evidence_local
    if image_local is not None and evidence_local is not None and image_local[1] <= evidence_local[0]:
        shrink = (image_local[1] - image_local_compressed[1]) if image_local_compressed is not None else 0
        evidence_local_compressed = (evidence_local[0] - shrink, evidence_local[1] - shrink)

    mock_before = []
    mock_after = []
    for b, a in zip(real_before, real_after):
        mb, ma, _ = base.exaggerate_pair(b, a, base.AMPLIFY)
        mock_before.append(base.compress_matrix_with_bins(mb, bins))
        mock_after.append(base.compress_matrix_with_bins(ma, bins))

    vals = np.concatenate([base.lower_triangle_values(x, include_diagonal=False) for x in mock_before + mock_after])
    vmin = float(np.percentile(vals, 10))
    vmax = float(np.percentile(vals, 96))
    seq_norm = base.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

    seq_cmap = base.plt.get_cmap("coolwarm").copy()
    seq_cmap.set_bad("#b0b0b0")

    fig, axes = base.plt.subplots(2, len(selected_heads), figsize=(7.4 * len(selected_heads), 13.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.28, hspace=0.10, wspace=0.10)
    rows = ["Before Ablation", "After Ablation"]
    n_tokens = len(bins)
    tick_positions = list(range(0, n_tokens, max(1, int(np.ceil(n_tokens / 6)))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    for col_idx, item in enumerate(selected_heads):
        mats = [
            base.suppress_diagonal_display(mock_before[col_idx]),
            base.suppress_diagonal_display(mock_after[col_idx]),
        ]
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(base.masked_upper(mats[row_idx]), cmap=seq_cmap, norm=seq_norm, interpolation="nearest")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=8 * base.FONT_SCALE * TEXT_BOOST, length=3.5)
            if row_idx == 0:
                ax.set_title(
                    f"Layer {item['layer']} Head {item['head']}",
                    fontsize=13 * base.FONT_SCALE * TEXT_BOOST,
                    pad=58,
                )
            if col_idx == 0:
                ax.set_ylabel(rows[row_idx], fontsize=12 * base.FONT_SCALE * TEXT_BOOST)
            add_aligned_span_markers(base, ax, image_local_compressed, evidence_local_compressed)
            cbar = fig.colorbar(im, ax=ax, fraction=0.062, pad=0.04)
            cbar.ax.tick_params(labelsize=8 * base.FONT_SCALE * TEXT_BOOST, length=3.5)

    out_png = base.OUT_DIR / f"{OUT_STEM}.png"
    out_pdf = base.OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    base.plt.close(fig)

    summary = {
        "selected_heads": selected_heads,
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
        "source_mixed_meta": str(MIXED_META_PATH),
    }
    summary_path = base.OUT_DIR / f"{OUT_STEM}_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {summary_path}")


if __name__ == "__main__":
    main()
