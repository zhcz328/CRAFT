#!/usr/bin/env python3
"""Display-only exaggerated single-head figure for the strongest 30+ layer head."""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


ROOT = Path("/root/logit_lens")
ATTN_DIR = ROOT / "PIC" / "attention_heatmap"
BASE_SCRIPT = ATTN_DIR / "plot_vqarad_downstream_attention_mock_exaggerated.py"
META_PATH = ATTN_DIR / "val_row0330_downstream_attention_before_after_diff_meta.json"
OUT_STEM = "val_row0330_downstream_attention_mock_exaggerated_single_highlayer"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mock_base_single", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def choose_best_highlayer(meta: dict) -> dict:
    candidates = meta["selection"]["top_ranked_candidates"]
    high = [x for x in candidates if int(x["layer"]) >= 30]
    if not high:
        raise RuntimeError("No 30+ layer heads found in top_ranked_candidates.")
    return sorted(high, key=lambda x: (-float(x["mean_abs_delta"]), int(x["layer"]), int(x["head"])))[0]


def main() -> None:
    base = load_module(BASE_SCRIPT)
    meta = base.read_json(META_PATH)
    chosen = choose_best_highlayer(meta)
    selected_heads = [chosen]

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

    real_before = [base.extract_window_matrix(before_attns, chosen["layer"], chosen["head"], window)]
    real_after = [base.extract_window_matrix(after_attns, chosen["layer"], chosen["head"], window)]
    image_local = base.remap_span_to_window(image_span, window)
    evidence_local = base.remap_span_to_window(evidence_span, window)
    bins, image_local_compressed = base.build_display_bins(window[1] - window[0], image_local, base.MAX_IMAGE_DISPLAY_BINS)
    evidence_local_compressed = evidence_local
    if image_local is not None and evidence_local is not None and image_local[1] <= evidence_local[0]:
        shrink = (image_local[1] - image_local_compressed[1]) if image_local_compressed is not None else 0
        evidence_local_compressed = (evidence_local[0] - shrink, evidence_local[1] - shrink)

    mock_before = []
    mock_after = []
    mock_diff = []
    for b, a in zip(real_before, real_after):
        mb, ma, md = base.exaggerate_pair(b, a, base.AMPLIFY)
        mock_before.append(base.compress_matrix_with_bins(mb, bins))
        mock_after.append(base.compress_matrix_with_bins(ma, bins))
        mock_diff.append(base.compress_matrix_with_bins(md, bins))

    vals = np.concatenate([base.lower_triangle_values(x, include_diagonal=False) for x in mock_before + mock_after])
    vmin = float(np.percentile(vals, 10))
    vmax = float(np.percentile(vals, 96))
    seq_norm = base.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

    diff_vals = np.concatenate([base.lower_triangle_values(x, include_diagonal=False) for x in mock_diff])
    diff_abs = float(np.percentile(np.abs(diff_vals), 97))
    diff_abs = max(diff_abs, 1e-3)
    diff_norm = base.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)

    seq_cmap = base.plt.get_cmap("coolwarm").copy()
    seq_cmap.set_bad("#b0b0b0")
    diff_cmap = base.plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad("#b0b0b0")

    fig, axes = base.plt.subplots(3, 1, figsize=(7.6, 15.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, hspace=0.08, wspace=0.08)
    rows = ["Mock Before", "Mock After", "Mock Difference"]
    n_tokens = len(bins)
    tick_positions = list(range(0, n_tokens, max(1, int(np.ceil(n_tokens / 6)))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    mats = [
        base.suppress_diagonal_display(mock_before[0]),
        base.suppress_diagonal_display(mock_after[0]),
        mock_diff[0],
    ]
    norms = [seq_norm, seq_norm, diff_norm]
    cmaps = [seq_cmap, seq_cmap, diff_cmap]
    for row_idx in range(3):
        ax = axes[row_idx]
        im = ax.imshow(base.masked_upper(mats[row_idx]), cmap=cmaps[row_idx], norm=norms[row_idx], interpolation="nearest")
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(labelsize=8 * base.FONT_SCALE, length=3)
        if row_idx == 0:
            ax.set_title(f"Layer {chosen['layer']} Head {chosen['head']}", fontsize=13 * base.FONT_SCALE, pad=28)
        ax.set_ylabel(rows[row_idx], fontsize=12 * base.FONT_SCALE)
        base.add_span_markers(ax, image_local_compressed, evidence_local_compressed)
        cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
        cbar.ax.tick_params(labelsize=8 * base.FONT_SCALE, length=3)

    out_png = base.OUT_DIR / f"{OUT_STEM}.png"
    out_pdf = base.OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    base.plt.close(fig)

    summary = {
        "selected_head": chosen,
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
    }
    summary_path = base.OUT_DIR / f"{OUT_STEM}_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(chosen, ensure_ascii=False, indent=2))
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {summary_path}")


if __name__ == "__main__":
    main()
