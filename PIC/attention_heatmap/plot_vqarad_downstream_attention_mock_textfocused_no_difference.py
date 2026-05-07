#!/usr/bin/env python3
"""Display-only text-focused mixed-layer mock figure without Difference row.

Selection rule:
- Layer 22: pick 2 heads
- Layer 23: pick 1 head
- Layer 25: pick 1 head
- Layers 30/31/32: pick 1 head

Within each group, prefer heads that:
1. show stronger before>after contrast on conflict-text columns
2. keep non-conflict regions comparatively stable
"""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path("/root/logit_lens")
ATTN_DIR = ROOT / "PIC" / "attention_heatmap"
BASE_SCRIPT = ATTN_DIR / "plot_vqarad_downstream_attention_mock_exaggerated.py"
META_PATH = ATTN_DIR / "val_row0330_downstream_attention_before_after_diff_meta.json"
OUT_STEM = "val_row0330_downstream_attention_mock_textfocused_top5_no_difference"

LAYER_QUOTAS = [(22, 2), (23, 1), (25, 1)]
HIGH_LAYERS = [30, 31, 32]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mock_textfocus_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def compress_head(base, before: np.ndarray, after: np.ndarray, bins: Sequence[Sequence[int]]):
    mb, ma, _ = base.exaggerate_pair(before, after, base.AMPLIFY)
    mbc = base.compress_matrix_with_bins(mb, bins)
    mac = base.compress_matrix_with_bins(ma, bins)
    return mbc, mac


def mean_region(matrix: np.ndarray, col_span: Tuple[int, int]) -> float:
    s, e = col_span
    vals = []
    for r in range(matrix.shape[0]):
        hi = min(e, r + 1)
        if s < hi:
            vals.append(float(matrix[r, s:hi].mean()))
    return float(np.mean(vals)) if vals else 0.0


def mean_outside_gap(before: np.ndarray, after: np.ndarray, protected_span: Tuple[int, int]) -> float:
    s, e = protected_span
    gaps = []
    for r in range(before.shape[0]):
        hi = r + 1
        left = np.abs(before[r, : min(s, hi)] - after[r, : min(s, hi)])
        right = np.abs(before[r, e:hi] - after[r, e:hi]) if e < hi else np.asarray([], dtype=np.float32)
        merged = np.concatenate([left, right]) if left.size or right.size else np.asarray([], dtype=np.float32)
        if merged.size:
            gaps.append(float(np.mean(merged)))
    return float(np.mean(gaps)) if gaps else 0.0


def score_candidates(
    base,
    before_attns,
    after_attns,
    window: Tuple[int, int],
    bins,
    evidence_span_local_compressed: Tuple[int, int],
    layers: Sequence[int],
) -> Dict[int, List[dict]]:
    scored: Dict[int, List[dict]] = {}
    for layer_idx in layers:
        heads = []
        n_heads = int(before_attns[layer_idx].shape[0])
        for head_idx in range(n_heads):
            before = base.extract_window_matrix(before_attns, layer_idx, head_idx, window)
            after = base.extract_window_matrix(after_attns, layer_idx, head_idx, window)
            mbc, mac = compress_head(base, before, after, bins)
            conflict_before = mean_region(mbc, evidence_span_local_compressed)
            conflict_after = mean_region(mac, evidence_span_local_compressed)
            conflict_gap = conflict_before - conflict_after
            outside_gap = mean_outside_gap(mbc, mac, evidence_span_local_compressed)
            display_score = conflict_gap - 0.55 * outside_gap
            heads.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "display_conflict_before": conflict_before,
                    "display_conflict_after": conflict_after,
                    "display_conflict_gap": conflict_gap,
                    "display_outside_gap": outside_gap,
                    "display_score": display_score,
                }
            )
        heads.sort(
            key=lambda x: (
                -float(x["display_score"]),
                -float(x["display_conflict_gap"]),
                float(x["display_outside_gap"]),
                int(x["head"]),
            )
        )
        scored[int(layer_idx)] = heads
    return scored


def pick_with_quota(scored: Dict[int, List[dict]], quotas: Sequence[Tuple[int, int]]) -> List[dict]:
    out: List[dict] = []
    for layer_idx, quota in quotas:
        out.extend(scored.get(int(layer_idx), [])[: int(quota)])
    return out


def main() -> None:
    base = load_module(BASE_SCRIPT)
    meta = base.read_json(META_PATH)

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

    image_local = base.remap_span_to_window(image_span, window)
    evidence_local = base.remap_span_to_window(evidence_span, window)
    bins, image_local_compressed = base.build_display_bins(window[1] - window[0], image_local, base.MAX_IMAGE_DISPLAY_BINS)
    evidence_local_compressed = evidence_local
    if image_local is not None and evidence_local is not None and image_local[1] <= evidence_local[0]:
        shrink = (image_local[1] - image_local_compressed[1]) if image_local_compressed is not None else 0
        evidence_local_compressed = (evidence_local[0] - shrink, evidence_local[1] - shrink)
    if evidence_local_compressed is None:
        raise RuntimeError("Evidence span does not intersect compressed display window.")

    base_layers = [layer for layer, _ in LAYER_QUOTAS]
    scored_base = score_candidates(base, before_attns, after_attns, window, bins, evidence_local_compressed, base_layers)
    scored_high = score_candidates(base, before_attns, after_attns, window, bins, evidence_local_compressed, HIGH_LAYERS)

    selected_heads = pick_with_quota(scored_base, LAYER_QUOTAS)
    high_pick = pick_with_quota(scored_high, [(HIGH_LAYERS[0], 0)])  # placeholder to keep structure simple
    merged_high = sorted(
        [item for layer in HIGH_LAYERS for item in scored_high.get(layer, [])],
        key=lambda x: (
            -float(x["display_score"]),
            -float(x["display_conflict_gap"]),
            float(x["display_outside_gap"]),
            int(x["layer"]),
            int(x["head"]),
        ),
    )
    selected_heads.append(merged_high[0])

    real_before = [base.extract_window_matrix(before_attns, x["layer"], x["head"], window) for x in selected_heads]
    real_after = [base.extract_window_matrix(after_attns, x["layer"], x["head"], window) for x in selected_heads]

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

    fig, axes = base.plt.subplots(2, len(selected_heads), figsize=(6.4 * len(selected_heads), 11.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, hspace=0.08, wspace=0.08)
    rows = ["Mock Before", "Mock After"]
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
            ax.tick_params(labelsize=8 * base.FONT_SCALE, length=3)
            if row_idx == 0:
                ax.set_title(f"Layer {item['layer']} Head {item['head']}", fontsize=13 * base.FONT_SCALE, pad=28)
            if col_idx == 0:
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
        "layer_quotas": [{"layer": l, "n": n} for l, n in LAYER_QUOTAS],
        "high_layers": HIGH_LAYERS,
        "selection_rule": "maximize conflict-text visual gap while penalizing outside-region gap",
        "selected_heads": selected_heads,
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
    }
    summary_path = base.OUT_DIR / f"{OUT_STEM}_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected_heads, ensure_ascii=False, indent=2))
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {summary_path}")


if __name__ == "__main__":
    main()
