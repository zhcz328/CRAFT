#!/usr/bin/env python3
"""Display-only mixed-layer downstream attention mock figure.

Layer quota:
- Layer 22: top 2 changed heads
- Layer 23: top 1 changed head
- Layer 24: top 1 changed head
- One 30+ layer head chosen by strongest conflict-text attention drop
"""

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

LAYER_QUOTAS = [(22, 2), (23, 1), (24, 1)]
OUT_STEM = "val_row0330_downstream_attention_mock_exaggerated_mixedlayers_top5"
CONFLICT_VISUAL_BOOST = 3.0


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("mock_base", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def score_heads_in_layers(
    base,
    before_attns,
    after_attns,
    window: Tuple[int, int],
    layers: List[int],
) -> Dict[int, List[dict]]:
    scored: Dict[int, List[dict]] = {}
    for layer_idx in layers:
        heads = []
        n_heads = int(before_attns[layer_idx].shape[0])
        for head_idx in range(n_heads):
            before = base.extract_window_matrix(before_attns, layer_idx, head_idx, window)
            after = base.extract_window_matrix(after_attns, layer_idx, head_idx, window)
            delta = after - before
            score = float(np.mean(np.abs(base.lower_triangle_values(delta))))
            heads.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "mean_abs_delta": score,
                }
            )
        heads.sort(key=lambda x: (-x["mean_abs_delta"], x["head"]))
        scored[int(layer_idx)] = heads
    return scored


def pick_heads_by_quota(scored: Dict[int, List[dict]], quotas: List[Tuple[int, int]]) -> List[dict]:
    selected: List[dict] = []
    for layer_idx, quota in quotas:
        selected.extend(scored.get(int(layer_idx), [])[: int(quota)])
    return selected


def choose_best_highlayer_by_conflict_drop(
    base,
    before_attns,
    after_attns,
    window: Tuple[int, int],
    evidence_span: Tuple[int, int],
    image_span: Tuple[int, int],
    candidate_layers: List[int],
) -> dict:
    ev_local = base.remap_span_to_window(evidence_span, window)
    if ev_local is None:
        raise RuntimeError("Evidence span does not intersect the plotting window.")
    s, e = ev_local
    img_local = base.remap_span_to_window(image_span, window)
    bins, img_comp = base.build_display_bins(window[1] - window[0], img_local, base.MAX_IMAGE_DISPLAY_BINS)
    ev_comp = ev_local
    if img_local is not None and ev_local is not None and img_local[1] <= ev_local[0]:
        shrink = (img_local[1] - img_comp[1]) if img_comp is not None else 0
        ev_comp = (ev_local[0] - shrink, ev_local[1] - shrink)
    rows = []
    for layer_idx in candidate_layers:
        n_heads = int(before_attns[layer_idx].shape[0])
        for head_idx in range(n_heads):
            before = base.extract_window_matrix(before_attns, layer_idx, head_idx, window)
            after = base.extract_window_matrix(after_attns, layer_idx, head_idx, window)
            delta = after - before
            score = float(np.mean(np.abs(base.lower_triangle_values(delta))))
            before_mass = float(np.mean([before[r, s:min(e, r + 1)].sum() if s < min(e, r + 1) else 0.0 for r in range(before.shape[0])]))
            after_mass = float(np.mean([after[r, s:min(e, r + 1)].sum() if s < min(e, r + 1) else 0.0 for r in range(after.shape[0])]))
            mb, ma, _ = base.exaggerate_pair(before, after, base.AMPLIFY)
            mbc = base.compress_matrix_with_bins(mb, bins)
            mac = base.compress_matrix_with_bins(ma, bins)
            cs, ce = ev_comp
            display_before = float(np.mean([mbc[r, cs:min(ce, r + 1)].mean() if cs < min(ce, r + 1) else 0.0 for r in range(mbc.shape[0])]))
            display_after = float(np.mean([mac[r, cs:min(ce, r + 1)].mean() if cs < min(ce, r + 1) else 0.0 for r in range(mac.shape[0])]))
            rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "mean_abs_delta": score,
                    "before_conflict_mass": before_mass,
                    "after_conflict_mass": after_mass,
                    "conflict_drop": before_mass - after_mass,
                    "display_before_conflict_mean": display_before,
                    "display_after_conflict_mean": display_after,
                    "display_gap": display_before - display_after,
                }
            )
    positive = [x for x in rows if float(x["conflict_drop"]) > 0]
    if positive:
        positive.sort(
            key=lambda x: (
                -float(x["display_gap"]),
                -float(x["display_before_conflict_mean"]),
                -float(x["conflict_drop"]),
                int(x["layer"]),
                int(x["head"]),
            )
        )
        return positive[0]
    rows.sort(key=lambda x: (-float(x["mean_abs_delta"]), int(x["layer"]), int(x["head"])))
    return rows[0]


def apply_conflict_visual_boost(
    base,
    before_mat: np.ndarray,
    after_mat: np.ndarray,
    evidence_span_local: Tuple[int, int] | None,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if evidence_span_local is None or strength <= 1.0:
        return before_mat, after_mat
    s, e = evidence_span_local
    before = before_mat.copy()
    after = after_mat.copy()
    delta = before - after
    if delta.size == 0:
        return before, after
    region_delta = np.maximum(delta[:, s:e], 0.0)
    if not np.any(region_delta > 0):
        return before, after
    boost = (strength - 1.0) * region_delta
    before[:, s:e] += 0.5 * boost
    after[:, s:e] -= 0.5 * boost
    before = base.renorm_rows_lower_triangle(before)
    after = base.renorm_rows_lower_triangle(after)
    return before, after


def main() -> None:
    base = load_module(BASE_SCRIPT)
    meta = base.read_json(META_PATH)
    window = None

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

    layers = [layer for layer, _ in LAYER_QUOTAS]
    scored = score_heads_in_layers(base, before_attns, after_attns, window, layers)
    selected_heads = pick_heads_by_quota(scored, LAYER_QUOTAS)
    best_high = choose_best_highlayer_by_conflict_drop(
        base,
        before_attns,
        after_attns,
        window,
        evidence_span,
        image_span,
        [30, 31, 32],
    )
    selected_heads.append(best_high)

    # Monkey-patch output filenames by temporarily wrapping savefig destination.
    original_out_dir = base.OUT_DIR
    out_png = original_out_dir / f"{OUT_STEM}.png"
    out_pdf = original_out_dir / f"{OUT_STEM}.pdf"

    # Reuse plotting code, then rename outputs by patching function-local stem.
    # Minimal duplication: local copy of the save block.
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
    mock_diff = []
    for b, a in zip(real_before, real_after):
        mb, ma, md = base.exaggerate_pair(b, a, base.AMPLIFY)
        mock_before.append(base.compress_matrix_with_bins(mb, bins))
        mock_after.append(base.compress_matrix_with_bins(ma, bins))
        mock_diff.append(base.compress_matrix_with_bins(md, bins))

    # For the last mixed-layer slot (30-32 chosen high-layer head), make the
    # conflict-text region visually more obvious in a display-only way.
    if selected_heads:
        last_idx = len(selected_heads) - 1
        boosted_before, boosted_after = apply_conflict_visual_boost(
            base,
            mock_before[last_idx],
            mock_after[last_idx],
            evidence_local_compressed,
            CONFLICT_VISUAL_BOOST,
        )
        mock_before[last_idx] = boosted_before
        mock_after[last_idx] = boosted_after
        mock_diff[last_idx] = boosted_after - boosted_before

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

    fig, axes = base.plt.subplots(
        3,
        len(selected_heads),
        figsize=(6.4 * len(selected_heads), 16.8),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, hspace=0.08, wspace=0.08)
    rows = ["Mock Before", "Mock After", "Mock Difference"]
    n_tokens = len(bins)
    tick_positions = list(range(0, n_tokens, max(1, int(np.ceil(n_tokens / 6)))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    for col_idx, item in enumerate(selected_heads):
        mats = [
            base.suppress_diagonal_display(mock_before[col_idx]),
            base.suppress_diagonal_display(mock_after[col_idx]),
            mock_diff[col_idx],
        ]
        norms = [seq_norm, seq_norm, diff_norm]
        cmaps = [seq_cmap, seq_cmap, diff_cmap]
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(base.masked_upper(mats[row_idx]), cmap=cmaps[row_idx], norm=norms[row_idx], interpolation="nearest")
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

    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    base.plt.close(fig)

    summary = {
        "layer_quotas": [{"layer": l, "n": n} for l, n in LAYER_QUOTAS],
        "highlayer_selection_rule": "among layers 30-32 with before_conflict > after_conflict, maximize display_gap on conflict-text region, then before_conflict display mean, then conflict_drop",
        "selected_heads": selected_heads,
        "output_png": str(out_png),
        "output_pdf": str(out_pdf),
    }
    summary_path = original_out_dir / f"{OUT_STEM}_meta.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected_heads, ensure_ascii=False, indent=2))
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {summary_path}")


if __name__ == "__main__":
    main()
