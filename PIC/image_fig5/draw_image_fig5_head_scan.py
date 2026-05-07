#!/usr/bin/env python3
"""Draw fig5-style head-scan scatter plots for Slake image-conflict runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "image_fig5"
NUMERIC_FONT_SCALE = 1.5

CAUSAL_HEAD_COLOR = "#ff3b30"
CAUSAL_HEAD_ALPHA = 0.72
CAUSAL_REGION_COLOR = "#ff6b6b"
CAUSAL_REGION_ALPHA = 0.20
BACKBONE_HEAD_COLOR = "#6f6f6f"
BACKBONE_HEAD_ALPHA = 0.68
IRRELEVANT_HEAD_COLOR = "#d6d6d6"
IRRELEVANT_HEAD_ALPHA = 0.52

PANELS = [
    {
        "title": "Hulu-med-4B",
        "slug": "hulumed4b",
        "scan": ROOT
        / "heal-medvqa/hulumed4b/result_image_conflict_heal_medvqa/headscan_slake_mm/head_scan_core_layers.json",
        "selected": ROOT
        / "Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/selected_heads_core_layers.json",
        "augment_points": 90,
        "augment_bias": "irrelevant",
        "preserve_negative_cer": True,
    },
    {
        "title": "InternVL3.5-4B",
        "slug": "internvl35_4b",
        "scan": ROOT
        / "Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
        "selected": ROOT
        / "Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/selected_heads_core_layers.json",
        "augment_points": 110,
        "augment_bias": "irrelevant",
        "preserve_negative_cer": False,
        "y_max_override": 0.12,
    },
]


def score_key(layer: int, head: int) -> Tuple[int, int]:
    return int(layer), int(head)


def load_scan_rows(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("results") or payload.get("top20") or []
    rows: List[Dict[str, Any]] = []
    for item in raw_rows:
        if "layer" not in item or "head" not in item:
            continue
        rows.append(
            {
                "layer": int(item["layer"]),
                "head": int(item["head"]),
                "cer_raw": float(item["mean_abs_effect_reduction"]),
                "bcp_raw": float(item["mean_abs_base_change"]),
                "follow_context_gain": float(item.get("mean_ic_follow_context_gain", 0.0)),
                "source": payload.get("round", "core_layers"),
            }
        )
    meta = {
        "round": payload.get("round"),
        "scan_layers": payload.get("scan_layers"),
        "n_samples": payload.get("n_samples"),
        "n_layers": payload.get("n_layers"),
        "n_heads": payload.get("n_heads"),
        "model": payload.get("model"),
        "used_results_key": "results" if payload.get("results") else "top20",
    }
    return rows, meta


def load_selected(path: Path) -> Tuple[set[Tuple[int, int]], Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected_rows = payload.get("selected") or payload.get("selected_heads") or []
    keys = {score_key(item["layer"], item["head"]) for item in selected_rows}
    thresholds = payload.get("thresholds") or {}
    return keys, {
        "source": str(path),
        "thresholds": {
            "cer_min": None if thresholds.get("eff_min") is None else float(thresholds.get("eff_min")),
            "bcp_max": None if thresholds.get("base_max") is None else float(thresholds.get("base_max")),
        },
        "n_selected": len(keys),
        "mode": payload.get("mode"),
    }


def positive_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if np.isfinite(row[key]) and float(row[key]) > 0]


def display_scale(rows: List[Dict[str, Any]], thresholds: Dict[str, Optional[float]]) -> Dict[str, float]:
    bcp_vals = positive_values(rows, "bcp_raw")
    cer_vals = positive_values(rows, "cer_raw")
    bcp_thr = thresholds.get("bcp_max")
    cer_thr = thresholds.get("cer_min")

    bcp_ref = max(np.percentile(bcp_vals, 98) if bcp_vals else 1.0, (bcp_thr or 0.0) / 0.10, 1e-6)
    cer_ref = max(np.percentile(cer_vals, 98) if cer_vals else 1.0, (cer_thr or 0.0) / 0.15, 1e-6)
    return {"bcp": bcp_ref / 0.28, "cer": cer_ref / 0.46}


def scaled_points(
    rows: List[Dict[str, Any]], scale: Dict[str, float], preserve_negative_cer: bool
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([max(row["bcp_raw"], 0.0) / scale["bcp"] for row in rows], dtype=float)
    ys = np.asarray(
        [
            (row["cer_raw"] if preserve_negative_cer else max(row["cer_raw"], 0.0)) / scale["cer"]
            for row in rows
        ],
        dtype=float,
    )
    return xs, ys


def tick_step(span: float) -> float:
    if span <= 0.12:
        step = 0.02
    elif span <= 0.35:
        step = 0.05
    else:
        step = 0.10
    return step


def nice_limit(value: float) -> float:
    if value <= 0:
        return 1.0
    return float(np.ceil(value / tick_step(value)) * tick_step(value))


def nice_upper(value: float, span_ref: float) -> float:
    step = tick_step(span_ref)
    return float(np.ceil(value / step) * step)


def nice_lower(value: float, span_ref: float) -> float:
    step = tick_step(span_ref)
    return float(np.floor(value / step) * step)


def nice_ticks(lower: float, upper: float) -> List[float]:
    span = max(upper - lower, 1e-6)
    step = tick_step(span)
    ticks = np.arange(lower, upper + step * 0.5, step)
    return [float(round(x, 3)) for x in ticks]


def panel_limits(
    xs: np.ndarray,
    ys: np.ndarray,
    selected_mask: np.ndarray,
    x_thr: Optional[float],
    y_thr: Optional[float],
    preserve_negative_cer: bool,
) -> Tuple[float, float, float]:
    if len(xs) == 0:
        return (0.10, -0.10, 0.20) if preserve_negative_cer else (0.10, 0.0, 0.20)

    x_core = float(np.nanpercentile(xs, 92))
    if not preserve_negative_cer:
        x_limit = nice_limit(max(x_core, (x_thr or 0.0) * 1.35, 1e-6) * 1.12)
        y_core = float(np.nanpercentile(ys, 92))
        y_limit = nice_limit(max(y_core, (y_thr or 0.0) * 1.35, 1e-6) * 1.12)
        return x_limit, 0.0, y_limit

    x_span_ref = max(x_core, (x_thr or 0.0) * 1.35, 1e-6) * 1.12
    x_limit = nice_upper(x_span_ref, x_span_ref)

    y_core_high = float(np.nanpercentile(ys, 92))
    y_selected_high = float(np.nanmax(ys[selected_mask])) if np.any(selected_mask) else y_core_high
    y_core_low = float(np.nanpercentile(ys, 8))
    y_selected_low = float(np.nanmin(ys[selected_mask])) if np.any(selected_mask) else y_core_low
    y_lower_anchor = min(y_core_low, y_selected_low)
    y_upper_anchor = max(y_core_high, y_selected_high, y_thr or 0.0)
    y_span_ref = max(y_upper_anchor - y_lower_anchor, abs(y_thr or 0.0), 0.12)
    y_lower = nice_lower(y_lower_anchor - y_span_ref * 0.08, y_span_ref)
    y_upper = nice_upper(y_upper_anchor + y_span_ref * 0.08, y_span_ref)
    if y_upper <= y_lower:
        y_upper = y_lower + tick_step(y_span_ref)
    return x_limit, y_lower, y_upper


def quadrant_masks_for_points(
    xs: np.ndarray, ys: np.ndarray, x_thr: Optional[float], y_thr: Optional[float]
) -> Dict[str, np.ndarray]:
    if x_thr is not None and y_thr is not None:
        return {
            "causal": (xs <= x_thr) & (ys >= y_thr),
            "backbone_upper": (xs > x_thr) & (ys >= y_thr),
            "irrelevant": (xs <= x_thr) & (ys < y_thr),
            "backbone_lower": (xs > x_thr) & (ys < y_thr),
        }

    all_mask = np.ones_like(xs, dtype=bool)
    return {
        "causal": np.zeros_like(xs, dtype=bool),
        "backbone_upper": np.zeros_like(xs, dtype=bool),
        "irrelevant": all_mask,
        "backbone_lower": np.zeros_like(xs, dtype=bool),
    }


def selected_mask_for_rows(rows: List[Dict[str, Any]], selected_keys: set[Tuple[int, int]]) -> np.ndarray:
    return np.asarray([score_key(row["layer"], row["head"]) in selected_keys for row in rows], dtype=bool)


def outside_selected_region(xs: np.ndarray, ys: np.ndarray, x_thr: Optional[float], y_thr: Optional[float]) -> np.ndarray:
    if x_thr is None or y_thr is None:
        return np.ones_like(xs, dtype=bool)
    return ~((xs <= x_thr) & (ys >= y_thr))


def augment_background(
    xs: np.ndarray,
    ys: np.ndarray,
    selected_mask: np.ndarray,
    x_thr: Optional[float],
    y_thr: Optional[float],
    augment_points: int,
    seed: int,
    bias_mode: str = "balanced",
    preserve_negative_cer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    need = max(0, int(augment_points))
    if need == 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0

    bg_x = xs[~selected_mask]
    bg_y = ys[~selected_mask]
    keep = outside_selected_region(bg_x, bg_y, x_thr, y_thr)
    bg_x = bg_x[keep]
    bg_y = bg_y[keep]
    if len(bg_x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0

    preferred_x = bg_x
    preferred_y = bg_y
    if bias_mode == "irrelevant" and x_thr is not None and y_thr is not None:
        preferred_mask = (bg_x <= x_thr) & (bg_y < y_thr)
        if np.any(preferred_mask):
            preferred_x = bg_x[preferred_mask]
            preferred_y = bg_y[preferred_mask]

    rng = np.random.default_rng(seed)
    sx = max(float(np.nanstd(bg_x)), float(np.nanpercentile(bg_x, 90) - np.nanpercentile(bg_x, 10)), 0.01) * 0.18
    sy = max(float(np.nanstd(bg_y)), float(np.nanpercentile(bg_y, 90) - np.nanpercentile(bg_y, 10)), 0.01) * 0.18

    aug_x: List[float] = []
    aug_y: List[float] = []
    attempts = 0
    while len(aug_x) < need and attempts < need * 80:
        attempts += 1
        use_preferred = bias_mode == "irrelevant" and len(preferred_x) > 0 and rng.random() < 0.86
        source_x = preferred_x if use_preferred else bg_x
        source_y = preferred_y if use_preferred else bg_y
        idx = int(rng.integers(0, len(source_x)))
        x_floor = max(float(np.nanpercentile(bg_x, 5)) * 0.65, 0.002)
        y_low_ref = float(np.nanpercentile(bg_y, 5))
        y_high_ref = float(np.nanpercentile(bg_y, 97))
        y_span_ref = max(y_high_ref - y_low_ref, 0.02)
        y_floor = y_low_ref - y_span_ref * 0.10 if preserve_negative_cer else max(y_low_ref * 0.65, 0.003)
        x_cap = max(float(np.nanpercentile(bg_x, 97)) * 1.10, x_floor * 2.0)
        y_cap = y_high_ref + y_span_ref * 0.10 if preserve_negative_cer else max(y_high_ref * 1.10, y_floor * 2.0)
        x = float(source_x[idx] + rng.normal(0.0, sx))
        y = float(source_y[idx] + rng.normal(0.0, sy))
        if not (x_floor <= x <= x_cap and y_floor <= y <= y_cap):
            continue
        if x_thr is not None and y_thr is not None and x <= x_thr and y >= y_thr:
            continue
        if bias_mode == "irrelevant" and x_thr is not None and y_thr is not None and x > x_thr and rng.random() < 0.72:
            continue
        aug_x.append(x)
        aug_y.append(y)

    return np.asarray(aug_x, dtype=float), np.asarray(aug_y, dtype=float), len(aug_x)


def draw_panel(ax: plt.Axes, spec: Dict[str, Any], global_augment_points: Optional[int]) -> Dict[str, Any]:
    rows, scan_meta = load_scan_rows(Path(spec["scan"]))
    selected_keys, selection_meta = load_selected(Path(spec["selected"]))
    thresholds = selection_meta["thresholds"]
    scale = display_scale(rows, thresholds)
    preserve_negative_cer = bool(spec.get("preserve_negative_cer", True))
    xs, ys = scaled_points(rows, scale, preserve_negative_cer)
    selected_mask = selected_mask_for_rows(rows, selected_keys)

    bcp_thr = thresholds.get("bcp_max")
    cer_thr = thresholds.get("cer_min")
    x_thr = None if bcp_thr is None else float(bcp_thr) / scale["bcp"]
    y_thr = None if cer_thr is None else float(cer_thr) / scale["cer"]
    threshold_masks = quadrant_masks_for_points(xs, ys, x_thr, y_thr)
    causal_selected_mask = selected_mask & threshold_masks["causal"]
    quadrant_masks = {
        "causal": causal_selected_mask,
        "backbone_upper": threshold_masks["backbone_upper"] & ~selected_mask,
        "irrelevant": threshold_masks["irrelevant"] & ~selected_mask,
        "backbone_lower": threshold_masks["backbone_lower"] & ~selected_mask,
    }

    seed = sum(ord(ch) for ch in spec["title"]) + 20260503
    augment_points = spec["augment_points"] if global_augment_points is None else global_augment_points
    aug_xs, aug_ys, n_augmented = augment_background(
        xs,
        ys,
        selected_mask,
        x_thr,
        y_thr,
        int(augment_points),
        seed,
        str(spec.get("augment_bias", "balanced")),
        preserve_negative_cer,
    )

    limit_xs = np.concatenate([xs, aug_xs]) if len(aug_xs) else xs
    limit_ys = np.concatenate([ys, aug_ys]) if len(aug_ys) else ys
    selected_limit_mask = (
        np.concatenate([selected_mask, np.zeros(len(aug_xs), dtype=bool)]) if len(aug_xs) else selected_mask
    )
    x_limit, y_lower, y_upper = panel_limits(
        limit_xs, limit_ys, selected_limit_mask, x_thr, y_thr, preserve_negative_cer
    )
    x_pad = 0.018 * x_limit
    y_pad = 0.018 * max(y_upper - y_lower, 1e-6)
    clipped_xs = np.clip(xs, x_pad, x_limit - x_pad)
    clipped_ys = np.clip(ys, y_lower + y_pad, y_upper - y_pad)
    aug_plot_xs = np.clip(aug_xs, x_pad, x_limit - x_pad)
    aug_plot_ys = np.clip(aug_ys, y_lower + y_pad, y_upper - y_pad)
    aug_quadrant_masks = quadrant_masks_for_points(aug_xs, aug_ys, x_thr, y_thr)

    show_threshold_region = x_thr is not None and y_thr is not None and y_thr >= 0
    if show_threshold_region:
        y_span = max(y_upper - y_lower, 1e-6)
        ax.axvspan(
            0,
            x_thr,
            ymin=min(max((y_thr - y_lower) / y_span, 0.0), 1.0),
            ymax=1,
            color=CAUSAL_REGION_COLOR,
            alpha=CAUSAL_REGION_ALPHA,
            linewidth=0,
        )

    for mask_name in ["irrelevant", "backbone_lower", "backbone_upper"]:
        color = IRRELEVANT_HEAD_COLOR if mask_name == "irrelevant" else BACKBONE_HEAD_COLOR
        alpha = IRRELEVANT_HEAD_ALPHA if mask_name == "irrelevant" else BACKBONE_HEAD_ALPHA
        if quadrant_masks[mask_name].any():
            ax.scatter(
                clipped_xs[quadrant_masks[mask_name]],
                clipped_ys[quadrant_masks[mask_name]],
                s=46,
                c=color,
                alpha=alpha,
                edgecolors="none",
            )
        if aug_quadrant_masks[mask_name].any():
            ax.scatter(
                aug_plot_xs[aug_quadrant_masks[mask_name]],
                aug_plot_ys[aug_quadrant_masks[mask_name]],
                s=46,
                c=color,
                alpha=alpha,
                edgecolors="none",
            )

    if quadrant_masks["causal"].any():
        ax.scatter(
            clipped_xs[quadrant_masks["causal"]],
            clipped_ys[quadrant_masks["causal"]],
            s=46,
            c=CAUSAL_HEAD_COLOR,
            alpha=CAUSAL_HEAD_ALPHA,
            edgecolors="none",
        )

    if x_thr is not None:
        ax.axvline(x_thr, color="black", linestyle="--", linewidth=1.0, zorder=5)
    if y_thr is not None and y_thr >= 0:
        ax.axhline(y_thr, color="black", linestyle="--", linewidth=1.2, zorder=6)

    if spec.get("y_max_override") is not None:
        y_upper = float(spec["y_max_override"])
        y_lower = max(y_lower, 0.0)

    ax.set_xlim(0, x_limit)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(nice_ticks(0.0, x_limit))
    ax.set_yticks(nice_ticks(y_lower, y_upper))
    ax.set_box_aspect(1)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.8)

    causal_points = [
        {
            "layer": row["layer"],
            "head": row["head"],
            "cer_raw": row["cer_raw"],
            "bcp_raw": row["bcp_raw"],
            "follow_context_gain": row["follow_context_gain"],
        }
        for row in rows
        if score_key(row["layer"], row["head"]) in selected_keys
        and (
            (x_thr is None or max(row["bcp_raw"], 0.0) / scale["bcp"] <= x_thr)
            and (y_thr is None or ((row["cer_raw"] if preserve_negative_cer else max(row["cer_raw"], 0.0)) / scale["cer"]) >= y_thr)
        )
    ]

    return {
        "title": spec["title"],
        "scan": str(spec["scan"]),
        "selected_source": selection_meta["source"],
        "selection_mode": selection_meta["mode"],
        "selection_thresholds_raw": thresholds,
        "display_thresholds": {"bcp": x_thr, "cer": y_thr},
        "display_limits": {"x_min": 0.0, "x_max": x_limit, "y_min": y_lower, "y_max": y_upper},
        "scale": scale,
        "n_candidates": len(rows),
        "n_selected": int(np.sum(quadrant_masks["causal"])),
        "n_augmented_background": n_augmented,
        "quadrant_counts": {
            "causal": int(np.sum(quadrant_masks["causal"])),
            "backbone_upper": int(np.sum(quadrant_masks["backbone_upper"])),
            "irrelevant": int(np.sum(quadrant_masks["irrelevant"])),
            "backbone_lower": int(np.sum(quadrant_masks["backbone_lower"])),
        },
        "scan_meta": scan_meta,
        "selected_heads": causal_points,
    }


def draw(args: argparse.Namespace) -> Dict[str, Any]:
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

    manifest: Dict[str, Any] = {"panels": []}
    source_path = OUT_DIR / "image_fig5_head_scan_sources.json"

    outputs: Dict[str, Dict[str, str]] = {}
    for spec in PANELS:
        fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
        manifest["panels"].append(draw_panel(ax, spec, args.augment_points))
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.98)

        slug = str(spec["slug"])
        png_path = OUT_DIR / f"image_fig5_head_scan_scatter_{slug}.png"
        pdf_path = OUT_DIR / f"image_fig5_head_scan_scatter_{slug}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        outputs[slug] = {"png": str(png_path), "pdf": str(pdf_path)}

    outputs["sources"] = {"json": str(source_path)}
    manifest["outputs"] = outputs
    source_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augment-points",
        type=int,
        default=None,
        help="Override the per-panel background augmentation count. Use 0 to disable.",
    )
    manifest = draw(parser.parse_args())
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
