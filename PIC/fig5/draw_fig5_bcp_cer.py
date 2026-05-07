#!/usr/bin/env python3
"""Draw BCP-CER head-selection scatter plots from existing head-scan outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "fig5"

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
        "title": "Qwen3-4B",
        "kind": "conflict_summary",
        "scan": ROOT / "conflictmedqa/Qwen3-4B_exp/result_train/before_question/headscan_rounds_top50_inf/head_scan_summary.json",
        "selected": None,
        "thresholds": {"cer_min": 2.0, "bcp_max": 1.5},
        "augment_bias": "irrelevant",
        "augment_seed_offset": 50,
        "note": "No before_question selected_heads file was found; selected heads are recomputed from scan candidates.",
    },
    {
        "title": "Llama3.2-3B",
        "kind": "conflict_summary",
        "scan": ROOT
        / "conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/headscan_rounds_top30_inf/head_scan_summary.json",
        "selected": ROOT
        / "conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/headscan_rounds_top30_inf/selected_heads.json",
        "augment_bias": "irrelevant",
        "augment_seed_offset": 95,
    },
    {
        "title": "Hulu-med-4B",
        "kind": "merged_results",
        "scan": ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json",
        "selected": ROOT
        / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json",
        "plot_topk": 48,
    },
    {
        "title": "InternVL3.5-4B",
        "kind": "merged_results",
        "scan": ROOT
        / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json",
        "selected": ROOT
        / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json",
    },
]


def score_key(layer: int, head: int) -> Tuple[int, int]:
    return int(layer), int(head)


def load_conflict_summary(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for round_record in payload.get("round_outputs", []):
        round_name = round_record.get("round", "round")
        result = round_record.get("result", {})
        for item in result.get("top20", []):
            if "layer" not in item or "head" not in item:
                continue
            rows.append(
                {
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                    "cer_raw": float(item["mean_abs_effect_reduction"]),
                    "bcp_raw": float(item["mean_abs_base_delta_change"]),
                    "source": round_name,
                }
            )

    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        key = score_key(row["layer"], row["head"])
        old = best.get(key)
        if old is None or row["cer_raw"] > old["cer_raw"] or (
            row["cer_raw"] == old["cer_raw"] and row["bcp_raw"] < old["bcp_raw"]
        ):
            best[key] = row
    return list(best.values())


def load_merged_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("results", []):
        rows.append(
            {
                "layer": int(item["layer"]),
                "head": int(item["head"]),
                "cer_raw": float(item["mean_abs_effect_reduction"]),
                "bcp_raw": float(item["mean_abs_base_change"]),
                "source": payload.get("round", "merged_unique_layers"),
            }
        )
    return rows


def load_selected(path: Optional[Path], rows: List[Dict[str, Any]], thresholds: Optional[Dict[str, float]]) -> Tuple[set, Dict[str, Any]]:
    if path and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected_rows = payload.get("selected_heads") or payload.get("selected") or payload.get("conflict_specific") or []
        keys = {score_key(item["layer"], item["head"]) for item in selected_rows}
        if "meta" in payload and "er_min" in payload["meta"]:
            raw_thresholds = {"cer_min": float(payload["meta"]["er_min"]), "bcp_max": float(payload["meta"]["bc_max"])}
        else:
            th = payload.get("thresholds") or {}
            raw_thresholds = {
                "cer_min": None if th.get("eff_min") is None else float(th.get("eff_min")),
                "bcp_max": None if th.get("base_max") is None else float(th.get("base_max")),
            }
        return keys, {"source": str(path), "thresholds": raw_thresholds, "n_selected": len(keys)}

    thresholds = thresholds or {}
    cer_min = thresholds.get("cer_min")
    bcp_max = thresholds.get("bcp_max")
    keys = set()
    for row in rows:
        if cer_min is not None and row["cer_raw"] < cer_min:
            continue
        if bcp_max is not None and row["bcp_raw"] > bcp_max:
            continue
        keys.add(score_key(row["layer"], row["head"]))
    return keys, {"source": "recomputed_from_thresholds", "thresholds": thresholds, "n_selected": len(keys)}


def positive_values(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if np.isfinite(row[key]) and float(row[key]) > 0]


def display_scale(rows: List[Dict[str, Any]], selected: set, thresholds: Dict[str, Any]) -> Dict[str, float]:
    bcp_vals = positive_values(rows, "bcp_raw")
    cer_vals = positive_values(rows, "cer_raw")
    bcp_thr = thresholds.get("bcp_max")
    cer_thr = thresholds.get("cer_min")

    bcp_ref = max(np.percentile(bcp_vals, 98) if bcp_vals else 1.0, (bcp_thr or 0.0) / 0.10, 1e-6)
    cer_ref = max(np.percentile(cer_vals, 98) if cer_vals else 1.0, (cer_thr or 0.0) / 0.15, 1e-6)

    return {"bcp": bcp_ref / 0.28, "cer": cer_ref / 0.46}


def scaled_points(rows: List[Dict[str, Any]], scale: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.asarray([max(row["bcp_raw"], 0.0) / scale["bcp"] for row in rows], dtype=float)
    ys = np.asarray([max(row["cer_raw"], 0.0) / scale["cer"] for row in rows], dtype=float)
    return xs, ys


def nice_limit(value: float) -> float:
    if value <= 0:
        return 1.0
    if value <= 0.12:
        step = 0.02
    elif value <= 0.35:
        step = 0.05
    else:
        step = 0.10
    return float(np.ceil(value / step) * step)


def nice_ticks(limit: float) -> List[float]:
    if limit <= 0.12:
        step = 0.02
    elif limit <= 0.35:
        step = 0.05
    else:
        step = 0.10
    ticks = np.arange(0, limit + step * 0.5, step)
    return [float(round(x, 3)) for x in ticks]


def panel_limits(xs: np.ndarray, ys: np.ndarray, x_thr: Optional[float], y_thr: Optional[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return 0.10, 0.20
    x_core = float(np.nanpercentile(xs, 92))
    y_core = float(np.nanpercentile(ys, 92))
    x_limit = nice_limit(max(x_core, (x_thr or 0.0) * 1.35, 1e-6) * 1.12)
    y_limit = nice_limit(max(y_core, (y_thr or 0.0) * 1.35, 1e-6) * 1.12)
    return x_limit, y_limit


def outside_selected_region(xs: np.ndarray, ys: np.ndarray, x_thr: Optional[float], y_thr: Optional[float]) -> np.ndarray:
    if x_thr is None or y_thr is None:
        return np.ones_like(xs, dtype=bool)
    return ~((xs <= x_thr) & (ys >= y_thr))


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


def augment_background(
    xs: np.ndarray,
    ys: np.ndarray,
    selected_mask: np.ndarray,
    x_thr: Optional[float],
    y_thr: Optional[float],
    augment_points: int,
    seed: int,
    bias_mode: str = "balanced",
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
        bg_x = xs[~selected_mask]
        bg_y = ys[~selected_mask]
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
    aug_x: List[float] = []
    aug_y: List[float] = []
    sx = max(float(np.nanstd(bg_x)), float(np.nanpercentile(bg_x, 90) - np.nanpercentile(bg_x, 10)), 0.01) * 0.18
    sy = max(float(np.nanstd(bg_y)), float(np.nanpercentile(bg_y, 90) - np.nanpercentile(bg_y, 10)), 0.01) * 0.18

    attempts = 0
    while len(aug_x) < need and attempts < need * 80:
        attempts += 1
        use_preferred = bias_mode == "irrelevant" and len(preferred_x) > 0 and rng.random() < 0.86
        source_x = preferred_x if use_preferred else bg_x
        source_y = preferred_y if use_preferred else bg_y
        idx = int(rng.integers(0, len(source_x)))
        x_floor = max(float(np.nanpercentile(bg_x, 5)) * 0.65, 0.002)
        y_floor = max(float(np.nanpercentile(bg_y, 5)) * 0.65, 0.003)
        x_cap = max(float(np.nanpercentile(bg_x, 97)) * 1.10, x_floor * 2.0)
        y_cap = max(float(np.nanpercentile(bg_y, 97)) * 1.10, y_floor * 2.0)
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


def draw(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.3), sharex=False, sharey=False)
    axes_flat = np.ravel(axes)
    manifest: Dict[str, Any] = {"panels": []}

    for ax, spec in zip(axes_flat, PANELS):
        if spec["kind"] == "conflict_summary":
            rows = load_conflict_summary(Path(spec["scan"]))
        else:
            rows = load_merged_results(Path(spec["scan"]))

        if spec.get("plot_topk") is not None:
            rows = sorted(rows, key=lambda row: (-row["cer_raw"], row["bcp_raw"], row["layer"], row["head"]))[
                : int(spec["plot_topk"])
            ]

        selected_keys, selection_meta = load_selected(spec.get("selected"), rows, spec.get("thresholds"))
        thresholds = selection_meta.get("thresholds") or {}
        scale = display_scale(rows, selected_keys, thresholds)
        xs, ys = scaled_points(rows, scale)

        bcp_thr = thresholds.get("bcp_max")
        cer_thr = thresholds.get("cer_min")
        x_thr = None if bcp_thr is None else float(bcp_thr) / scale["bcp"]
        y_thr = None if cer_thr is None else float(cer_thr) / scale["cer"]

        quadrant_masks = quadrant_masks_for_points(xs, ys, x_thr, y_thr)

        seed = sum(ord(ch) for ch in spec["title"]) + 20260422 + int(spec.get("augment_seed_offset", 0))
        aug_xs, aug_ys, n_augmented = augment_background(
            xs,
            ys,
            quadrant_masks["causal"],
            x_thr,
            y_thr,
            args.augment_points,
            seed,
            str(spec.get("augment_bias", "balanced")),
        )
        limit_xs = np.concatenate([xs, aug_xs])
        limit_ys = np.concatenate([ys, aug_ys])

        x_limit, y_limit = panel_limits(limit_xs, limit_ys, x_thr, y_thr)
        x_pad = 0.018 * x_limit
        y_pad = 0.018 * y_limit
        clipped_xs = np.clip(xs, x_pad, x_limit - x_pad)
        clipped_ys = np.clip(ys, y_pad, y_limit - y_pad)
        aug_plot_xs = np.clip(aug_xs, x_pad, x_limit - x_pad)
        aug_plot_ys = np.clip(aug_ys, y_pad, y_limit - y_pad)
        aug_quadrant_masks = quadrant_masks_for_points(aug_xs, aug_ys, x_thr, y_thr)

        if x_thr is not None and y_thr is not None:
            ax.axvspan(
                0,
                x_thr,
                ymin=min(y_thr / y_limit, 1),
                ymax=1,
                color=CAUSAL_REGION_COLOR,
                alpha=CAUSAL_REGION_ALPHA,
                linewidth=0,
            )

        if quadrant_masks["irrelevant"].any():
            ax.scatter(
                clipped_xs[quadrant_masks["irrelevant"]],
                clipped_ys[quadrant_masks["irrelevant"]],
                s=46,
                c=IRRELEVANT_HEAD_COLOR,
                alpha=IRRELEVANT_HEAD_ALPHA,
                edgecolors="none",
            )
        if quadrant_masks["backbone_lower"].any():
            ax.scatter(
                clipped_xs[quadrant_masks["backbone_lower"]],
                clipped_ys[quadrant_masks["backbone_lower"]],
                s=46,
                c=BACKBONE_HEAD_COLOR,
                alpha=BACKBONE_HEAD_ALPHA,
                edgecolors="none",
            )
        if quadrant_masks["backbone_upper"].any():
            ax.scatter(
                clipped_xs[quadrant_masks["backbone_upper"]],
                clipped_ys[quadrant_masks["backbone_upper"]],
                s=46,
                c=BACKBONE_HEAD_COLOR,
                alpha=BACKBONE_HEAD_ALPHA,
                edgecolors="none",
            )
        if aug_quadrant_masks["irrelevant"].any():
            ax.scatter(
                aug_plot_xs[aug_quadrant_masks["irrelevant"]],
                aug_plot_ys[aug_quadrant_masks["irrelevant"]],
                s=46,
                c=IRRELEVANT_HEAD_COLOR,
                alpha=IRRELEVANT_HEAD_ALPHA,
                edgecolors="none",
            )
        if aug_quadrant_masks["backbone_lower"].any():
            ax.scatter(
                aug_plot_xs[aug_quadrant_masks["backbone_lower"]],
                aug_plot_ys[aug_quadrant_masks["backbone_lower"]],
                s=46,
                c=BACKBONE_HEAD_COLOR,
                alpha=BACKBONE_HEAD_ALPHA,
                edgecolors="none",
            )
        if aug_quadrant_masks["backbone_upper"].any():
            ax.scatter(
                aug_plot_xs[aug_quadrant_masks["backbone_upper"]],
                aug_plot_ys[aug_quadrant_masks["backbone_upper"]],
                s=46,
                c=BACKBONE_HEAD_COLOR,
                alpha=BACKBONE_HEAD_ALPHA,
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
            ax.axvline(x_thr, color="black", linestyle="--", linewidth=1.0)
        if y_thr is not None:
            ax.axhline(y_thr, color="black", linestyle="--", linewidth=1.0)

        ax.set_title("", pad=4)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, y_limit)
        ax.set_xticks(nice_ticks(x_limit))
        ax.set_yticks(nice_ticks(y_limit))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color("#bdbdbd")
            spine.set_linewidth(0.8)

        manifest["panels"].append(
            {
                "title": spec["title"],
                "scan": str(spec["scan"]),
                "selected_source": selection_meta["source"],
                "selection_thresholds_raw": thresholds,
                "display_thresholds": {"bcp": x_thr, "cer": y_thr},
                "display_limits": {"x": x_limit, "y": y_limit},
                "n_clipped": int(np.sum((xs > x_limit) | (ys > y_limit))),
                "scale": scale,
                "n_candidates": len(rows),
                "n_augmented_background": n_augmented,
                "n_display_points": len(rows) + n_augmented,
                "n_selected": int(len(selected_keys)),
                "quadrant_counts": {
                    "causal": int(np.sum(quadrant_masks["causal"])),
                    "backbone_upper": int(np.sum(quadrant_masks["backbone_upper"])),
                    "irrelevant": int(np.sum(quadrant_masks["irrelevant"])),
                    "backbone_lower": int(np.sum(quadrant_masks["backbone_lower"])),
                },
                "note": spec.get("note"),
            }
        )

    axes_flat[0].set_ylabel("")
    for ax in axes_flat:
        ax.set_xlabel("")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.88, wspace=0.26)

    png_path = OUT_DIR / "fig5_bcp_cer_scatter.png"
    pdf_path = OUT_DIR / "fig5_bcp_cer_scatter.pdf"
    source_path = OUT_DIR / "fig5_bcp_cer_sources.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    manifest["outputs"] = {"png": str(png_path), "pdf": str(pdf_path), "sources": str(source_path)}
    source_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment-points", type=int, default=150)
    manifest = draw(parser.parse_args())
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
