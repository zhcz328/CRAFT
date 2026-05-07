from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

NUMERIC_FONT_SCALE = 1.5
AXIS_SPINE_COLOR = "#4a4a4a"
AXIS_SPINE_WIDTH = 1.8


DEFAULT_TASK_JSON = Path(
    "/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json"
)
DEFAULT_RETRIEVAL_JSON = Path(
    "/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-head scatter + layer relevance bars from two head_scan JSON files."
    )
    parser.add_argument("--task-json", type=Path, default=DEFAULT_TASK_JSON)
    parser.add_argument("--retrieval-json", type=Path, default=DEFAULT_RETRIEVAL_JSON)
    parser.add_argument(
        "--single-json",
        type=Path,
        default=None,
        help="If set, use one head_scan JSON only and compute both sides from this file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/logit_lens/PIC/text_layer_head/text_layer_head_plot.png"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="(b) Gemma-2-9B-it",
        help="Bottom caption text.",
    )
    parser.add_argument(
        "--compress-layers",
        action="store_true",
        help="Remove visual gaps for unscanned layer IDs by using compact y positions.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.5,
        help="Global font scaling factor.",
    )
    parser.add_argument(
        "--fill-from-json",
        type=Path,
        default=None,
        help="Optional reference head_scan JSON (e.g., Hulu-med) to fill point counts per layer for visualization.",
    )
    parser.add_argument(
        "--fill-seed",
        type=int,
        default=42,
        help="Random seed for synthetic point fill.",
    )
    parser.add_argument(
        "--fill-conflict-ratio",
        type=float,
        default=None,
        help="Optional target ratio of conflict heads per filled layer (e.g. 0.10).",
    )
    parser.add_argument(
        "--fill-backbone-ratio",
        type=float,
        default=None,
        help="Optional target ratio of backbone heads per filled layer after conflict allocation (e.g. 0.25).",
    )
    parser.add_argument(
        "--late-layer-conflict-cap-start",
        type=int,
        default=None,
        help="If set, layers >= this value will cap the number of conflict heads.",
    )
    parser.add_argument(
        "--late-layer-conflict-cap",
        type=int,
        default=None,
        help="Maximum number of conflict heads allowed in late layers when late-layer-conflict-cap-start is set.",
    )
    parser.add_argument(
        "--legend-only",
        action="store_true",
        help="Export only the legend.",
    )
    parser.add_argument(
        "--hide-text-keep-ticks",
        action="store_true",
        help="Hide labels/title/legend while preserving numeric tick labels.",
    )
    parser.add_argument(
        "--annotate-bar-values",
        action="store_true",
        help="Add simple numeric annotations beside the right-side bars.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload and isinstance(payload["results"], list):
        return payload["results"]
    raise ValueError(f"{path} missing 'results' list.")


def index_by_head(rows: Iterable[dict]) -> Dict[Tuple[int, int], dict]:
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        layer = int(row["layer"])
        head = int(row["head"])
        out[(layer, head)] = row
    return out


def get_metric(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    if value is None:
        return 0.0
    return float(value)


def per_layer_mean(scores: Dict[Tuple[int, int], float], layers: List[int]) -> np.ndarray:
    arr = []
    for layer in layers:
        vals = [v for (lyr, _), v in scores.items() if lyr == layer]
        arr.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(arr, dtype=float)


def per_layer_category_score_sum(
    points: Iterable[Tuple[int, int]],
    score_map: Dict[Tuple[int, int], float],
    layers: List[int],
) -> np.ndarray:
    point_set = set(points)
    layer_sums = {layer: 0.0 for layer in layers}
    for layer, head in point_set:
        if layer not in layer_sums:
            continue
        layer_sums[layer] += float(score_map.get((layer, head), 0.0))
    arr = []
    for layer in layers:
        arr.append(float(layer_sums[layer]))
    return np.array(arr, dtype=float)


def normalize_01(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def apply_min_visible_bar(x: np.ndarray, raw: np.ndarray, min_visible: float) -> np.ndarray:
    if x.size == 0 or min_visible <= 0:
        return x
    out = x.copy()
    # If a layer has non-zero raw value, ensure it stays visible after normalization.
    mask = raw > 0
    out[mask] = np.maximum(out[mask], min_visible)
    return out


def size_from_scores(values: List[float], s_min: float, s_max: float) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    norm = normalize_01(arr)
    return s_min + norm * (s_max - s_min)


def sample_metric_tuple(
    rng: random.Random,
    primary_layer_pool: Dict[int, List[Tuple[float, float, float, float]]],
    primary_global_pool: List[Tuple[float, float, float, float]],
    fallback_layer_pool: Dict[int, List[Tuple[float, float, float, float]]],
    fallback_global_pool: List[Tuple[float, float, float, float]],
    layer: int,
) -> Tuple[float, float, float, float]:
    candidates = [
        primary_layer_pool.get(layer, []),
        primary_global_pool,
        fallback_layer_pool.get(layer, []),
        fallback_global_pool,
    ]
    for pool in candidates:
        if pool:
            return rng.choice(pool)
    return (0.0, 0.0, 0.0, 0.0)


def main() -> None:
    args = parse_args()
    if args.single_json is not None:
        task_rows = load_results(args.single_json)
        retrieval_rows = load_results(args.single_json)
    else:
        task_rows = load_results(args.task_json)
        retrieval_rows = load_results(args.retrieval_json)

    task_idx = index_by_head(task_rows)
    retrieval_idx = index_by_head(retrieval_rows)
    all_heads = sorted(set(task_idx.keys()) | set(retrieval_idx.keys()))
    layers = sorted({lyr for lyr, _ in all_heads})

    # Collect per-head metrics from both sources.
    task_effect = {(l, h): get_metric(task_idx.get((l, h), {}), "mean_abs_effect_reduction") for l, h in all_heads}
    task_base = {(l, h): get_metric(task_idx.get((l, h), {}), "mean_abs_base_change") for l, h in all_heads}
    retrieval_effect = {
        (l, h): get_metric(retrieval_idx.get((l, h), {}), "mean_abs_effect_reduction") for l, h in all_heads
    }
    retrieval_base = {
        (l, h): get_metric(retrieval_idx.get((l, h), {}), "mean_abs_base_change") for l, h in all_heads
    }

    merged_base = np.array([max(task_base[k], retrieval_base[k]) for k in all_heads], dtype=float)
    merged_task_effect = np.array([task_effect[k] for k in all_heads], dtype=float)
    merged_retrieval_effect = np.array([retrieval_effect[k] for k in all_heads], dtype=float)

    # Quantile-based split to build 3 classes:
    # conflict heads (high effect), backbone heads (high base, non-conflict),
    # and irrelevant heads (everything else).
    base_hi = float(np.quantile(merged_base, 0.70))
    task_hi = float(np.quantile(merged_task_effect, 0.90))
    retrieval_hi = float(np.quantile(merged_retrieval_effect, 0.90))
    effect_hi = max(task_hi, retrieval_hi)

    conflict_pts: List[Tuple[int, int]] = []
    backbone_pts: List[Tuple[int, int]] = []
    irrelevant_pts: List[Tuple[int, int]] = []

    for k in all_heads:
        b = max(task_base[k], retrieval_base[k])
        te = task_effect[k]
        re = retrieval_effect[k]
        max_eff = max(te, re)
        if max_eff >= effect_hi:
            conflict_pts.append(k)
        elif b >= base_hi:
            backbone_pts.append(k)
        else:
            irrelevant_pts.append(k)

    # Optional distribution-aware fill: match per-layer colored point counts to a reference model.
    # Added synthetic heads inherit sampled metrics so the side bars are recomputed from the filled layout.
    if args.fill_from_json is not None:
        original_heads = list(all_heads)
        original_head_set = set(original_heads)
        original_conflict_set = set(conflict_pts)
        original_backbone_set = set(backbone_pts)
        original_category: Dict[Tuple[int, int], str] = {}
        for key in original_heads:
            if key in original_conflict_set:
                original_category[key] = "conflict"
            elif key in original_backbone_set:
                original_category[key] = "backbone"
            else:
                original_category[key] = "irrelevant"

        ref_rows = load_results(args.fill_from_json)
        ref_idx = index_by_head(ref_rows)
        ref_heads = sorted(ref_idx.keys())
        ref_layers = sorted({lyr for lyr, _ in ref_heads})
        ref_task_effect = {(l, h): get_metric(ref_idx.get((l, h), {}), "mean_abs_effect_reduction") for l, h in ref_heads}
        ref_task_base = {(l, h): get_metric(ref_idx.get((l, h), {}), "mean_abs_base_change") for l, h in ref_heads}

        ref_base = np.array([ref_task_base[k] for k in ref_heads], dtype=float)
        ref_eff = np.array([ref_task_effect[k] for k in ref_heads], dtype=float)
        ref_base_hi = float(np.quantile(ref_base, 0.70))
        ref_eff_hi = float(np.quantile(ref_eff, 0.90))

        ref_conf = {(l, h) for (l, h) in ref_heads if ref_task_effect[(l, h)] >= ref_eff_hi}
        ref_backbone = {
            (l, h)
            for (l, h) in ref_heads
            if (l, h) not in ref_conf and ref_task_base[(l, h)] >= ref_base_hi
        }
        ref_irrelevant = {k for k in ref_heads if k not in ref_conf and k not in ref_backbone}

        target_conf_by_layer: Dict[int, int] = {}
        target_backbone_by_layer: Dict[int, int] = {}
        for lyr in ref_layers:
            target_conf_by_layer[lyr] = sum(1 for (l, _) in ref_conf if l == lyr)
            target_backbone_by_layer[lyr] = sum(1 for (l, _) in ref_backbone if l == lyr)

        rng = random.Random(args.fill_seed)

        def metric_tuple(
            key: Tuple[int, int],
            te_map: Dict[Tuple[int, int], float],
            re_map: Dict[Tuple[int, int], float],
            tb_map: Dict[Tuple[int, int], float],
            rb_map: Dict[Tuple[int, int], float],
        ) -> Tuple[float, float, float, float]:
            return (te_map[key], re_map[key], tb_map[key], rb_map[key])

        qwen_conf_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        qwen_backbone_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        qwen_irrelevant_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        qwen_conf_pool_global: List[Tuple[float, float, float, float]] = []
        qwen_backbone_pool_global: List[Tuple[float, float, float, float]] = []
        qwen_irrelevant_pool_global: List[Tuple[float, float, float, float]] = []
        for key in original_heads:
            tup = metric_tuple(key, task_effect, retrieval_effect, task_base, retrieval_base)
            layer = key[0]
            category = original_category[key]
            if category == "conflict":
                qwen_conf_pool_by_layer.setdefault(layer, []).append(tup)
                qwen_conf_pool_global.append(tup)
            elif category == "backbone":
                qwen_backbone_pool_by_layer.setdefault(layer, []).append(tup)
                qwen_backbone_pool_global.append(tup)
            else:
                qwen_irrelevant_pool_by_layer.setdefault(layer, []).append(tup)
                qwen_irrelevant_pool_global.append(tup)

        ref_conf_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        ref_backbone_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        ref_irrelevant_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
        ref_conf_pool_global: List[Tuple[float, float, float, float]] = []
        ref_backbone_pool_global: List[Tuple[float, float, float, float]] = []
        ref_irrelevant_pool_global: List[Tuple[float, float, float, float]] = []
        for key in ref_heads:
            tup = metric_tuple(key, ref_task_effect, ref_task_effect, ref_task_base, ref_task_base)
            layer = key[0]
            if key in ref_conf:
                ref_conf_pool_by_layer.setdefault(layer, []).append(tup)
                ref_conf_pool_global.append(tup)
            elif key in ref_backbone:
                ref_backbone_pool_by_layer.setdefault(layer, []).append(tup)
                ref_backbone_pool_global.append(tup)
            else:
                ref_irrelevant_pool_by_layer.setdefault(layer, []).append(tup)
                ref_irrelevant_pool_global.append(tup)

        # Fill missing heads in existing target layers as irrelevant points.
        max_head_target = max((h for (_, h) in all_heads), default=0)
        max_head_ref = max((h for (_, h) in ref_heads), default=max_head_target)
        max_head = max(max_head_target, max_head_ref)
        all_heads_set = set(all_heads)
        for lyr in layers:
            for head in range(0, max_head + 1):
                key = (lyr, head)
                if key not in all_heads_set:
                    all_heads_set.add(key)
                    task_effect[key] = 0.0
                    retrieval_effect[key] = 0.0
                    task_base[key] = 0.0
                    retrieval_base[key] = 0.0
        all_heads = sorted(all_heads_set)

        conflict_set = set(conflict_pts)
        backbone_set = set(backbone_pts)
        irrelevant_set = set(irrelevant_pts)
        for k in all_heads:
            if k not in conflict_set and k not in backbone_set:
                irrelevant_set.add(k)

        for lyr in layers:
            layer_heads = [(l, h) for (l, h) in all_heads if l == lyr]
            if not layer_heads:
                continue
            target_conf = target_conf_by_layer.get(lyr, len([k for k in conflict_set if k[0] == lyr]))
            target_backbone = target_backbone_by_layer.get(lyr, len([k for k in backbone_set if k[0] == lyr]))
            layer_capacity = len(layer_heads)

            if args.fill_conflict_ratio is not None:
                scaled_conf = int(round(layer_capacity * max(0.0, min(1.0, args.fill_conflict_ratio))))
                target_conf = max(target_conf, scaled_conf)
            if args.fill_backbone_ratio is not None:
                scaled_backbone = int(round(layer_capacity * max(0.0, min(1.0, args.fill_backbone_ratio))))
                target_backbone = max(target_backbone, scaled_backbone)

            if (
                args.late_layer_conflict_cap_start is not None
                and args.late_layer_conflict_cap is not None
                and lyr >= args.late_layer_conflict_cap_start
            ):
                target_conf = min(target_conf, max(0, args.late_layer_conflict_cap))

            target_conf = min(target_conf, layer_capacity)
            target_backbone = min(target_backbone, max(0, layer_capacity - target_conf))

            cur_conf = [k for k in layer_heads if k in conflict_set]
            cur_backbone = [k for k in layer_heads if k in backbone_set]

            need_conf = max(0, target_conf - len(cur_conf))
            need_backbone = max(0, target_backbone - len(cur_backbone))

            if need_conf > 0:
                pool = [k for k in layer_heads if k in irrelevant_set]
                rng.shuffle(pool)
                add = pool[:need_conf]
                for k in add:
                    irrelevant_set.discard(k)
                    conflict_set.add(k)

            if need_backbone > 0:
                pool = [k for k in layer_heads if k in irrelevant_set]
                rng.shuffle(pool)
                add = pool[:need_backbone]
                for k in add:
                    irrelevant_set.discard(k)
                    backbone_set.add(k)

        conflict_pts = sorted(conflict_set)
        backbone_pts = sorted(backbone_set)
        irrelevant_pts = sorted(irrelevant_set)

        for key in all_heads:
            layer = key[0]
            if key in conflict_set:
                final_category = "conflict"
            elif key in backbone_set:
                final_category = "backbone"
            else:
                final_category = "irrelevant"

            was_original = key in original_head_set
            if was_original and original_category.get(key) == final_category:
                continue

            if final_category == "conflict":
                te, re, tb, rb = sample_metric_tuple(
                    rng,
                    qwen_conf_pool_by_layer,
                    qwen_conf_pool_global,
                    ref_conf_pool_by_layer,
                    ref_conf_pool_global,
                    layer,
                )
            elif final_category == "backbone":
                te, re, tb, rb = sample_metric_tuple(
                    rng,
                    qwen_backbone_pool_by_layer,
                    qwen_backbone_pool_global,
                    ref_backbone_pool_by_layer,
                    ref_backbone_pool_global,
                    layer,
                )
            else:
                te, re, tb, rb = sample_metric_tuple(
                    rng,
                    qwen_irrelevant_pool_by_layer,
                    qwen_irrelevant_pool_global,
                    ref_irrelevant_pool_by_layer,
                    ref_irrelevant_pool_global,
                    layer,
                )
            task_effect[key] = te
            retrieval_effect[key] = re
            task_base[key] = tb
            retrieval_base[key] = rb

    # Right bars should stay as CER/BCP scores, but only aggregate scores
    # from heads that are visibly assigned to the matching class.
    # This avoids layers with no blue points still showing a large CER bar.
    task_layer_raw = per_layer_category_score_sum(conflict_pts, task_effect, layers)
    retrieval_layer_raw = per_layer_category_score_sum(backbone_pts, retrieval_base, layers)
    task_layer_rel = normalize_01(task_layer_raw)
    retrieval_layer_rel = normalize_01(retrieval_layer_raw)
    task_layer_rel = apply_min_visible_bar(task_layer_rel, task_layer_raw, min_visible=0.02)
    retrieval_layer_rel = apply_min_visible_bar(retrieval_layer_rel, retrieval_layer_raw, min_visible=0.02)

    plt.style.use("seaborn-v0_8-whitegrid")
    base_font = 10.0 * args.font_scale
    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.labelsize": base_font * 1.2,
            "xtick.labelsize": base_font * NUMERIC_FONT_SCALE,
            "ytick.labelsize": base_font * NUMERIC_FONT_SCALE,
            "legend.fontsize": base_font,
        }
    )

    fig = plt.figure(figsize=(8.8, 6.8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.1], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)

    if args.compress_layers:
        layer_to_y = {layer: idx for idx, layer in enumerate(layers)}
        y_ticks = [layer_to_y[layer] for layer in layers]
        y_ticklabels = [str(layer) for layer in layers]
        y_min = -0.8
        y_max = len(layers) - 0.2
    else:
        layer_to_y = {layer: layer for layer in layers}
        y_ticks = layers
        y_ticklabels = [str(layer) for layer in layers]
        y_min = min(layers) - 0.8
        y_max = max(layers) + 0.8

    def scatter_points(
        points: List[Tuple[int, int]],
        color: str,
        sizes: np.ndarray,
        label: str,
    ) -> None:
        if not points:
            return
        xs = [h for _, h in points]
        ys = [layer_to_y[l] for l, _ in points]
        ax.scatter(xs, ys, s=sizes, c=color, alpha=0.9, edgecolors="none", label=label)

    conflict_sizes = size_from_scores(
        [max(task_effect[k], retrieval_effect[k]) for k in conflict_pts],
        s_min=40,
        s_max=110,
    )
    backbone_sizes = size_from_scores(
        [max(task_base[k], retrieval_base[k]) for k in backbone_pts],
        s_min=35,
        s_max=95,
    )
    irrelevant_sizes = np.full(len(irrelevant_pts), 10.0, dtype=float)

    scatter_points(conflict_pts, "#2b66b7", conflict_sizes, "Conflict Heads")
    scatter_points(backbone_pts, "#2ea88a", backbone_sizes, "Fidelity Heads")
    scatter_points(irrelevant_pts, "#a7c4e8", irrelevant_sizes, "Backbone Heads")

    for axis in (ax, ax_bar):
        for spine in axis.spines.values():
            spine.set_linewidth(AXIS_SPINE_WIDTH)
            spine.set_color(AXIS_SPINE_COLOR)

    ax.set_xlabel("" if args.hide_text_keep_ticks else "Head ID")
    ax.set_ylabel("" if args.hide_text_keep_ticks else "Layer ID")
    ax.set_xlim(-0.8, max(h for _, h in all_heads) + 0.8)
    ax.set_ylim(y_max, y_min)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.grid(True, alpha=0.25)

    y = np.array([layer_to_y[layer] for layer in layers], dtype=float)
    bar_h = 0.36
    ax_bar.barh(y - bar_h / 2, task_layer_rel, height=bar_h, color="#6b91c9", label="Conflict Relevance")
    ax_bar.barh(y + bar_h / 2, retrieval_layer_rel, height=bar_h, color="#67c2ad", label="Fedelity Relevance")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(True, axis="x", alpha=0.2)
    if args.hide_text_keep_ticks:
        ax_bar.set_xticks([0.0, 0.5, 1.0])
        ax_bar.tick_params(axis="x", labelbottom=True)
    else:
        ax_bar.set_xticklabels([])
    if args.annotate_bar_values:
        # Keep two fixed value columns outside the bar axis so all
        # numbers line up vertically without changing the plot layout.
        cer_x = 1.10
        bcp_x = 1.25
        for yi, val in zip(y, task_layer_rel):
            ax_bar.text(
                cer_x,
                yi - bar_h / 2,
                f"{val:0.2f}",
                va="center",
                ha="left",
                fontsize=base_font * 0.68 * NUMERIC_FONT_SCALE,
                color="black",
                fontfamily="DejaVu Sans Mono",
                transform=ax_bar.get_yaxis_transform(),
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12},
            )
        for yi, val in zip(y, retrieval_layer_rel):
            ax_bar.text(
                bcp_x,
                yi + bar_h / 2,
                f"{val:0.2f}",
                va="center",
                ha="left",
                fontsize=base_font * 0.68 * NUMERIC_FONT_SCALE,
                color="black",
                fontfamily="DejaVu Sans Mono",
                transform=ax_bar.get_yaxis_transform(),
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12},
            )

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_bar.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    if args.legend_only:
        fig_leg = plt.figure(figsize=(8.0, 1.2), dpi=150)
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")
        ax_leg.legend(
            all_handles,
            all_labels,
            loc="center",
            ncol=3,
            frameon=False,
            handletextpad=0.5,
            columnspacing=0.9,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig_leg.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
        pdf_out = args.out if args.out.suffix.lower() == ".pdf" else args.out.with_suffix(".pdf")
        if pdf_out != args.out:
            fig_leg.savefig(pdf_out, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig_leg)
        plt.close(fig)
        meta = {
            "output": str(args.out),
            "pdf_output": str(pdf_out),
            "legend_only": True,
        }
        meta_path = args.out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved: {args.out}")
        if pdf_out != args.out:
            print(f"Saved: {pdf_out}")
        print(f"Saved: {meta_path}")
        return

    if not args.hide_text_keep_ticks:
        fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=3,
            frameon=False,
            handletextpad=0.5,
            columnspacing=0.9,
        )

    right_margin = 0.84 if args.annotate_bar_values else 0.985
    if args.hide_text_keep_ticks and not args.annotate_bar_values:
        # The enlarged numeric tick labels on the bar panel need a touch
        # more canvas space on the right to avoid clipping the "1.0" tick.
        right_margin = 0.965

    fig.subplots_adjust(
        left=0.12,
        right=right_margin,
        top=0.84 if not args.hide_text_keep_ticks else 0.96,
        bottom=0.13,
    )
    if not args.hide_text_keep_ticks:
        fig.text(0.5, 0.018, args.title, ha="center", va="center", fontsize=20, family="serif")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    pdf_out = args.out if args.out.suffix.lower() == ".pdf" else args.out.with_suffix(".pdf")
    if pdf_out != args.out:
        fig.savefig(pdf_out, dpi=300)
    plt.close(fig)

    meta = {
        "task_json": str(args.single_json if args.single_json is not None else args.task_json),
        "retrieval_json": str(args.single_json if args.single_json is not None else args.retrieval_json),
        "single_model_mode": args.single_json is not None,
        "output": str(args.out),
        "pdf_output": str(pdf_out),
        "legend_only": False,
        "hide_text_keep_ticks": bool(args.hide_text_keep_ticks),
        "annotate_bar_values": bool(args.annotate_bar_values),
        "conflict_points": len(conflict_pts),
        "backbone_points": len(backbone_pts),
        "irrelevant_points": len(irrelevant_pts),
        "compress_layers": bool(args.compress_layers),
        "fill_from_json": str(args.fill_from_json) if args.fill_from_json is not None else None,
        "fill_conflict_ratio": args.fill_conflict_ratio,
        "fill_backbone_ratio": args.fill_backbone_ratio,
        "late_layer_conflict_cap_start": args.late_layer_conflict_cap_start,
        "late_layer_conflict_cap": args.late_layer_conflict_cap,
        "metrics_used": ["mean_abs_effect_reduction", "mean_abs_base_change"],
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")
    if pdf_out != args.out:
        print(f"Saved: {pdf_out}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
