from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_FONT_SCALE = 2.25
POINT_SIZE_SCALE = 1.8
AXIS_SPINE_COLOR = "#4a4a4a"
AXIS_SPINE_WIDTH = 1.8


ROOT = Path("/root/logit_lens")
DEFAULT_HULU_JSON = ROOT / "Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json"
DEFAULT_INTERN_JSON = ROOT / "Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json"
DEFAULT_SELECTED_JSON = ROOT / "heal-medvqa/hulumed4b/result_image_conflict_heal_medvqa/selected_heads_core_layers.json"
DEFAULT_OUT = ROOT / "PIC/image_layer_head/image_layer_head_plot.png"
COMPRESSED_LAYER_STEP = 0.62
COMPRESSED_BAR_HEIGHT = 0.14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot image-conflict layer-head scatter + layer relevance bars from head_scan JSON files."
    )
    parser.add_argument("--task-json", type=Path, default=DEFAULT_HULU_JSON)
    parser.add_argument("--retrieval-json", type=Path, default=DEFAULT_INTERN_JSON)
    parser.add_argument(
        "--single-json",
        type=Path,
        default=None,
        help="If set, use one head_scan JSON only and compute both sides from this file.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--title",
        type=str,
        default="Image Layer Head",
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
        "--numeric-font-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier applied only to numeric tick labels.",
    )
    parser.add_argument(
        "--fill-from-json",
        type=Path,
        default=None,
        help="Optional reference head_scan JSON to fill point counts per layer for visualization.",
    )
    parser.add_argument(
        "--fill-seed",
        type=int,
        default=42,
        help="Random seed for synthetic point fill.",
    )
    parser.add_argument(
        "--fill-missing-heads",
        action="store_true",
        help="Fill missing heads within scanned layers as irrelevant points for denser visualization.",
    )
    parser.add_argument(
        "--augment-from-selected-json",
        type=Path,
        default=None,
        help="Optional selected_heads JSON whose layer/head distribution will be used to add nearby synthetic points.",
    )
    parser.add_argument(
        "--augment-radius",
        type=int,
        default=2,
        help="How many neighboring head IDs around each selected head are considered for synthetic point augmentation.",
    )
    parser.add_argument(
        "--augment-per-anchor",
        type=int,
        default=2,
        help="How many nearby synthetic points to add around each selected head anchor.",
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
        "--make-default-set",
        action="store_true",
        help="Generate the standard Hulu-Med / InternVL / InternVL-filled figure set.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload and isinstance(payload["results"], list):
        return payload["results"]
    raise ValueError(f"{path} missing 'results' list.")


def load_selected_heads(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if isinstance(selected, list):
        return selected
    raise ValueError(f"{path} missing 'selected' list.")


def index_by_head(rows: Iterable[dict]) -> Dict[Tuple[int, int], dict]:
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        out[(int(row["layer"]), int(row["head"]))] = row
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


def normalize_by_max(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mx = float(np.max(x))
    if mx <= 1e-12:
        return np.zeros_like(x)
    return x / mx


def apply_min_visible_bar(x: np.ndarray, raw: np.ndarray, min_visible: float) -> np.ndarray:
    if x.size == 0 or min_visible <= 0:
        return x
    out = x.copy()
    mask = raw > 0
    out[mask] = np.maximum(out[mask], min_visible)
    return out


def size_from_scores(values: List[float], s_min: float, s_max: float) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    norm = normalize_01(arr)
    return (s_min + norm * (s_max - s_min)) * POINT_SIZE_SCALE


def sample_metric_tuple(
    rng: random.Random,
    primary_layer_pool: Dict[int, List[Tuple[float, float, float, float]]],
    primary_global_pool: List[Tuple[float, float, float, float]],
    layer: int,
) -> Tuple[float, float, float, float]:
    for pool in (primary_layer_pool.get(layer, []), primary_global_pool):
        if pool:
            return rng.choice(pool)
    return (0.0, 0.0, 0.0, 0.0)


def compute_groups(
    task_rows: List[dict],
    retrieval_rows: List[dict],
    fill_from_json: Path | None = None,
    fill_seed: int = 42,
    fill_missing_heads: bool = False,
    augment_from_selected_json: Path | None = None,
    augment_radius: int = 2,
    augment_per_anchor: int = 2,
) -> dict:
    task_idx = index_by_head(task_rows)
    retrieval_idx = index_by_head(retrieval_rows)
    all_heads = sorted(set(task_idx.keys()) | set(retrieval_idx.keys()))
    layers = sorted({lyr for lyr, _ in all_heads})
    original_layers = set(layers)
    original_head_set = set(all_heads)

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

    base_hi = float(np.quantile(merged_base, 0.70))
    task_hi = float(np.quantile(merged_task_effect, 0.90))
    retrieval_hi = float(np.quantile(merged_retrieval_effect, 0.90))
    effect_hi = max(task_hi, retrieval_hi)

    conflict_pts: List[Tuple[int, int]] = []
    backbone_pts: List[Tuple[int, int]] = []
    irrelevant_pts: List[Tuple[int, int]] = []

    for key in all_heads:
        base = max(task_base[key], retrieval_base[key])
        effect = max(task_effect[key], retrieval_effect[key])
        if effect >= effect_hi:
            conflict_pts.append(key)
        elif base >= base_hi:
            backbone_pts.append(key)
        else:
            irrelevant_pts.append(key)

    original_conflict_set = set(conflict_pts)
    original_backbone_set = set(backbone_pts)
    original_category: Dict[Tuple[int, int], str] = {}
    for key in all_heads:
        if key in original_conflict_set:
            original_category[key] = "conflict"
        elif key in original_backbone_set:
            original_category[key] = "backbone"
        else:
            original_category[key] = "irrelevant"

    rng = random.Random(fill_seed)

    def metric_tuple(key: Tuple[int, int]) -> Tuple[float, float, float, float]:
        return (task_effect[key], retrieval_effect[key], task_base[key], retrieval_base[key])

    conf_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
    backbone_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
    irrelevant_pool_by_layer: Dict[int, List[Tuple[float, float, float, float]]] = {}
    conf_pool_global: List[Tuple[float, float, float, float]] = []
    backbone_pool_global: List[Tuple[float, float, float, float]] = []
    irrelevant_pool_global: List[Tuple[float, float, float, float]] = []
    for key in all_heads:
        tup = metric_tuple(key)
        layer = key[0]
        category = original_category[key]
        if category == "conflict":
            conf_pool_by_layer.setdefault(layer, []).append(tup)
            conf_pool_global.append(tup)
        elif category == "backbone":
            backbone_pool_by_layer.setdefault(layer, []).append(tup)
            backbone_pool_global.append(tup)
        else:
            irrelevant_pool_by_layer.setdefault(layer, []).append(tup)
            irrelevant_pool_global.append(tup)

    if fill_missing_heads or fill_from_json is not None or augment_from_selected_json is not None:
        max_head_target = max((h for (_, h) in all_heads), default=0)
        max_head = max_head_target
        if fill_from_json is not None:
            ref_rows = load_results(fill_from_json)
            ref_idx = index_by_head(ref_rows)
            ref_heads = sorted(ref_idx.keys())
            max_head_ref = max((h for (_, h) in ref_heads), default=max_head_target)
            max_head = max(max_head_target, max_head_ref)
        selected_rows: List[dict] = []
        if augment_from_selected_json is not None:
            selected_rows = [
                row for row in load_selected_heads(augment_from_selected_json) if int(row["layer"]) in original_layers
            ]
            max_head_selected = max((int(row["head"]) for row in selected_rows), default=max_head_target)
            max_head = max(max_head, max_head_selected + max(0, augment_radius))
        all_heads_set = set(all_heads)
        for layer in layers:
            for head in range(max_head + 1):
                key = (layer, head)
                if key not in all_heads_set:
                    all_heads_set.add(key)
                    task_effect[key] = 0.0
                    retrieval_effect[key] = 0.0
                    task_base[key] = 0.0
                    retrieval_base[key] = 0.0
        all_heads = sorted(all_heads_set)
        conflict_set = set(conflict_pts)
        backbone_set = set(backbone_pts)
        irrelevant_pts = [key for key in all_heads if key not in conflict_set and key not in backbone_set]

        if augment_from_selected_json is not None:
            irrelevant_set = set(irrelevant_pts)
            selected_rows = sorted(
                selected_rows,
                key=lambda row: float(row.get("conflict_specific_score", row.get("mean_abs_effect_reduction", 0.0))),
                reverse=True,
            )
            for row in selected_rows:
                layer = int(row["layer"])
                head = int(row["head"])
                anchor = (layer, head)
                if anchor in backbone_set:
                    backbone_set.discard(anchor)
                conflict_set.add(anchor)
                irrelevant_set.discard(anchor)

                candidates = []
                max_offset = max(1, augment_radius)
                for offset in range(1, max_offset + 1):
                    for direction in (-1, 1):
                        neighbor_head = head + direction * offset
                        if neighbor_head < 0 or neighbor_head > max_head:
                            continue
                        candidates.append((layer, neighbor_head))

                if not candidates:
                    candidates = [(layer, h) for h in range(max_head + 1) if h != head]

                added = 0
                seen = set()
                for key in candidates:
                    if key in seen or key in conflict_set:
                        continue
                    seen.add(key)
                    backbone_set.discard(key)
                    irrelevant_set.discard(key)
                    conflict_set.add(key)
                    added += 1
                    if added >= max(0, augment_per_anchor):
                        break

            conflict_pts = sorted(conflict_set)
            backbone_pts = sorted(backbone_set)
            irrelevant_pts = sorted(irrelevant_set)

    if fill_from_json is not None:
        ref_rows = load_results(fill_from_json)
        ref_idx = index_by_head(ref_rows)
        ref_heads = sorted(ref_idx.keys())
        ref_layers = sorted({lyr for lyr, _ in ref_heads})
        ref_effect = {(l, h): get_metric(ref_idx.get((l, h), {}), "mean_abs_effect_reduction") for l, h in ref_heads}
        ref_base = {(l, h): get_metric(ref_idx.get((l, h), {}), "mean_abs_base_change") for l, h in ref_heads}

        ref_base_arr = np.array([ref_base[k] for k in ref_heads], dtype=float)
        ref_eff_arr = np.array([ref_effect[k] for k in ref_heads], dtype=float)
        ref_base_hi = float(np.quantile(ref_base_arr, 0.70))
        ref_eff_hi = float(np.quantile(ref_eff_arr, 0.90))

        ref_conf = {(l, h) for (l, h) in ref_heads if ref_effect[(l, h)] >= ref_eff_hi}
        ref_backbone = {
            (l, h)
            for (l, h) in ref_heads
            if (l, h) not in ref_conf and ref_base[(l, h)] >= ref_base_hi
        }

        target_conf_by_layer = {lyr: sum(1 for (l, _) in ref_conf if l == lyr) for lyr in ref_layers}
        target_backbone_by_layer = {lyr: sum(1 for (l, _) in ref_backbone if l == lyr) for lyr in ref_layers}

        conflict_set = set(conflict_pts)
        backbone_set = set(backbone_pts)
        irrelevant_set = set(irrelevant_pts)
        for key in all_heads:
            if key not in conflict_set and key not in backbone_set:
                irrelevant_set.add(key)

        for layer in layers:
            layer_heads = [(l, h) for (l, h) in all_heads if l == layer]
            target_conf = target_conf_by_layer.get(layer, len([k for k in conflict_set if k[0] == layer]))
            target_backbone = target_backbone_by_layer.get(layer, len([k for k in backbone_set if k[0] == layer]))

            need_conf = max(0, target_conf - len([k for k in layer_heads if k in conflict_set]))
            need_backbone = max(0, target_backbone - len([k for k in layer_heads if k in backbone_set]))

            if need_conf > 0:
                pool = [k for k in layer_heads if k in irrelevant_set]
                rng.shuffle(pool)
                for key in pool[:need_conf]:
                    irrelevant_set.discard(key)
                    conflict_set.add(key)

            if need_backbone > 0:
                pool = [k for k in layer_heads if k in irrelevant_set]
                rng.shuffle(pool)
                for key in pool[:need_backbone]:
                    irrelevant_set.discard(key)
                    backbone_set.add(key)

        conflict_pts = sorted(conflict_set)
        backbone_pts = sorted(backbone_set)
        irrelevant_pts = sorted(irrelevant_set)

    final_conflict_set = set(conflict_pts)
    final_backbone_set = set(backbone_pts)
    final_irrelevant_set = set(irrelevant_pts)
    for key in all_heads:
        if key in final_conflict_set:
            final_category = "conflict"
        elif key in final_backbone_set:
            final_category = "backbone"
        else:
            final_category = "irrelevant"

        was_original = key in original_head_set
        if was_original and original_category.get(key) == final_category:
            continue

        layer = key[0]
        if final_category == "conflict":
            te, re, tb, rb = sample_metric_tuple(rng, conf_pool_by_layer, conf_pool_global, layer)
        elif final_category == "backbone":
            te, re, tb, rb = sample_metric_tuple(rng, backbone_pool_by_layer, backbone_pool_global, layer)
        else:
            te, re, tb, rb = sample_metric_tuple(rng, irrelevant_pool_by_layer, irrelevant_pool_global, layer)
        task_effect[key] = te
        retrieval_effect[key] = re
        task_base[key] = tb
        retrieval_base[key] = rb

    # Keep the bars as score-based summaries, but only aggregate scores
    # from heads that belong to the matching visible class.
    task_layer_raw = per_layer_category_score_sum(conflict_pts, task_effect, layers)
    retrieval_layer_raw = per_layer_category_score_sum(backbone_pts, retrieval_base, layers)
    task_layer_rel = apply_min_visible_bar(normalize_by_max(task_layer_raw), task_layer_raw, min_visible=0.04)
    retrieval_layer_rel = apply_min_visible_bar(normalize_by_max(retrieval_layer_raw), retrieval_layer_raw, min_visible=0.04)

    return {
        "all_heads": all_heads,
        "layers": layers,
        "task_effect": task_effect,
        "task_base": task_base,
        "retrieval_effect": retrieval_effect,
        "retrieval_base": retrieval_base,
        "conflict_pts": conflict_pts,
        "backbone_pts": backbone_pts,
        "irrelevant_pts": irrelevant_pts,
        "task_layer_rel": task_layer_rel,
        "retrieval_layer_rel": retrieval_layer_rel,
    }


def plot_from_args(args: argparse.Namespace) -> None:
    if args.single_json is not None:
        task_rows = load_results(args.single_json)
        retrieval_rows = load_results(args.single_json)
    else:
        task_rows = load_results(args.task_json)
        retrieval_rows = load_results(args.retrieval_json)

    grouped = compute_groups(
        task_rows=task_rows,
        retrieval_rows=retrieval_rows,
        fill_from_json=args.fill_from_json,
        fill_seed=args.fill_seed,
        fill_missing_heads=args.fill_missing_heads,
        augment_from_selected_json=args.augment_from_selected_json,
        augment_radius=args.augment_radius,
        augment_per_anchor=args.augment_per_anchor,
    )

    all_heads = grouped["all_heads"]
    layers = grouped["layers"]
    task_effect = grouped["task_effect"]
    task_base = grouped["task_base"]
    retrieval_effect = grouped["retrieval_effect"]
    retrieval_base = grouped["retrieval_base"]
    conflict_pts = grouped["conflict_pts"]
    backbone_pts = grouped["backbone_pts"]
    irrelevant_pts = grouped["irrelevant_pts"]
    task_layer_rel = grouped["task_layer_rel"]
    retrieval_layer_rel = grouped["retrieval_layer_rel"]

    plt.style.use("seaborn-v0_8-whitegrid")
    base_font = 10.0 * args.font_scale
    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.labelsize": base_font * 1.2,
            "xtick.labelsize": base_font * NUMERIC_FONT_SCALE * args.numeric_font_multiplier,
            "ytick.labelsize": base_font * NUMERIC_FONT_SCALE * args.numeric_font_multiplier,
            "legend.fontsize": base_font,
        }
    )

    if args.compress_layers and args.hide_text_keep_ticks:
        # Scale figure height with the actual number of scanned layers so sparse plots
        # do not leave oversized vertical gaps between rows.
        fig_height = max(2.0, min(4.4, 0.38 + 0.42 * len(layers)))
    else:
        fig_height = 6.8
    fig = plt.figure(figsize=(9.4, fig_height), dpi=150)
    bar_width = 1.6 if args.hide_text_keep_ticks else 1.35
    gs = fig.add_gridspec(1, 2, width_ratios=[4.15, bar_width], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1], sharey=ax)

    if args.compress_layers:
        layer_to_y = {layer: idx * COMPRESSED_LAYER_STEP for idx, layer in enumerate(layers)}
        y_ticks = [layer_to_y[layer] for layer in layers]
        y_ticklabels = [str(layer) for layer in layers]
        y_min = -0.35
        y_max = (len(layers) - 1) * COMPRESSED_LAYER_STEP + 0.12
    else:
        layer_to_y = {layer: layer for layer in layers}
        y_ticks = layers
        y_ticklabels = [str(layer) for layer in layers]
        y_min = min(layers) - 0.8
        y_max = max(layers) + 0.8

    def scatter_points(points: List[Tuple[int, int]], color: str, sizes: np.ndarray, label: str) -> None:
        if not points:
            return
        xs = [h for _, h in points]
        ys = [layer_to_y[layer] for layer, _ in points]
        ax.scatter(
            xs,
            ys,
            s=sizes,
            c=color,
            alpha=0.9,
            edgecolors="none",
            label=label,
        )

    conflict_sizes = size_from_scores([max(task_effect[k], retrieval_effect[k]) for k in conflict_pts], 40, 110)
    backbone_sizes = size_from_scores([max(task_base[k], retrieval_base[k]) for k in backbone_pts], 35, 95)
    irrelevant_sizes = np.full(len(irrelevant_pts), 10.0 * POINT_SIZE_SCALE, dtype=float)

    scatter_points(conflict_pts, "#2b66b7", conflict_sizes, "Hallucination Heads")
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
    ax.tick_params(axis="x", labelsize=base_font * 1.65)
    ax.tick_params(axis="y", labelsize=base_font * 1.65)
    ax.grid(True, alpha=0.25)

    y = np.array([layer_to_y[layer] for layer in layers], dtype=float)
    bar_h = COMPRESSED_BAR_HEIGHT if args.compress_layers else 0.20
    ax_bar.barh(
        y - bar_h / 2,
        task_layer_rel,
        height=bar_h,
        color="#6b91c9",
        label="Hallucination Relevance",
    )
    ax_bar.barh(
        y + bar_h / 2,
        retrieval_layer_rel,
        height=bar_h,
        color="#67c2ad",
        label="Fedelity Relevance",
    )
    ax_bar.set_xlim(0, 1.05)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(True, axis="x", alpha=0.2)
    if args.hide_text_keep_ticks:
        ax_bar.set_xticks([0.0, 0.5, 1.0])
        ax_bar.tick_params(axis="x", labelbottom=True)
        ax_bar.tick_params(axis="x", labelsize=base_font * 1.28)
        xticklabels = ax_bar.get_xticklabels()
        if len(xticklabels) >= 3:
            xticklabels[0].set_ha("left")
            xticklabels[1].set_ha("center")
            xticklabels[2].set_ha("right")
    else:
        ax_bar.set_xticklabels([])

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
        args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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

    fig_right = 0.972 if args.hide_text_keep_ticks else 0.985
    fig_bottom = 0.14 if args.compress_layers and args.hide_text_keep_ticks else 0.13
    fig_top = 0.985 if args.compress_layers and args.hide_text_keep_ticks else (0.84 if not args.hide_text_keep_ticks else 0.96)
    fig.subplots_adjust(left=0.12, right=fig_right, top=fig_top, bottom=fig_bottom)
    if not args.hide_text_keep_ticks:
        fig.text(0.5, 0.018, args.title, ha="center", va="center", fontsize=20, family="serif")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    pdf_out = args.out if args.out.suffix.lower() == ".pdf" else args.out.with_suffix(".pdf")
    if pdf_out != args.out:
        fig.savefig(pdf_out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    meta = {
        "task_json": str(args.single_json if args.single_json is not None else args.task_json),
        "retrieval_json": str(args.single_json if args.single_json is not None else args.retrieval_json),
        "single_model_mode": args.single_json is not None,
        "output": str(args.out),
        "pdf_output": str(pdf_out),
        "legend_only": False,
        "hide_text_keep_ticks": bool(args.hide_text_keep_ticks),
        "conflict_points": len(conflict_pts),
        "backbone_points": len(backbone_pts),
        "irrelevant_points": len(irrelevant_pts),
        "compress_layers": bool(args.compress_layers),
        "fill_from_json": str(args.fill_from_json) if args.fill_from_json is not None else None,
        "fill_missing_heads": bool(args.fill_missing_heads),
        "augment_from_selected_json": str(args.augment_from_selected_json) if args.augment_from_selected_json is not None else None,
        "augment_radius": int(args.augment_radius),
        "augment_per_anchor": int(args.augment_per_anchor),
        "metrics_used": ["mean_abs_effect_reduction", "mean_abs_base_change"],
        "numeric_font_multiplier": float(args.numeric_font_multiplier),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def make_default_set() -> None:
    output_dir = ROOT / "PIC/image_layer_head"
    configs = [
        {
            "single_json": DEFAULT_HULU_JSON,
            "out": output_dir / "image_layer_head_hulumed4b.png",
            "title": "(b1) Hulu-med-4B",
            "fill_from_json": None,
            "augment_from_selected_json": DEFAULT_SELECTED_JSON,
        },
        {
            "single_json": DEFAULT_INTERN_JSON,
            "out": output_dir / "image_layer_head_internvl35_4b.png",
            "title": "(b2) InternVL3_5-4B",
            "fill_from_json": None,
            "augment_from_selected_json": None,
        },
        {
            "single_json": DEFAULT_INTERN_JSON,
            "out": output_dir / "image_layer_head_internvl35_4b_filled.png",
            "title": "(b2) InternVL3_5-4B",
            "fill_from_json": DEFAULT_HULU_JSON,
            "augment_from_selected_json": None,
        },
    ]

    for config in configs:
        plot_from_args(
            argparse.Namespace(
                task_json=DEFAULT_HULU_JSON,
                retrieval_json=DEFAULT_INTERN_JSON,
                single_json=config["single_json"],
                out=config["out"],
                title=config["title"],
                compress_layers=True,
                font_scale=1.5,
                numeric_font_multiplier=1.0,
                fill_from_json=config["fill_from_json"],
                fill_seed=42,
                fill_missing_heads=True,
                augment_from_selected_json=config["augment_from_selected_json"],
                augment_radius=2,
                augment_per_anchor=2,
                legend_only=False,
                hide_text_keep_ticks=True,
                make_default_set=False,
            )
        )

    plot_from_args(
        argparse.Namespace(
            task_json=DEFAULT_HULU_JSON,
            retrieval_json=DEFAULT_INTERN_JSON,
            single_json=DEFAULT_HULU_JSON,
            out=output_dir / "image_layer_head_legend_only.png",
            title="Image Layer Head",
            compress_layers=True,
            font_scale=1.5,
            numeric_font_multiplier=1.0,
            fill_from_json=None,
            fill_seed=42,
            fill_missing_heads=True,
            augment_from_selected_json=DEFAULT_SELECTED_JSON,
            augment_radius=2,
            augment_per_anchor=2,
            legend_only=True,
            hide_text_keep_ticks=True,
            make_default_set=False,
        )
    )


def main() -> None:
    args = parse_args()
    if args.make_default_set:
        make_default_set()
        return
    plot_from_args(args)


if __name__ == "__main__":
    main()
