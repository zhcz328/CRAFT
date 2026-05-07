#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA = Path("/root/logit_lens/PIC/tuned_lens/head_distribution/text/data/head_margin_samples.jsonl")
DEFAULT_OUT_DIR = Path("/root/logit_lens/PIC/tuned_lens/head_distribution/text")

COLORS = {
    "NC": "#19b394",
    "IC": "#ef5a72",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot tuned-lens head margin distributions from cached JSONL data.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--x-label", type=str, default="Tuned lens value")
    ap.add_argument("--panel-width", type=float, default=3.3, help="Width of each subplot in inches.")
    ap.add_argument("--panel-height", type=float, default=2.6, help="Height of each subplot in inches.")
    ap.add_argument("--font-scale", type=float, default=1.0, help="Global font scaling factor.")
    ap.add_argument("--bandwidth", type=float, default=0.75)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--quantile-low", type=float, default=0.02)
    ap.add_argument("--quantile-high", type=float, default=0.98)
    ap.add_argument("--pad-ratio", type=float, default=0.10)
    ap.add_argument("--xtick-step", type=float, default=2.0)
    ap.add_argument("--per-head-range", action="store_true", default=True)
    ap.add_argument(
        "--xtick-count",
        type=int,
        default=3,
        help="Number of x-axis tick labels to show per subplot.",
    )
    ap.add_argument(
        "--only-nc-gt-ic",
        action="store_true",
        help="Keep only heads whose NC mean margin is larger than IC mean margin.",
    )
    ap.add_argument(
        "--topk-heads",
        type=int,
        default=0,
        help="Optional limit on the number of plotted heads after filtering/sorting.",
    )
    ap.add_argument(
        "--sort-by-gap",
        action="store_true",
        help="Sort heads by (NC mean - IC mean) descending instead of head id.",
    )
    ap.add_argument(
        "--annotate-stats",
        action="store_true",
        help="Annotate each subplot with NC mean, IC mean, and gap.",
    )
    return ap.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gaussian_kde_curve(values: list[float], grid: np.ndarray, bandwidth: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros_like(grid)
    if arr.size == 1:
        sigma = max(1e-3, bandwidth)
    else:
        sigma = np.std(arr, ddof=1)
        if sigma < 1e-6:
            q25, q75 = np.quantile(arr, [0.25, 0.75])
            sigma = max(abs(q75 - q25) / 1.349, 1e-3)
        sigma = max(1e-3, sigma * bandwidth)
    diffs = (grid[:, None] - arr[None, :]) / sigma
    density = np.exp(-0.5 * diffs**2).sum(axis=1)
    density /= arr.size * sigma * math.sqrt(2.0 * math.pi)
    return density


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data)
    if not rows:
        raise RuntimeError(f"No rows found in {args.data}")

    grouped_rows: dict[tuple[int, int, str], list[dict]] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]), row["head_label"])
        grouped_rows.setdefault(key, []).append(row)

    head_infos = []
    for key, head_rows in grouped_rows.items():
        nc = np.asarray([float(row["margin"]) for row in head_rows if row["condition"] == "NC"], dtype=float)
        ic = np.asarray([float(row["margin"]) for row in head_rows if row["condition"] == "IC"], dtype=float)
        nc_mean = float(np.mean(nc)) if nc.size else float("nan")
        ic_mean = float(np.mean(ic)) if ic.size else float("nan")
        gap = nc_mean - ic_mean
        head_infos.append(
            {
                "key": key,
                "head_rows": head_rows,
                "nc_mean": nc_mean,
                "ic_mean": ic_mean,
                "gap": gap,
            }
        )

    if args.only_nc_gt_ic:
        head_infos = [info for info in head_infos if np.isfinite(info["gap"]) and info["gap"] > 0]

    if args.sort_by_gap:
        head_infos.sort(key=lambda info: (-info["gap"], int(info["key"][2].split()[1])))
    else:
        head_infos.sort(key=lambda info: int(info["key"][2].split()[1]))

    if args.topk_heads > 0:
        head_infos = head_infos[: args.topk_heads]

    head_order = [info["key"] for info in head_infos]
    head_rows_map = {info["key"]: info["head_rows"] for info in head_infos}
    head_stats_map = {info["key"]: info for info in head_infos}
    if not head_order:
        raise RuntimeError("No heads left after filtering. Relax the head filter settings and try again.")

    all_values = np.asarray([float(row["margin"]) for row in rows], dtype=float)
    global_q_low = float(np.quantile(all_values, args.quantile_low))
    global_q_high = float(np.quantile(all_values, args.quantile_high))

    n_heads = len(head_order)
    n_cols = max(1, args.cols)
    n_rows = math.ceil(n_heads / n_cols)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8 * args.font_scale,
            "axes.titlesize": 9 * args.font_scale,
            "axes.labelsize": 8 * args.font_scale,
            "xtick.labelsize": 7 * args.font_scale,
            "ytick.labelsize": 7 * args.font_scale,
            "legend.fontsize": 7 * args.font_scale,
        }
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * args.panel_width, n_rows * args.panel_height),
        squeeze=False,
    )

    for idx, (layer, head, head_label) in enumerate(head_order):
        ax = axes[idx // n_cols][idx % n_cols]
        head_key = (layer, head, head_label)
        head_rows = head_rows_map[head_key]
        stats = head_stats_map[head_key]
        head_values = np.asarray([float(row["margin"]) for row in head_rows], dtype=float)
        if args.per_head_range:
            x_min = float(np.quantile(head_values, args.quantile_low))
            x_max = float(np.quantile(head_values, args.quantile_high))
        else:
            x_min = global_q_low
            x_max = global_q_high
        span = x_max - x_min if x_max > x_min else 1.0
        pad = max(0.10, args.pad_ratio * span)
        grid = np.linspace(x_min - pad, x_max + pad, 300)

        for cond in ("NC", "IC"):
            values = [float(row["margin"]) for row in head_rows if row["condition"] == cond]
            if not values:
                continue
            visible_values = [value for value in values if grid[0] <= value <= grid[-1]]
            density = gaussian_kde_curve(visible_values or values, grid, bandwidth=args.bandwidth)
            ax.fill_between(grid, density, color=COLORS[cond], alpha=0.65, label=cond)
            ax.plot(grid, density, color=COLORS[cond], linewidth=1.1)

        ax.set_title(f"L{layer:02d}H{head:02d}", pad=6)
        ax.set_xlabel(args.x_label)
        if idx % n_cols == 0:
            ax.set_ylabel("Density")
        ax.set_xlim(grid[0], grid[-1])
        if args.xtick_count > 1:
            tick_positions = np.linspace(grid[0], grid[-1], args.xtick_count)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{tick:.2f}" for tick in tick_positions])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(length=3)
        if args.annotate_stats:
            stat_text = "\n".join(
                [
                    f"NC {stats['nc_mean']:.3f}",
                    f"IC {stats['ic_mean']:.3f}",
                    f"\u0394 {stats['gap']:.3f}",
                ]
            )
            ax.text(
                0.02,
                0.96,
                stat_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.5 * args.font_scale,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    for idx in range(n_heads, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.995), frameon=False, ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.98))

    png_path = args.out_dir / "tuned_lens_head_distribution_val.png"
    pdf_path = args.out_dir / "tuned_lens_head_distribution_val.pdf"
    meta_path = args.out_dir / "tuned_lens_head_distribution_val.meta.json"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "data": str(args.data),
        "n_rows": len(rows),
        "n_heads": n_heads,
        "conditions": ["NC", "IC"],
        "only_nc_gt_ic": args.only_nc_gt_ic,
        "topk_heads": args.topk_heads,
        "sort_by_gap": args.sort_by_gap,
        "annotate_stats": args.annotate_stats,
        "xtick_count": args.xtick_count,
        "panel_width": args.panel_width,
        "panel_height": args.panel_height,
        "font_scale": args.font_scale,
        "x_label": args.x_label,
        "x_range_global": [global_q_low, global_q_high],
        "quantiles": [args.quantile_low, args.quantile_high],
        "outputs": [str(png_path), str(pdf_path)],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
