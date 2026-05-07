#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


OUT_DIR = Path("/root/logit_lens/PIC/tuned_lens/image_lens")

SPECS = [
    {
        "title": "InternVL3.5-4B Conflict",
        "slug": "internvl35_4b_conflict",
        "summary": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/conflict/summary.json"
        ),
    },
    {
        "title": "InternVL3.5-4B Hallucination",
        "slug": "internvl35_4b_hallucination",
        "summary": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/hallucination/summary.json"
        ),
    },
    {
        "title": "Hulu-med-4B Conflict",
        "slug": "hulumed4b_conflict",
        "summary": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/conflict/summary.json"
        ),
    },
    {
        "title": "Hulu-med-4B Hallucination",
        "slug": "hulumed4b_hallucination",
        "summary": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/hallucination/summary.json"
        ),
    },
]

COLORS = {
    "unknown_before": "#ff1f1f",
    "best_before": "#0a8a12",
    "best_after": "#1d2dff",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plus_tick_formatter(x: float, _: float) -> str:
    if math.isclose(x, 0.0):
        return "0"
    return f"{x:g}"


def combined_y_bounds(curve_sets: list[tuple[list[int], list[float]]]) -> tuple[float, float]:
    values: list[float] = []
    for _, ys in curve_sets:
        values.extend(ys)
    if not values:
        return -1.0, 1.0
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = max(1.0, abs(y_min) * 0.1)
    else:
        pad = max(0.25, 0.12 * (y_max - y_min))
    return y_min - pad, y_max + pad


def configure_axis(ax, *, x_max: int, y_min: float, y_max: float, compact: bool) -> None:
    tick_size = 10 if compact else 16

    ax.set_facecolor("#fffdf8")
    ax.axhspan(0.0, y_max, facecolor="#dff3de", alpha=0.50, zorder=0)
    ax.axhspan(y_min, 0.0, facecolor="#ffdede", alpha=0.52, zorder=0)
    ax.axhline(0.0, color="black", linestyle=(0, (6, 3)), linewidth=1.7 if compact else 2.3, zorder=2)

    ax.set_xlim(1, x_max)
    ax.set_ylim(y_min, y_max)

    xticks = [1]
    for tick in [6, 12, 18, 24, 30, 36]:
        if tick < x_max:
            xticks.append(tick)
    if x_max not in xticks:
        xticks.append(x_max)
    ax.set_xticks(sorted(set(xticks)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7 if compact else 8))
    ax.yaxis.set_major_formatter(FuncFormatter(plus_tick_formatter))

    ax.grid(True, color="#a7a7a7", linewidth=0.6 if compact else 1.0, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2 if compact else 1.8)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=tick_size, width=1.2 if compact else 1.6, length=5 if compact else 8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def plot_curve(
    ax,
    layers: list[int],
    values: list[float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    zorder: float,
) -> None:
    ax.plot(
        layers,
        values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def load_payload(spec: dict) -> dict:
    summary = read_json(spec["summary"])
    before = summary["before_ablation"]
    after = summary["after_ablation"]

    layers = [int(layer) + 1 for layer in before["lens_layer_indices"]]
    unknown_before = [float(v) for v in before["tuned_lens_mean_unknown_logprob_by_layer"]]
    best_before = [float(v) for v in before["tuned_lens_mean_best_competing_logprob_by_layer"]]
    unknown_after = [float(v) for v in after["tuned_lens_mean_unknown_logprob_by_layer"]]
    best_after = [float(v) for v in after["tuned_lens_mean_best_competing_logprob_by_layer"]]

    before_margin = [u - b for u, b in zip(unknown_before, best_before)]
    after_margin = [u - b for u, b in zip(unknown_after, best_after)]

    return {
        "title": spec["title"],
        "slug": spec["slug"],
        "layers": layers,
        "before_margin": before_margin,
        "after_margin": after_margin,
    }


def draw_axis(ax, payload: dict, *, compact: bool) -> None:
    layers = payload["layers"]
    before_margin = payload["before_margin"]
    after_margin = payload["after_margin"]

    y_min, y_max = combined_y_bounds(
        [
            (layers, before_margin),
            (layers, after_margin),
        ]
    )
    x_max = max(layers) if layers else 36
    configure_axis(ax, x_max=x_max, y_min=y_min, y_max=y_max, compact=compact)

    plot_curve(
        ax,
        layers,
        before_margin,
        color=COLORS["unknown_before"],
        linestyle="-",
        linewidth=2.0 if compact else 4.2,
        zorder=4,
    )
    plot_curve(
        ax,
        layers,
        after_margin,
        color=COLORS["best_after"],
        linestyle="--",
        linewidth=2.2 if compact else 4.0,
        zorder=5,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def create_single_figure(payload: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.2, 10.6))
    draw_axis(ax, payload, compact=False)
    fig.tight_layout()
    return save_figure(fig, f"image_lens_{payload['slug']}")


def create_combined_figure(payloads: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 13.8))
    for ax, payload in zip(axes.flatten(), payloads):
        draw_axis(ax, payload, compact=True)
    fig.tight_layout(pad=1.2, w_pad=1.4, h_pad=1.6)
    return save_figure(fig, "image_lens_all")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    payloads = [load_payload(spec) for spec in SPECS if spec["summary"].exists()]
    saved_paths: list[Path] = []
    for payload in payloads:
        saved_paths.extend(create_single_figure(payload))
    if payloads:
        saved_paths.extend(create_combined_figure(payloads))

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
