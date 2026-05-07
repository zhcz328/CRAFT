#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import plot_fig9_tuned_lens_reference_style as ref


OUT_DIR = Path("/root/logit_lens/PIC/tuned_lens/text_lens")
COLORS = ref.COLORS


def plus_tick_formatter(x: float, _: float) -> str:
    if abs(x) < 1e-12:
        return "0"
    return f"{x:g}"


def plot_curve(
    ax,
    layers: list[int],
    values: list[float],
    bands: list[float],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    alpha: float,
    zorder: float,
) -> None:
    n = min(len(layers), len(values), len(bands))
    x = layers[:n]
    y = values[:n]
    b = [bb * 1.15 for bb in bands[:n]]
    if any(bb > 0 for bb in b):
        low = [yy - bb for yy, bb in zip(y, b)]
        high = [yy + bb for yy, bb in zip(y, b)]
        ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0, zorder=zorder - 1)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)


def configure_axis(ax, *, x_max: int, y_min: float, y_max: float, compact: bool) -> None:
    tick_size = 20 if compact else 32

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


def draw_model_axis(ax, payload: dict, *, compact: bool) -> None:
    summary = payload["summary"]
    follow_layers = payload["follow_layers"]
    follow_values = payload["follow_values"]
    follow_bands = payload["follow_bands"]
    resist_layers = payload["resist_layers"]
    resist_values = payload["resist_values"]
    resist_bands = payload["resist_bands"]
    after_layers = payload["after_layers"]
    after_values = payload["after_values"]
    after_bands = payload["after_bands"]

    y_min, y_max = ref.combined_y_bounds(
        [
            (follow_layers, follow_values, follow_bands),
            (resist_layers, resist_values, resist_bands),
            (after_layers, after_values, after_bands),
        ]
    )
    x_max = int(max(summary["layer_indices"])) + 1 if summary.get("layer_indices") else 36
    configure_axis(ax, x_max=x_max, y_min=y_min, y_max=y_max, compact=compact)

    plot_curve(
        ax,
        follow_layers,
        follow_values,
        follow_bands,
        color=COLORS["follow_conflict"],
        linestyle="-",
        linewidth=2.0 if compact else 4.2,
        alpha=0.18 if compact else 0.24,
        zorder=4,
    )
    plot_curve(
        ax,
        resist_layers,
        resist_values,
        resist_bands,
        color=COLORS["resist"],
        linestyle="-",
        linewidth=2.0 if compact else 4.2,
        alpha=0.18 if compact else 0.24,
        zorder=4,
    )
    plot_curve(
        ax,
        after_layers,
        after_values,
        after_bands,
        color=COLORS["after"],
        linestyle="--",
        linewidth=2.2 if compact else 4.0,
        alpha=0.15 if compact else 0.20,
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


def create_single_model_figure(payload: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.2, 10.6))
    draw_model_axis(ax, payload, compact=False)
    fig.tight_layout()
    return save_figure(fig, f"text_lens_{payload['model']['slug']}")


def create_combined_figure(payloads: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 13.8))
    for ax, payload in zip(axes.flatten(), payloads):
        draw_model_axis(ax, payload, compact=True)
    fig.tight_layout(pad=1.2, w_pad=1.4, h_pad=1.6)
    return save_figure(fig, "text_lens_all_models")


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

    ref.ensure_qwen_flip_summary()

    payloads: list[dict] = []
    for model in ref.MODELS:
        payload, _, _, _ = ref.prepare_model_payload(model)
        if payload is not None:
            payloads.append(payload)

    saved_paths: list[Path] = []
    for payload in payloads:
        saved_paths.extend(create_single_model_figure(payload))
    if payloads:
        saved_paths.extend(create_combined_figure(payloads))

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
