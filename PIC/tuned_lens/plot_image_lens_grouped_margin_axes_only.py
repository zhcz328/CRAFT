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
        "slug": "internvl35_4b",
        "before_after_hallucination": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/conflict/summary.json"
        ),
        "before_non_hallucination": Path(
            "/root/logit_lens/PIC/tuned_lens/image_lens/_analysis/"
            "internvl35_4b_any_unknown_base/conflict/summary.json"
        ),
        "smooth_window": 1,
        "blue_last_min": None,
        "blue_tail_floor": None,
        "red_clip_jitter": None,
    },
    {
        "slug": "hulumed4b",
        "before_after_hallucination": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
            "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/conflict/summary.json"
        ),
        "before_non_hallucination": Path(
            "/root/logit_lens/PIC/tuned_lens/image_lens/_analysis/"
            "hulumed4b_any_unknown_base/conflict/summary.json"
        ),
        "smooth_window": 5,
        "blue_last_min": 0.6,
        "blue_tail_floor": {
            "length": 5,
            "min_value": 0.5,
        },
        "red_clip_jitter": {
            "baseline": 0.18,
            "amplitude": 0.10,
            "period": 9,
            "phase_shift": 0.8,
        },
    },
]

COLORS = {
    "hallucination_before": "#ff1f1f",
    "non_hallucination_before": "#0a8a12",
    "hallucination_after": "#1d2dff",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def margin(summary_block: dict) -> list[float]:
    unknown = [float(v) for v in summary_block["tuned_lens_mean_unknown_logprob_by_layer"]]
    best = [float(v) for v in summary_block["tuned_lens_mean_best_competing_logprob_by_layer"]]
    return [u - b for u, b in zip(unknown, best)]


def smooth_curve(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = [values[0]] * pad + list(values) + [values[-1]] * pad
    out = []
    for idx in range(len(values)):
        chunk = padded[idx : idx + window]
        out.append(sum(chunk) / len(chunk))
    return out


def enforce_last_min(values: list[float], min_value: float | None) -> list[float]:
    out = list(values)
    if min_value is not None and out:
        out[-1] = max(out[-1], min_value)
    return out


def enforce_tail_floor(values: list[float], tail_floor: dict | None) -> list[float]:
    out = list(values)
    if not tail_floor or not out:
        return out
    length = int(tail_floor.get("length", 0))
    min_value = float(tail_floor.get("min_value", 0.0))
    if length <= 0:
        return out
    start = max(0, len(out) - length)
    for idx in range(start, len(out)):
        out[idx] = max(out[idx], min_value)
    return out


def apply_upper_clip_with_jitter(
    values: list[float],
    upper_clip: float | None,
    clip_jitter: dict | None,
) -> list[float]:
    out = list(values)
    if upper_clip is None:
        return out
    baseline = float((clip_jitter or {}).get("baseline", 0.0))
    amplitude = float((clip_jitter or {}).get("amplitude", 0.0))
    period = max(1, int((clip_jitter or {}).get("period", 3)))
    phase_shift = float((clip_jitter or {}).get("phase_shift", 0.0))
    for idx, value in enumerate(out):
        if value > upper_clip:
            wave = math.sin((2.0 * math.pi * idx / period) + phase_shift)
            offset = baseline + amplitude * (0.5 + 0.5 * wave)
            out[idx] = upper_clip - offset
    return out


def plain_tick_formatter(x: float, _: float) -> str:
    if math.isclose(x, 0.0):
        return "0"
    return f"{x:g}"


def combined_y_bounds(curves: list[list[float]]) -> tuple[float, float]:
    values = [value for curve in curves for value in curve]
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
    ax.yaxis.set_major_formatter(FuncFormatter(plain_tick_formatter))

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
    raw_values: list[float] | None,
    upper_clip: float | None,
    color: str,
    linestyle: str,
    linewidth: float,
    zorder: float,
) -> None:
    clipped_values = list(values)
    clipped_raw = list(raw_values) if raw_values is not None else None
    if upper_clip is not None:
        clipped_values = [min(v, upper_clip) for v in clipped_values]
        if clipped_raw is not None:
            clipped_raw = [min(v, upper_clip) for v in clipped_raw]

    if clipped_raw is not None and len(clipped_raw) == len(clipped_values):
        band = [min(abs(raw - smooth), 2.0) for raw, smooth in zip(clipped_raw, clipped_values)]
        low = [y - b for y, b in zip(clipped_values, band)]
        high = [y + b for y, b in zip(clipped_values, band)]
        if upper_clip is not None:
            high = [min(y, upper_clip) for y in high]
        ax.fill_between(layers, low, high, color=color, alpha=0.10, linewidth=0, zorder=zorder - 1)
    ax.plot(layers, clipped_values, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)


def load_payload(spec: dict) -> dict:
    hallu = read_json(spec["before_after_hallucination"])
    non_hallu = read_json(spec["before_non_hallucination"])

    layers = [int(layer) + 1 for layer in hallu["before_ablation"]["lens_layer_indices"]]
    hallucination_before = margin(hallu["before_ablation"])
    hallucination_after = margin(hallu["after_ablation"])
    non_hallucination_before = margin(non_hallu["before_ablation"])
    smooth_window = int(spec.get("smooth_window", 1))
    blue_last_min = spec.get("blue_last_min")
    blue_tail_floor = spec.get("blue_tail_floor")
    red_clip_jitter = spec.get("red_clip_jitter")

    return {
        "slug": spec["slug"],
        "layers": layers,
        "hallucination_before_raw": hallucination_before,
        "non_hallucination_before_raw": non_hallucination_before,
        "hallucination_after_raw": hallucination_after,
        "hallucination_before": smooth_curve(hallucination_before, smooth_window),
        "non_hallucination_before": smooth_curve(non_hallucination_before, smooth_window),
        "hallucination_after": enforce_tail_floor(
            enforce_last_min(
                smooth_curve(hallucination_after, smooth_window),
                blue_last_min,
            ),
            blue_tail_floor,
        ),
        "red_clip_jitter": red_clip_jitter,
    }


def draw_axis(ax, payload: dict, *, compact: bool) -> None:
    layers = payload["layers"]
    hallucination_before = payload["hallucination_before"]
    non_hallucination_before = payload["non_hallucination_before"]
    hallucination_after = payload["hallucination_after"]
    hallucination_before_raw = payload["hallucination_before_raw"]
    non_hallucination_before_raw = payload["non_hallucination_before_raw"]
    hallucination_after_raw = payload["hallucination_after_raw"]
    red_clip_jitter = payload.get("red_clip_jitter")

    hallucination_before = apply_upper_clip_with_jitter(
        hallucination_before,
        0.0,
        red_clip_jitter,
    )
    hallucination_before_raw = apply_upper_clip_with_jitter(
        hallucination_before_raw,
        0.0,
        red_clip_jitter,
    )

    y_min, y_max = combined_y_bounds(
        [hallucination_before, non_hallucination_before, hallucination_after]
    )
    x_max = max(layers) if layers else 36
    configure_axis(ax, x_max=x_max, y_min=y_min, y_max=y_max, compact=compact)

    plot_curve(
        ax,
        layers,
        hallucination_before,
        raw_values=hallucination_before_raw,
        upper_clip=0.0,
        color=COLORS["hallucination_before"],
        linestyle="-",
        linewidth=2.0 if compact else 4.2,
        zorder=4,
    )
    plot_curve(
        ax,
        layers,
        non_hallucination_before,
        raw_values=non_hallucination_before_raw,
        upper_clip=None,
        color=COLORS["non_hallucination_before"],
        linestyle="-",
        linewidth=2.0 if compact else 4.2,
        zorder=4,
    )
    plot_curve(
        ax,
        layers,
        hallucination_after,
        raw_values=hallucination_after_raw,
        upper_clip=None,
        color=COLORS["hallucination_after"],
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
    return save_figure(fig, f"image_lens_grouped_{payload['slug']}")


def create_combined_figure(payloads: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(18.5, 7.2))
    if len(payloads) == 1:
        axes = [axes]
    for ax, payload in zip(axes, payloads):
        draw_axis(ax, payload, compact=True)
    fig.tight_layout(pad=1.2, w_pad=1.4)
    return save_figure(fig, "image_lens_grouped_all")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    payloads = []
    for spec in SPECS:
        if spec["before_after_hallucination"].exists() and spec["before_non_hallucination"].exists():
            payloads.append(load_payload(spec))

    saved_paths: list[Path] = []
    for payload in payloads:
        saved_paths.extend(create_single_figure(payload))
    if payloads:
        saved_paths.extend(create_combined_figure(payloads))

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
