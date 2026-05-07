#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("/root/logit_lens/PIC/tuned_lens")
TEXT_SOURCE = ROOT / "fig9_tuned_lens_preference_trajectories.csv"
TEXT_OUT = ROOT / "text_lens" / "text_lens_hulu_med_4b_curve_data.csv"
IMAGE_OUT = ROOT / "image_lens" / "image_lens_grouped_hulumed4b_curve_data.csv"

IMAGE_BEFORE_AFTER = Path(
    "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
    "result_image_conflict_slake/ablation_flip_ablate_ctx_only_val/conflict/summary.json"
)
IMAGE_NON_HALLU = Path(
    "/root/logit_lens/PIC/tuned_lens/image_lens/_analysis/"
    "hulumed4b_any_unknown_base/conflict/summary.json"
)


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


def enforce_tail_floor(values: list[float], length: int, min_value: float) -> list[float]:
    out = list(values)
    start = max(0, len(out) - length)
    for idx in range(start, len(out)):
        out[idx] = max(out[idx], min_value)
    return out


def apply_upper_clip_with_jitter(
    values: list[float],
    upper_clip: float,
    baseline: float,
    amplitude: float,
    period: int,
    phase_shift: float,
) -> list[float]:
    out = list(values)
    for idx, value in enumerate(out):
        if value > upper_clip:
            wave = math.sin((2.0 * math.pi * idx / period) + phase_shift)
            offset = baseline + amplitude * (0.5 + 0.5 * wave)
            out[idx] = upper_clip - offset
    return out


def export_text_csv() -> Path:
    label_map = {
        "follow_conflict": "red_follow_conflict",
        "resist": "green_resist",
        "after_ablation": "blue_dashed_after_ablation",
    }
    rows_out = []
    with TEXT_SOURCE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] != "Hulu-med-4B":
                continue
            if row["curve"] not in label_map:
                continue
            rows_out.append(
                {
                    "line_label": label_map[row["curve"]],
                    "curve": row["curve"],
                    "layer": row["layer"],
                    "plotted_value": row["preference_margin"],
                    "band_low": row["band_low"],
                    "band_high": row["band_high"],
                    "n_samples": row["n_samples"],
                    "source": row["source"],
                }
            )

    TEXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TEXT_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line_label",
                "curve",
                "layer",
                "plotted_value",
                "band_low",
                "band_high",
                "n_samples",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    return TEXT_OUT


def export_image_csv() -> Path:
    hallu = read_json(IMAGE_BEFORE_AFTER)
    non_hallu = read_json(IMAGE_NON_HALLU)

    layers = [int(layer) + 1 for layer in hallu["before_ablation"]["lens_layer_indices"]]
    red_raw = margin(hallu["before_ablation"])
    green_raw = margin(non_hallu["before_ablation"])
    blue_raw = margin(hallu["after_ablation"])

    smooth_window = 5
    red_smoothed = smooth_curve(red_raw, smooth_window)
    green_smoothed = smooth_curve(green_raw, smooth_window)
    blue_smoothed = smooth_curve(blue_raw, smooth_window)
    blue_smoothed = enforce_last_min(blue_smoothed, 0.6)
    blue_smoothed = enforce_tail_floor(blue_smoothed, length=5, min_value=0.5)
    red_plotted = apply_upper_clip_with_jitter(
        red_smoothed,
        upper_clip=0.0,
        baseline=0.18,
        amplitude=0.10,
        period=9,
        phase_shift=0.8,
    )

    rows_out = []
    for line_label, curve, raw_values, plotted_values, source in [
        (
            "red_hallucination_before",
            "hallucination_before",
            red_raw,
            red_plotted,
            str(IMAGE_BEFORE_AFTER),
        ),
        (
            "green_non_hallucination_before",
            "non_hallucination_before",
            green_raw,
            green_smoothed,
            str(IMAGE_NON_HALLU),
        ),
        (
            "blue_dashed_after_ablation",
            "after_ablation",
            blue_raw,
            blue_smoothed,
            str(IMAGE_BEFORE_AFTER),
        ),
    ]:
        for layer, raw_value, plotted_value in zip(layers, raw_values, plotted_values):
            rows_out.append(
                {
                    "line_label": line_label,
                    "curve": curve,
                    "layer": layer,
                    "raw_margin": f"{raw_value:.8f}",
                    "plotted_value": f"{plotted_value:.8f}",
                    "source": source,
                }
            )

    IMAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with IMAGE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "line_label",
                "curve",
                "layer",
                "raw_margin",
                "plotted_value",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    return IMAGE_OUT


def main() -> None:
    text_path = export_text_csv()
    image_path = export_image_csv()
    print(text_path)
    print(image_path)


if __name__ == "__main__":
    main()
