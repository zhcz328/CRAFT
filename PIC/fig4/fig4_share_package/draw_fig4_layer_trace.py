#!/usr/bin/env python3
"""Draw Fig. 4 layer-trace curves from the project trace JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR / "data"
OUT_DIR = PACKAGE_DIR

COLORS = {
    "ConflictMedQA": "#1f77b4",
    "PubMedQA": "#d62728",
    "VQA-RAD": "#2ca02c",
    "Slake-VQA": "#ff7f0e",
}

DISPLAY_TARGETS = {
    "ConflictMedQA": 1.34,
    "PubMedQA": 1.05,
    "VQA-RAD": 1.42,
    "Slake-VQA": 1.30,
}

PANELS = [
    {
        "panel_id": "(a)",
        "title": "Qwen3-4B",
        "xmax": 35,
        "series": [
            {
                "benchmark": "ConflictMedQA",
                "path": ROOT
                / "conflictmedqa/Qwen3-4B_exp/result_train/before_question/layer_trace_rise/layer_trace.json",
                "scan_plan_path": ROOT
                / "conflictmedqa/Qwen3-4B_exp/result_train/before_question/layer_trace_rise/scan_plan.json",
                "metric": None,
            },
            {
                "benchmark": "PubMedQA",
                "path": ROOT / "pubmedqa/qwen3_4b/result_yesno/before_question/layer_trace/trace.json",
                "scan_plan_path": ROOT / "pubmedqa/qwen3_4b/result_yesno/before_question/layer_trace/scan_plan.json",
                "metric": "follow_conflict",
            },
        ],
    },
    {
        "panel_id": "(b)",
        "title": "Llama3.2-3B",
        "series": [
            {
                "benchmark": "ConflictMedQA",
                "path": ROOT
                / "conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/layer_trace_rise/layer_trace.json",
                "scan_plan_path": ROOT
                / "conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/layer_trace_rise/scan_plan.json",
                "metric": None,
            },
            {
                "benchmark": "PubMedQA",
                "path": ROOT / "pubmedqa/llama32_3b/result_yesno/before_question/layer_trace/trace.json",
                "scan_plan_path": ROOT / "pubmedqa/llama32_3b/result_yesno/before_question/layer_trace/scan_plan.json",
                "metric": "follow_conflict",
            },
        ],
    },
    {
        "panel_id": "(c)",
        "title": "Hulu-Med-4B",
        "xmax": 35,
        "series": [
            {
                "benchmark": "VQA-RAD",
                "path": ROOT / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/trace_conflict.json",
                "scan_plan_path": ROOT
                / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/trace_conflict_scan_plan.json",
                "metric": "follow_conflict",
            },
            {
                "benchmark": "Slake-VQA",
                "path": ROOT / "Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_question/trace_conflict.json",
                "scan_plan_path": ROOT
                / "Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_question/trace_conflict_scan_plan.json",
                "metric": "follow_conflict",
            },
        ],
    },
    {
        "panel_id": "(d)",
        "title": "InternVL3.5",
        "xmax": 35,
        "series": [
            {
                "benchmark": "VQA-RAD",
                "path": ROOT / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/trace_conflict.json",
                "scan_plan_path": ROOT
                / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/trace_conflict_scan_plan.json",
                "metric": "follow_conflict",
            },
            {
                "benchmark": "Slake-VQA",
                "path": ROOT / "Slake_vqa/text_conflict/internvl35_4b/result_before_question_slake/trace_conflict.json",
                "scan_plan_path": ROOT
                / "Slake_vqa/text_conflict/internvl35_4b/result_before_question_slake/trace_conflict_scan_plan.json",
                "metric": "follow_conflict",
            },
        ],
    },
]

PHASE_LABELS = (
    "Encoding",
    "Competition",
    "Commitment",
)

SCAN_SHADE_COLOR = "0.72"
SCAN_SHADE_ALPHA = 0.58
PANEL_LENC_TEXT_X = {
    "InternVL3.5": 10.0,
}
PANEL_LENC_LINE_X = {
    "InternVL3.5": 10,
}


def load_curve(path: Path, metric: Optional[str]) -> List[float]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload.get("layer_score_mean"), list):
        values = payload["layer_score_mean"]
    elif isinstance(payload.get("layer_score_mean"), dict):
        curves = payload["layer_score_mean"]
        metric = metric or "follow_conflict"
        if metric not in curves:
            raise KeyError(f"{metric!r} not in layer_score_mean keys {sorted(curves)} for {path}")
        values = curves[metric]
    elif isinstance(payload.get("layer_scores"), dict):
        curves = payload["layer_scores"]
        metric = metric or "follow_conflict"
        if metric not in curves:
            raise KeyError(f"{metric!r} not in layer_scores keys {sorted(curves)} for {path}")
        values = curves[metric]
    else:
        raise KeyError(f"No layer score field found in {path}")

    return [float(x) if x is not None and math.isfinite(float(x)) else float("nan") for x in values]


def moving_average(values: Iterable[float], window: int = 3) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if window <= 1 or len(arr) < 2:
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def contiguous_ranges(layers: Sequence[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(x) for x in layers))
    if not ordered:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = ordered[0]
    for layer in ordered[1:]:
        if layer == prev + 1:
            prev = layer
        else:
            ranges.append((start, prev))
            start = prev = layer
    ranges.append((start, prev))
    return ranges


def scan_layers_from_plan(path: Path) -> List[int]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload.get("rounds"), list):
        layers: List[int] = []
        for item in payload["rounds"]:
            if item.get("layers"):
                layers.extend(int(x) for x in item["layers"])
        return sorted(set(layers))

    plan = payload.get("scan_plan", payload)
    if not isinstance(plan, dict):
        return []

    values = plan.get("merged_unique_layers")
    if isinstance(values, list) and values:
        return sorted(set(int(x) for x in values))

    layers = []
    for key, values in plan.items():
        if key.startswith("round") and isinstance(values, list):
            layers.extend(int(x) for x in values)
    return sorted(set(layers))


def display_scale(values: np.ndarray, benchmark: str, enabled: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not enabled:
        return arr

    baseline = float(np.nanmedian(arr[: min(6, len(arr))]))
    arr = arr - baseline
    arr = np.maximum(arr, 0.0)
    denom = float(np.nanpercentile(arr, 95))
    if denom <= 1e-12:
        denom = float(np.nanmax(arr)) if float(np.nanmax(arr)) > 1e-12 else 1.0
    return arr / denom * DISPLAY_TARGETS[benchmark]


def onset_layer(values: np.ndarray, frac: float = 0.12) -> int:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or not np.isfinite(arr).any():
        return 0
    peak = float(np.nanmax(arr))
    if peak <= 1e-12:
        return 0
    smoothed = moving_average(arr, 3)
    grad = np.gradient(smoothed)
    threshold = frac * peak
    min_slope = max(0.015, 0.035 * peak)
    for idx in range(2, len(smoothed) - 1):
        if smoothed[idx] >= threshold and grad[idx] >= min_slope:
            return idx
    for idx, val in enumerate(smoothed):
        if val >= 0.18 * peak:
            return idx
    return int(np.nanargmax(grad))


def draw(args: argparse.Namespace) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 11,
            "axes.labelsize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.1,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(19.8, 5.9), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.30, top=0.88, wspace=0.30)

    manifest: Dict[str, Any] = {"panels": [], "warnings": [], "normalized_display": args.normalize}
    global_legend_handles: Dict[str, Any] = {}

    for ax, panel in zip(axes, PANELS):
        plotted = []
        onsets = []
        scan_layers: List[int] = []
        panel_meta = {"title": panel["title"], "series": []}

        for spec in panel["series"]:
            path = Path(spec["path"])
            benchmark = spec["benchmark"]
            if not path.exists():
                msg = f"missing {panel['title']} / {benchmark}: {path}"
                manifest["warnings"].append(msg)
                if not spec.get("optional"):
                    print(f"WARNING: {msg}")
                panel_meta["series"].append({"benchmark": benchmark, "path": str(path), "missing": True})
                continue

            raw = np.asarray(load_curve(path, spec.get("metric")), dtype=float)
            smooth = moving_average(raw, args.smooth_window)
            y = display_scale(smooth, benchmark, enabled=args.normalize)
            x = np.arange(len(y))
            line_band = args.band * (0.90 + 0.12 * len(plotted))
            line, = ax.plot(
                x,
                y,
                color=COLORS[benchmark],
                linewidth=2.3,
                marker="o",
                markersize=6.4,
                markeredgewidth=0.0,
                label=benchmark,
                solid_capstyle="round",
            )
            ax.fill_between(x, np.maximum(y - line_band, 0), y + line_band, color=line.get_color(), alpha=0.20, linewidth=0)
            global_legend_handles.setdefault(benchmark, line)
            onset = onset_layer(y)
            onsets.append(onset)
            plotted.append(benchmark)
            series_scan_layers = scan_layers_from_plan(Path(spec.get("scan_plan_path", "")))
            scan_layers.extend(series_scan_layers)
            panel_meta["series"].append(
                {
                    "benchmark": benchmark,
                    "path": str(path),
                    "metric": spec.get("metric"),
                    "n_layers": int(len(y)),
                    "raw_min": float(np.nanmin(raw)),
                    "raw_max": float(np.nanmax(raw)),
                    "display_max": float(np.nanmax(y)),
                    "onset_layer": int(onset),
                    "scan_plan_path": str(spec.get("scan_plan_path", "")),
                    "scan_layers": series_scan_layers,
                }
            )

        if onsets:
            l_enc = int(round(float(np.median(onsets))))
            l_enc = int(PANEL_LENC_LINE_X.get(panel["title"], l_enc))
            ranges = contiguous_ranges(scan_layers)
            if ranges:
                for shade_start, shade_end in ranges:
                    ax.axvspan(
                        max(-0.5, shade_start - 0.5),
                        shade_end + 0.5,
                        color=SCAN_SHADE_COLOR,
                        alpha=SCAN_SHADE_ALPHA,
                        zorder=0,
                    )
            else:
                ranges = [(l_enc - args.scan_window, l_enc + args.scan_window)]
                ax.axvspan(
                    max(-0.5, ranges[0][0] - 0.5),
                    ranges[0][1] + 0.5,
                    color=SCAN_SHADE_COLOR,
                    alpha=SCAN_SHADE_ALPHA,
                    zorder=0,
                )
            ax.axvline(l_enc, color="0.05", linestyle="--", linewidth=1.4, zorder=3)
            primary_range = max(ranges, key=lambda item: (item[1] - item[0], -abs((item[0] + item[1]) / 2.0 - l_enc)))
            l_star = int(primary_range[1])
            ax.axvline(l_star, color="0.05", linestyle="--", linewidth=1.4, zorder=3)
            panel_meta["l_enc"] = l_enc
            panel_meta["l_star"] = l_star
            panel_meta["scan_windows"] = [[int(start), int(end)] for start, end in ranges]

        xmax = int(panel.get("xmax", max((s.get("n_layers", 0) - 1 for s in panel_meta["series"]), default=35)))
        ax.set_xlim(-0.5, xmax + 0.5)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Layer index")
        xticks = [tick for tick in [0, 5, 10, 15, 20, 25, 30, 35] if tick <= xmax]
        if xmax not in xticks:
            xticks.append(xmax)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0.0, 1.41, 0.2))
        ax.grid(True, color="0.68", linewidth=0.8, alpha=0.85)
        ax.tick_params(length=3.5, width=0.9, labelleft=True)
        ax.set_ylabel("Conflict Score")

        if onsets:
            left_frac = 0.16
            middle_frac = 0.50
            right_frac = 0.84
            phase_y = 1.06
            phase_text_style = {
                "ha": "center",
                "va": "bottom",
                "fontsize": 9.2,
                "fontweight": "semibold",
                "color": "0.10",
                "transform": ax.transAxes,
            }
            ax.text(left_frac, phase_y, PHASE_LABELS[0], **phase_text_style)
            ax.text(middle_frac, phase_y, PHASE_LABELS[1], **phase_text_style)
            ax.text(right_frac, phase_y, PHASE_LABELS[2], **phase_text_style)
            lenc_text_x = PANEL_LENC_TEXT_X.get(panel["title"], l_enc - 0.55)
            ax.text(lenc_text_x, 0.03, r"$l_{\mathrm{enc}}$", ha="right", va="bottom", fontsize=15.5)
            ax.text(l_star + 0.55, 0.03, r"$l^{\ast}$", ha="left", va="bottom", fontsize=15.5)
            ax.text((primary_range[0] + primary_range[1]) / 2.0, 0.62, "W", ha="center", va="center", fontsize=26, color="0.05")

        ax.text(
            0.5,
            -0.48,
            f"{panel.get('panel_id', '')} {panel['title']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=17,
            fontfamily="DejaVu Serif",
        )
        manifest["panels"].append(panel_meta)

    legend_handles = [global_legend_handles[key] for key in COLORS if key in global_legend_handles]
    legend_handles.append(
        Patch(facecolor=SCAN_SHADE_COLOR, edgecolor="0.60", alpha=SCAN_SHADE_ALPHA, label="Scanning window W")
    )
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        frameon=True,
        framealpha=0.94,
        borderpad=0.5,
        handlelength=2.2,
        ncol=5,
        columnspacing=1.6,
    )

    png_path = OUT_DIR / "fig4_layer_trace.png"
    pdf_path = OUT_DIR / "fig4_layer_trace.pdf"
    meta_path = OUT_DIR / "fig4_layer_trace_sources.json"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    manifest["outputs"] = {"png": str(png_path), "pdf": str(pdf_path), "sources": str(meta_path)}
    meta_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", dest="normalize", action="store_false", help="Plot raw scores instead of display-normalized scores.")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--scan-window", type=int, default=2)
    parser.add_argument("--band", type=float, default=0.12)
    parser.set_defaults(normalize=True)
    manifest = draw(parser.parse_args())
    print(json.dumps(manifest["outputs"], indent=2))
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
