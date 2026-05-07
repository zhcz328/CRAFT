#!/usr/bin/env python3
"""
Draw Fig. 8 style probe-score-by-layer curves from saved probe summaries.

Data source:
  Probe-A: summary.json -> metrics_by_layer[*].val_metrics.auroc
  Probe-B: summary.json -> metrics_by_layer[*].val_metrics.macro_f1

The script writes:
  - fig8_probe_scores_by_layer.png
  - fig8_probe_scores_by_layer.pdf
  - fig8_probe_scores_by_layer.csv
  - data_sources.md
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("/root/logit_lens/PIC/fig8")

SERIES = [
    {
        "label": "Qwen3-4B",
        "color": "#1f77b4",
        "conflict": Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/conflict_linear_before_question/summary.json"
        ),
        "follow": Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/follow_linear_before_question/summary.json"
        ),
    },
    {
        "label": "InternVL3.5-4B",
        "color": "#d62728",
        "conflict": Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/conflict_linear/summary.json"
        ),
        "follow": Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/follow_conflict_linear/summary.json"
        ),
    },
    {
        "label": "HuluMed",
        "color": "#2ca02c",
        "conflict": Path(
            "/root/autodl-tmp/Hulumed/probe_slake_vqa/results/"
            "conflict_linear_before_question/summary.json"
        ),
        "follow": Path(
            "/root/autodl-tmp/Hulumed/probe_slake_vqa/results/"
            "follow_linear_before_question/summary.json"
        ),
    },
    {
        "label": "Llama3.2-3B",
        "color": "#ff7f0e",
        "conflict": Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/conflict_linear/summary.json"
        ),
        "follow": Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/follow_linear/summary.json"
        ),
    },
]


PROBES = [
    {
        "title": "Probe-A: Conflict Detection",
        "source_key": "conflict",
        "metric": "auroc",
        "ylabel": "AUROC",
        "baseline": 0.5,
    },
    {
        "title": "Probe-B: Follow-Conflict Prediction",
        "source_key": "follow",
        "metric": "macro_f1",
        "ylabel": "Macro-F1",
        "baseline": 0.5,
    },
]


def read_metric_curve(path: Path, metric: str) -> list[tuple[int, float]]:
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    curve = []
    for item in summary["metrics_by_layer"]:
        # Store layers as 1-based values for plotting, matching paper-style figures.
        layer = int(item["layer_idx"]) + 1
        value = float(item["val_metrics"][metric])
        curve.append((layer, value))
    return sorted(curve)


def best_layer(curve: list[tuple[int, float]]) -> tuple[int, float]:
    return max(curve, key=lambda x: x[1])


def estimate_visual_band(
    curve: list[tuple[int, float]],
    min_width: float = 0.025,
    max_width: float = 0.085,
) -> list[tuple[float, float]]:
    """Estimate a narrow visual band from local layer-to-layer variation.

    The result files have one validation score per layer rather than repeated
    seeds or bootstrap samples, so this is a readability band, not a confidence
    interval.
    """
    values = [value for _, value in curve]
    band = []
    for idx, value in enumerate(values):
        start = max(0, idx - 1)
        end = min(len(values), idx + 2)
        local_values = values[start:end]
        if len(local_values) > 1:
            width = statistics.pstdev(local_values) * 0.9
        else:
            width = min_width
        width = min(max(width, min_width), max_width)
        band.append((max(0.0, value - width), min(1.0, value + width)))
    return band


def write_curve_csv(rows: list[dict[str, object]]) -> None:
    csv_path = OUT_DIR / "fig8_probe_scores_by_layer.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "probe",
                "model",
                "layer",
                "metric",
                "value",
                "band_low",
                "band_high",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_sources_md(sources: list[dict[str, object]]) -> None:
    md_path = OUT_DIR / "data_sources.md"
    lines = [
        "# Fig. 8 Data Sources",
        "",
        "This figure reads validation metrics from:",
        "",
        "- Probe-A: `summary.json -> metrics_by_layer[*].val_metrics.auroc`",
        "- Probe-B: `summary.json -> metrics_by_layer[*].val_metrics.macro_f1`",
        "",
        "Layers in the plot are displayed as 1-based layer indices. "
        "The JSON files store `layer_idx` as 0-based indices.",
        "",
        "The shaded regions are local visual bands estimated from adjacent-layer "
        "score variation because these result files contain one run per layer. "
        "They are not confidence intervals.",
        "",
        "| Probe | Model | Metric | Source | Best plotted layer | Best validation score |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in sources:
        lines.append(
            f"| {item['probe']} | {item['model']} | {item['metric']} | "
            f"`{item['source']}` | {item['best_layer']} | {item['best_value']:.6f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def style_axis(
    ax,
    title: str,
    ylabel: str,
    baseline: float,
    x_max: int,
    y_min: float,
) -> None:
    ax.set_title(title, fontsize=12, pad=5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, x_max)
    ax.set_ylim(y_min, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, x_max])
    ax.axhline(baseline, color="0.55", linestyle="--", linewidth=1.0, zorder=0)
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax.grid(False, axis="x")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict[str, list[tuple[int, float]]]] = {}
    csv_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    max_layer = 0

    for series in SERIES:
        label = str(series["label"])
        curves[label] = {}
        for probe in PROBES:
            probe_name = str(probe["title"])
            source_key = str(probe["source_key"])
            metric = str(probe["metric"])
            source = series[source_key]
            curve = read_metric_curve(source, metric)
            curves[label][probe_name] = curve
            max_layer = max(max_layer, max(layer for layer, _ in curve))

            best_l, best_v = best_layer(curve)
            source_rows.append(
                {
                    "probe": probe_name,
                    "model": label,
                    "metric": metric,
                    "source": source,
                    "best_layer": best_l,
                    "best_value": best_v,
                }
            )
            band = estimate_visual_band(curve)
            for (layer, value), (band_low, band_high) in zip(curve, band):
                csv_rows.append(
                    {
                        "probe": probe_name,
                        "model": label,
                        "layer": layer,
                        "metric": metric,
                        "value": f"{value:.8f}",
                        "band_low": f"{band_low:.8f}",
                        "band_high": f"{band_high:.8f}",
                        "source": str(source),
                    }
                )

    write_curve_csv(csv_rows)
    write_sources_md(source_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=False)

    for ax, probe in zip(axes, PROBES):
        probe_name = str(probe["title"])
        y_min = 1.0
        for series in SERIES:
            label = str(series["label"])
            curve = curves[label][probe_name]
            y_min = min(y_min, min(low for low, _ in estimate_visual_band(curve)))
        y_min = max(0.0, y_min - 0.025)
        style_axis(
            ax,
            probe_name,
            str(probe["ylabel"]),
            float(probe["baseline"]),
            max_layer,
            y_min,
        )
        for series in SERIES:
            label = str(series["label"])
            color = str(series["color"])
            curve = curves[label][probe_name]
            layers = [x for x, _ in curve]
            values = [y for _, y in curve]
            band = estimate_visual_band(curve)
            lows = [low for low, _ in band]
            highs = [high for _, high in band]
            ax.fill_between(layers, lows, highs, color=color, alpha=0.16, linewidth=0)
            ax.plot(layers, values, color=color, linewidth=1.8, label=label)

            best_l, best_v = best_layer(curve)
            ax.scatter(
                [best_l],
                [best_v],
                s=18,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.tight_layout(w_pad=1.6)

    png_path = OUT_DIR / "fig8_probe_scores_by_layer.png"
    pdf_path = OUT_DIR / "fig8_probe_scores_by_layer.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {OUT_DIR / 'fig8_probe_scores_by_layer.csv'}")
    print(f"Saved: {OUT_DIR / 'data_sources.md'}")


if __name__ == "__main__":
    main()
