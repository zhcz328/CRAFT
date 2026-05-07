#!/usr/bin/env python3
"""
Plot tuned-lens preference trajectories from saved trajectory summaries.

The plotted margin is converted to:
  preference_margin = log p(gold) - log p(wrong)

The trajectory analyzer stores:
  delta = log p(wrong) - log p(gold)

so this script plots -delta.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("/root/logit_lens/PIC/fig8")

MODELS = [
    {
        "title": "Qwen3-4B",
        "root": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results"
        ),
        "summary": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results/trajectory_before_question/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results/trajectory_before_question/trajectories.jsonl"
        ),
    },
    {
        "title": "InternVL3.5-4B",
        "root": Path("/root/autodl-tmp/tuned_lens/internvl35_4b"),
        "summary": Path(
            "/root/autodl-tmp/tuned_lens/internvl35_4b/results/"
            "before_question/trajectory_train_idreg/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/internvl35_4b/results/"
            "before_question/trajectory_train_idreg/trajectories.jsonl"
        ),
    },
    {
        "title": "Hulu-med-4B",
        "root": Path("/root/autodl-tmp/tuned_lens/hulumed_4b"),
        "summary": Path(
            "/root/autodl-tmp/tuned_lens/hulumed_4b/results/"
            "trajectory_before_question_idreg/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/hulumed_4b/results/"
            "trajectory_before_question_idreg/trajectories.jsonl"
        ),
    },
    {
        "title": "Llama3.2-3B",
        "root": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question"
        ),
        "summary": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory_before_question/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory_before_question/trajectories.jsonl"
        ),
    },
]

CURVES = [
    {
        "name": "Follow-conflict samples",
        "kind": "group_raw",
        "group": "follow_conflict",
        "field": "raw_delta_by_layer",
        "summary_field": "raw_mean_delta_by_layer",
        "color": "#ff4d4d",
        "linestyle": "-",
    },
    {
        "name": "Resist samples",
        "kind": "group_raw",
        "group": "resist",
        "field": "raw_delta_by_layer",
        "summary_field": "raw_mean_delta_by_layer",
        "color": "#4daf4a",
        "linestyle": "-",
    },
    {
        "name": "Tuned lens (all samples)",
        "kind": "overall_tuned",
        "group": "all",
        "field": "tuned_delta_by_layer",
        "summary_field": "overall_tuned_mean_delta_by_layer",
        "color": "#3366ff",
        "linestyle": "--",
    },
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def summarize_record_curves(records: list[dict], curve_spec: dict) -> tuple[list[float], list[float]]:
    if curve_spec["kind"] == "group_raw":
        selected = [
            row
            for row in records
            if row.get("follow_label") == curve_spec["group"]
            and curve_spec["field"] in row
        ]
    else:
        selected = [row for row in records if curve_spec["field"] in row]

    if not selected:
        return [], []

    n_layers = len(selected[0][curve_spec["field"]])
    means = []
    bands = []
    for idx in range(n_layers):
        vals = [-float(row[curve_spec["field"]][idx]) for row in selected]
        means.append(mean(vals))
        # Use one standard error for the shaded region.
        bands.append(std(vals) / math.sqrt(max(1, len(vals))))
    return means, bands


def load_model_curves(model: dict) -> tuple[dict[str, dict], list[int], str]:
    if not model["summary"].exists():
        return {}, [], "missing summary.json"

    summary = load_json(model["summary"])
    layer_indices = [int(x) + 1 for x in summary["layer_indices"]]
    records = read_records(model["trajectories"]) if model["trajectories"].exists() else []

    curves = {}
    for spec in CURVES:
        values, bands = summarize_record_curves(records, spec) if records else ([], [])
        if not values:
            if spec["kind"] == "group_raw":
                group = summary.get("by_follow_label", {}).get(spec["group"], {})
                source_values = group.get(spec["summary_field"], [])
            else:
                source_values = summary.get(spec["summary_field"], [])
            values = [-float(x) for x in source_values]
            bands = [0.0 for _ in values]

        curves[spec["name"]] = {
            "values": values,
            "bands": bands,
            "source": str(model["trajectories"] if records else model["summary"]),
            "count": count_for_curve(records, summary, spec),
        }
    return curves, layer_indices, "ok"


def count_for_curve(records: list[dict], summary: dict, spec: dict) -> int:
    if records:
        if spec["kind"] == "group_raw":
            return sum(1 for row in records if row.get("follow_label") == spec["group"])
        return len(records)
    if spec["kind"] == "group_raw":
        return int(summary.get("by_follow_label", {}).get(spec["group"], {}).get("count", 0))
    return int(summary.get("n_records", 0))


def write_csv(rows: list[dict]) -> None:
    out_path = OUT_DIR / "tuned_lens_preference_trajectories.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "curve",
                "layer",
                "preference_margin",
                "band_low",
                "band_high",
                "n_samples",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_sources(rows: list[dict], missing: list[dict]) -> None:
    lines = [
        "# Tuned Lens Preference Trajectory Data Sources",
        "",
        "The plotted value is `log p(gold) - log p(wrong)`.",
        "The saved trajectory analyzer stores delta as `log p(wrong) - log p(gold)`, so the plotted margin is `-delta`.",
        "",
        "Curves:",
        "- Red: follow-conflict samples, `raw_delta_by_layer` grouped by `follow_label == follow_conflict`",
        "- Green: resist samples, `raw_delta_by_layer` grouped by `follow_label == resist`",
        "- Blue dashed: tuned lens over all samples, `tuned_delta_by_layer` over all records",
        "",
        "Shaded regions are one standard error computed from per-record trajectories when `trajectories.jsonl` is available.",
        "",
        "| Model | Curve | Samples | Source |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['curve']} | {row['n_samples']} | `{row['source']}` |"
        )
    if missing:
        lines.extend(["", "Missing trajectory data:", "", "| Model | Directory | Reason |", "|---|---|---|"])
        for item in missing:
            lines.append(f"| {item['model']} | `{item['root']}` | {item['reason']} |")
    (OUT_DIR / "tuned_lens_data_sources.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "legend.fontsize": 6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9), sharey=True)
    csv_rows = []
    source_rows = []
    missing = []

    for ax, model in zip(axes, MODELS):
        curves, layers, status = load_model_curves(model)
        ax.set_title(model["title"], fontweight="bold", pad=3)
        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.7, zorder=0)
        ax.axvline(20, color="0.25", linestyle=":", linewidth=0.7, zorder=0)
        ax.set_xlim(1, 36)
        ax.set_ylim(-3.0, 3.0)
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        ax.grid(True, axis="y", color="0.9", linewidth=0.5)

        if status != "ok":
            ax.text(
                0.5,
                0.5,
                "trajectory\nmissing",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.35",
            )
            missing.append({"model": model["title"], "root": model["root"], "reason": status})
            continue

        for spec in CURVES:
            item = curves[spec["name"]]
            values = item["values"]
            bands = item["bands"]
            n = min(len(layers), len(values))
            x = layers[:n]
            y = values[:n]
            b = bands[:n]
            low = [max(-3.0, yy - bb) for yy, bb in zip(y, b)]
            high = [min(3.0, yy + bb) for yy, bb in zip(y, b)]
            ax.fill_between(x, low, high, color=spec["color"], alpha=0.20, linewidth=0)
            ax.plot(
                x,
                y,
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=1.35,
                label=spec["name"],
            )

            source_rows.append(
                {
                    "model": model["title"],
                    "curve": spec["name"],
                    "n_samples": item["count"],
                    "source": item["source"],
                }
            )
            for layer, value, low_value, high_value in zip(x, y, low, high):
                csv_rows.append(
                    {
                        "model": model["title"],
                        "curve": spec["name"],
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{low_value:.8f}",
                        "band_high": f"{high_value:.8f}",
                        "n_samples": item["count"],
                        "source": item["source"],
                    }
                )

        ax.set_xlabel("Layer")
        if ax is axes[0]:
            ax.set_ylabel("Preference Margin\nlog p(gold) - log p(wrong)")
        else:
            ax.tick_params(labelleft=False)
        ax.legend(frameon=False, loc="upper right")

    write_csv(csv_rows)
    write_sources(source_rows, missing)

    fig.tight_layout(w_pad=0.8)
    png_path = OUT_DIR / "tuned_lens_preference_trajectories.png"
    pdf_path = OUT_DIR / "tuned_lens_preference_trajectories.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {OUT_DIR / 'tuned_lens_preference_trajectories.csv'}")
    print(f"Saved: {OUT_DIR / 'tuned_lens_data_sources.md'}")


if __name__ == "__main__":
    main()
