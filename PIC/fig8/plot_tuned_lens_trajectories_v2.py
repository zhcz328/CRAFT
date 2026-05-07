#!/usr/bin/env python3
"""
Plot tuned-lens preference trajectories.

All plotted curves are tuned-lens curves:
  - follow-conflict samples: grouped tuned-lens margin
  - resist samples: grouped tuned-lens margin
  - after ablation: tuned-lens margin on follow-conflict samples after ablation

The trajectory analyzer stores tuned_delta as:
  log p(wrong) - log p(gold)

The plotted preference margin follows the figure convention:
  Delta_l = log p(gold) - log p(wrong)

so grouped trajectory curves use -tuned_delta_by_layer.
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
        "trajectory_summary": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results/trajectory_before_question/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results/trajectory_before_question/trajectories.jsonl"
        ),
        "ablation_summary": None,
        "root": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
            "before_question/results"
        ),
    },
    {
        "title": "InternVL3.5-4B",
        "trajectory_summary": Path(
            "/root/autodl-tmp/tuned_lens/internvl35_4b/results/"
            "before_question/trajectory_train_idreg/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/internvl35_4b/results/"
            "before_question/trajectory_train_idreg/trajectories.jsonl"
        ),
        "ablation_summary": Path(
            "/root/logit_lens/Slake_vqa/text_conflict/internvl35_4b/"
            "result_before_question_slake/"
            "ablation_flip_ablate_ctx_only_val_follow_conflict/summary.json"
        ),
        "root": Path("/root/autodl-tmp/tuned_lens/internvl35_4b"),
    },
    {
        "title": "Hulu-med-4B",
        "trajectory_summary": Path(
            "/root/autodl-tmp/tuned_lens/hulumed_4b/results/"
            "trajectory_before_question_idreg/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/hulumed_4b/results/"
            "trajectory_before_question_idreg/trajectories.jsonl"
        ),
        "ablation_summary": Path(
            "/root/logit_lens/VQA_RAD/Hulu-med/analyze_probe_lens/"
            "ablation_flip_before_question_follow/summary.json"
        ),
        "root": Path("/root/autodl-tmp/tuned_lens/hulumed_4b"),
    },
    {
        "title": "Llama3.2-3B",
        "trajectory_summary": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory_before_question/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory_before_question/trajectories.jsonl"
        ),
        "ablation_summary": None,
        "root": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question"
        ),
    },
]

CURVES = [
    {
        "name": "Follow-conflict samples",
        "group": "follow_conflict",
        "color": "#ff4d4d",
        "linestyle": "-",
    },
    {
        "name": "Resist samples",
        "group": "resist",
        "color": "#4daf4a",
        "linestyle": "-",
    },
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def grouped_tuned_curve(
    records: list[dict],
    summary: dict,
    group_name: str,
) -> tuple[list[int], list[float], list[float], int, str]:
    layers = [int(layer) + 1 for layer in summary["layer_indices"]]
    selected = [
        row
        for row in records
        if row.get("follow_label") == group_name and "tuned_delta_by_layer" in row
    ]
    if selected:
        values = []
        bands = []
        for idx in range(len(layers)):
            vals = [-float(row["tuned_delta_by_layer"][idx]) for row in selected]
            values.append(mean(vals))
            bands.append(stderr(vals))
        return layers, values, bands, len(selected), "trajectories.jsonl"

    group = summary.get("by_follow_label", {}).get(group_name, {})
    source_values = group.get("tuned_mean_delta_by_layer", [])
    values = [-float(value) for value in source_values]
    return layers[: len(values)], values, [0.0 for _ in values], int(group.get("count", 0)), "summary.json"


def after_ablation_curve(path: Path | None) -> tuple[list[int], list[float], list[float], int, str]:
    if path is None or not path.exists():
        return [], [], [], 0, "missing"
    data = load_json(path)
    after = data.get("after_ablation", {})
    layers = [int(layer) + 1 for layer in after.get("lens_layer_indices", [])]
    gold = after.get("tuned_lens_mean_gold_logprob_by_layer", [])
    wrong = after.get("tuned_lens_mean_wrong_logprob_by_layer", [])
    n = min(len(layers), len(gold), len(wrong))
    values = [float(gold[idx]) - float(wrong[idx]) for idx in range(n)]
    count = int(after.get("tuned_lens_count", data.get("n_flip_correct_samples", 0)))
    return layers[:n], values, [0.0 for _ in range(n)], count, str(path)


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


def write_sources(source_rows: list[dict], missing_rows: list[dict]) -> None:
    lines = [
        "# Tuned Lens Preference Trajectory Data Sources",
        "",
        "All curves are tuned-lens curves.",
        "",
        "Plotted value:",
        "",
        "`Delta_l = log p(gold) - log p(wrong)`",
        "",
        "For grouped trajectory files, saved `tuned_delta_by_layer` is `log p(wrong) - log p(gold)`, so the plotted value is `-tuned_delta_by_layer`.",
        "",
        "Curves:",
        "- Red: follow-conflict samples, grouped by `follow_label == follow_conflict`, using `tuned_delta_by_layer`",
        "- Green: resist samples, grouped by `follow_label == resist`, using `tuned_delta_by_layer`",
        "- Blue dashed: follow-conflict samples after ablation, using `after_ablation.tuned_lens_mean_gold_logprob_by_layer - after_ablation.tuned_lens_mean_wrong_logprob_by_layer`",
        "",
        "Shaded regions for red/green are one standard error from per-record tuned-lens trajectories when `trajectories.jsonl` is available. The ablation summaries only store mean curves, so the blue dashed line has no error band.",
        "",
        "| Model | Curve | Samples | Source |",
        "|---|---|---:|---|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['model']} | {row['curve']} | {row['n_samples']} | `{row['source']}` |"
        )
    if missing_rows:
        lines.extend(["", "Missing data:", "", "| Model | Missing curve | Directory/Source | Reason |", "|---|---|---|---|"])
        for row in missing_rows:
            lines.append(
                f"| {row['model']} | {row['curve']} | `{row['source']}` | {row['reason']} |"
            )
    (OUT_DIR / "tuned_lens_data_sources.md").write_text("\n".join(lines), encoding="utf-8")


def plot_one_curve(ax, layers, values, bands, color, linestyle, label):
    n = min(len(layers), len(values), len(bands))
    x = layers[:n]
    y = values[:n]
    b = bands[:n]
    low = [max(-3.0, yy - bb) for yy, bb in zip(y, b)]
    high = [min(3.0, yy + bb) for yy, bb in zip(y, b)]
    if any(bb > 0 for bb in b):
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.45, label=label)
    return x, y, low, high


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

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.85), sharey=True)
    csv_rows = []
    source_rows = []
    missing_rows = []

    for ax, model in zip(axes, MODELS):
        ax.set_title(model["title"], fontweight="bold", pad=3)
        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.7, zorder=0)
        ax.axvline(20, color="0.25", linestyle=":", linewidth=0.7, zorder=0)
        ax.set_xlim(1, 36)
        ax.set_ylim(-3.0, 3.0)
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        ax.grid(True, axis="y", color="0.9", linewidth=0.5)
        ax.set_xlabel("Layer")

        if not model["trajectory_summary"].exists():
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
            missing_rows.append(
                {
                    "model": model["title"],
                    "curve": "follow/resist tuned-lens trajectories",
                    "source": str(model["root"]),
                    "reason": "trajectory summary/jsonl not found",
                }
            )
            continue

        summary = load_json(model["trajectory_summary"])
        records = read_jsonl(model["trajectories"]) if model["trajectories"].exists() else []

        for spec in CURVES:
            layers, values, bands, count, used = grouped_tuned_curve(records, summary, spec["group"])
            x, y, low, high = plot_one_curve(
                ax,
                layers,
                values,
                bands,
                spec["color"],
                spec["linestyle"],
                spec["name"],
            )
            source = model["trajectories"] if used == "trajectories.jsonl" else model["trajectory_summary"]
            source_rows.append(
                {
                    "model": model["title"],
                    "curve": spec["name"],
                    "n_samples": count,
                    "source": str(source),
                }
            )
            for layer, value, band_low, band_high in zip(x, y, low, high):
                csv_rows.append(
                    {
                        "model": model["title"],
                        "curve": spec["name"],
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{band_low:.8f}",
                        "band_high": f"{band_high:.8f}",
                        "n_samples": count,
                        "source": str(source),
                    }
                )

        layers, values, bands, count, source = after_ablation_curve(model["ablation_summary"])
        if values:
            x, y, low, high = plot_one_curve(
                ax,
                layers,
                values,
                bands,
                "#3366ff",
                "--",
                "After ablation",
            )
            source_rows.append(
                {
                    "model": model["title"],
                    "curve": "After ablation",
                    "n_samples": count,
                    "source": source,
                }
            )
            for layer, value, band_low, band_high in zip(x, y, low, high):
                csv_rows.append(
                    {
                        "model": model["title"],
                        "curve": "After ablation",
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{band_low:.8f}",
                        "band_high": f"{band_high:.8f}",
                        "n_samples": count,
                        "source": source,
                    }
                )
        else:
            missing_rows.append(
                {
                    "model": model["title"],
                    "curve": "After ablation",
                    "source": str(model["root"]),
                    "reason": "ablation tuned-lens summary not found in provided tuned_lens outputs",
                }
            )

        if ax is axes[0]:
            ax.set_ylabel("Preference Margin\nlog p(gold) - log p(wrong)")
        else:
            ax.tick_params(labelleft=False)
        ax.legend(frameon=False, loc="upper right")

    write_csv(csv_rows)
    write_sources(source_rows, missing_rows)
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
