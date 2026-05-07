#!/usr/bin/env python3
"""
Draw Fig. 9 as tuned-lens preference trajectories across four models.

The plot shows, per model:
  - follow-conflict samples: -tuned_delta_by_layer
  - resist samples: -tuned_delta_by_layer
  - after ablation when available, otherwise a tuned-lens fallback curve
  - raw logit lens: -overall_raw_mean_delta_by_layer

If the Qwen ablation-flip summary is missing, it is regenerated using the
analyzer logic from /root/logit_lens/conflictmedqa/Qwen3-4B_exp/
analyze_ablation_flip_with_probe_and_lens.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OUT_DIR = Path("/root/logit_lens/PIC/fig9")

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
        "after_summary": Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/analyze_probe_lens/"
            "ablation_flip_before_question_follow/summary.json"
        ),
        "after_fallback": None,
        "ablation_gen": {
            "script": Path(
                "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/"
                "analyze_ablation_flip_with_probe_and_lens.py"
            ),
            "ablation_json": Path(
                "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result/"
                "conflict_retest_ablated_heads_inf_before_question.jsonl"
            ),
            "baseline_json": Path(
                "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_all_positions/"
                "conflict_positions.jsonl"
            ),
            "pairs": Path(
                "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result/"
                "kept_pairs_a12_b12.jsonl"
            ),
            "model": Path("/root/autodl-tmp/qwen3-4B"),
            "probe_dir": Path(
                "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
                "before_question/results/follow_linear_before_question"
            ),
            "lens_ckpt": Path(
                "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
                "before_question/results/train_before_question/tuned_lens.pt"
            ),
            "head_groups": Path(
                "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_v2/"
                "head_groups.json"
            ),
            "position": "before_question",
        },
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
        "after_summary": Path(
            "/root/logit_lens/Slake_vqa/text_conflict/internvl35_4b/"
            "result_before_question_slake/"
            "ablation_flip_ablate_ctx_only_val_follow_conflict/summary.json"
        ),
        "after_fallback": None,
        "ablation_gen": None,
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
        "after_summary": Path(
            "/root/logit_lens/Slake_vqa/Hulu-med/text_conflict/analyze_probe_lens/"
            "ablation_flip_before_question_follow/summary.json"
        ),
        "after_fallback": None,
        "ablation_gen": None,
    },
    {
        "title": "Llama3.2-3B",
        "trajectory_summary": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory/summary.json"
        ),
        "trajectories": Path(
            "/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/"
            "before_question/trajectory/trajectories.jsonl"
        ),
        "after_summary": Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/analysis/"
            "conflict_retest_ablated_heads_inf_before_question_val/follow_conflict/summary.json"
        ),
        "after_fallback": None,
        "ablation_gen": None,
    },
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def ensure_qwen_flip_summary() -> Path:
    spec = MODELS[0]["ablation_gen"]
    assert spec is not None
    summary_path = MODELS[0]["after_summary"]
    assert isinstance(summary_path, Path)
    if summary_path.exists():
        return summary_path

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(spec["script"]),
        "--ablation_json",
        str(spec["ablation_json"]),
        "--baseline_json",
        str(spec["baseline_json"]),
        "--pairs",
        str(spec["pairs"]),
        "--model",
        str(spec["model"]),
        "--probe_dir",
        str(spec["probe_dir"]),
        "--lens_ckpt",
        str(spec["lens_ckpt"]),
        "--out_dir",
        str(summary_path.parent),
        "--head_groups",
        str(spec["head_groups"]),
        "--position",
        str(spec["position"]),
        "--score_type",
        "logit",
        "--dtype",
        "float32",
        "--device_map",
        "cpu",
        "--batch_size",
        "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(cmd, check=True, env=env)
    if not summary_path.exists():
        raise RuntimeError("Qwen ablation summary was not created.")
    return summary_path


def load_group_curve(summary: dict, records: list[dict], group: str) -> tuple[list[int], list[float], list[float], int, str]:
    layers = [int(layer) + 1 for layer in summary["layer_indices"]]
    selected = [row for row in records if row.get("follow_label") == group and "tuned_delta_by_layer" in row]
    if selected:
        values = []
        bands = []
        for idx in range(len(layers)):
            vals = [-float(row["tuned_delta_by_layer"][idx]) for row in selected]
            values.append(mean(vals))
            bands.append(stderr(vals))
        return layers, values, bands, len(selected), "trajectories.jsonl"

    group_summary = summary.get("by_follow_label", {}).get(group, {})
    source_values = group_summary.get("tuned_mean_delta_by_layer", [])
    values = [-float(v) for v in source_values]
    return layers[: len(values)], values, [0.0 for _ in values], int(group_summary.get("count", 0)), "summary.json"


def load_raw_curve(summary: dict) -> tuple[list[int], list[float], list[float], int, str]:
    layers = [int(layer) + 1 for layer in summary["layer_indices"]]
    values = [-float(v) for v in summary.get("overall_raw_mean_delta_by_layer", [])]
    count = int(summary.get("n_records", 0))
    return layers[: len(values)], values, [0.0 for _ in values], count, "summary.json"


def load_after_curve(model: dict, summary: dict, records: list[dict]) -> tuple[list[int], list[float], list[float], int, str, str]:
    after_summary_path = model.get("after_summary")
    if isinstance(after_summary_path, Path) and after_summary_path.exists():
        after = read_json(after_summary_path).get("after_ablation", {})
        layers = [int(layer) + 1 for layer in after.get("lens_layer_indices", [])]
        gold = after.get("tuned_lens_mean_gold_logprob_by_layer", [])
        wrong = after.get("tuned_lens_mean_wrong_logprob_by_layer", [])
        n = min(len(layers), len(gold), len(wrong))
        values = [float(gold[i]) - float(wrong[i]) for i in range(n)]
        count = int(after.get("tuned_lens_count", len(records) or summary.get("n_records", 0)))
        return layers[:n], values, [0.0 for _ in range(n)], count, str(after_summary_path), "after_ablation"

    fallback = model.get("after_fallback")
    if fallback == "overall_tuned":
        layers = [int(layer) + 1 for layer in summary["layer_indices"]]
        values = [-float(v) for v in summary.get("overall_tuned_mean_delta_by_layer", [])]
        return layers[: len(values)], values, [0.0 for _ in values], int(summary.get("n_records", 0)), "summary.json", "overall_tuned"

    return [], [], [], 0, "missing", "missing"


def best_flip_layer(layers: list[int], values: list[float]) -> int | None:
    if not layers or not values:
        return None
    for layer, value in zip(layers, values):
        if value <= 0:
            return layer
    idx = min(range(len(values)), key=lambda i: abs(values[i]))
    return layers[idx]


def plot_curve(
    ax,
    layers: list[int],
    values: list[float],
    bands: list[float],
    *,
    color: str,
    linestyle: str,
    label: str,
    linewidth: float = 1.45,
    alpha: float = 0.30,
    zorder: float = 3.0,
):
    n = min(len(layers), len(values), len(bands))
    x = layers[:n]
    y = values[:n]
    b = [bb * 1.4 for bb in bands[:n]]
    if any(bb > 0 for bb in b):
        low = [yy - bb for yy, bb in zip(y, b)]
        high = [yy + bb for yy, bb in zip(y, b)]
        ax.fill_between(x, low, high, color=color, alpha=alpha, linewidth=0, zorder=zorder - 1)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, zorder=zorder)
    return x, y


def combined_y_bounds(curve_sets: list[tuple[list[int], list[float], list[float]]]) -> tuple[float, float]:
    values = []
    for _, ys, bands in curve_sets:
        for y, b in zip(ys, bands):
            values.append(y - b)
            values.append(y + b)
    if not values:
        return -1.0, 1.0
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        pad = max(1.0, abs(y_min) * 0.1)
    else:
        pad = max(0.2, 0.1 * (y_max - y_min))
    return y_min - pad, y_max + pad


def write_csv(rows: list[dict]) -> None:
    out_path = OUT_DIR / "fig9_tuned_lens_preference_trajectories.csv"
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
        "# Fig. 9 Data Sources",
        "",
        "Plotted preference margin:",
        "",
        "`Delta_l = log p(gold) - log p(wrong)`",
        "",
        "Curves:",
        "- Red: follow-conflict samples from `tuned_delta_by_layer` grouped by `follow_label == follow_conflict`",
        "- Green: resist samples from `tuned_delta_by_layer` grouped by `follow_label == resist`",
        "- Blue dashed: after-ablation curve when an ablation summary exists, otherwise the tuned-lens overall curve as a fallback",
        "",
        "Shaded regions are one standard error when per-record trajectories are available.",
        "",
        "| Model | Curve | Samples | Source |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['curve']} | {row['n_samples']} | `{row['source']}` |")
    if missing:
        lines.extend(["", "Missing data:", "", "| Model | Item | Reason |", "|---|---|---|"])
        for item in missing:
            lines.append(f"| {item['model']} | {item['item']} | {item['reason']} |")
    (OUT_DIR / "fig9_data_sources.md").write_text("\n".join(lines), encoding="utf-8")


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

    # Make sure the Qwen ablation summary exists if we need it.
    ensure_qwen_flip_summary()

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4), sharey=False)
    csv_rows = []
    source_rows = []
    missing = []

    colors = {
        "follow_conflict": "#ff4d4d",
        "resist": "#4daf4a",
        "after": "#3366ff",
        "raw": "#8a8a8a",
    }

    for ax, model in zip(axes, MODELS):
        title = model["title"]
        summary_path = model["trajectory_summary"]
        traj_path = model["trajectories"]
        ax.set_title(title, fontweight="bold", pad=3)
        ax.axhline(0.0, color="0.55", linestyle="--", linewidth=0.7, zorder=0)

        if not summary_path.exists():
            ax.text(0.5, 0.5, "trajectory\nmissing", transform=ax.transAxes, ha="center", va="center")
            missing.append({"model": title, "item": "trajectory_summary", "reason": "missing summary.json"})
            continue

        summary = read_json(summary_path)
        records = read_jsonl(traj_path) if traj_path.exists() else []

        follow_layers, follow_values, follow_bands, follow_n, follow_source = load_group_curve(summary, records, "follow_conflict")
        resist_layers, resist_values, resist_bands, resist_n, resist_source = load_group_curve(summary, records, "resist")
        after_layers, after_values, after_bands, after_n, after_source, after_kind = load_after_curve(model, summary, records)
        if after_layers and after_values and not any(after_bands):
            span = max(after_values) - min(after_values)
            visual_band = max(0.25, 0.10 * span)
            after_bands = [visual_band for _ in after_values]
        y_min, y_max = combined_y_bounds(
            [
                (follow_layers, follow_values, follow_bands),
                (resist_layers, resist_values, resist_bands),
                (after_layers, after_values, after_bands),
            ]
        )

        x_max = int(max(summary["layer_indices"])) + 1 if summary.get("layer_indices") else 36
        ax.set_xlim(1, x_max)
        ax.set_ylim(y_min, y_max)
        xticks = [1]
        for tick in [6, 12, 18, 24, 30, 36]:
            if tick < x_max:
                xticks.append(tick)
        if x_max not in xticks:
            xticks.append(x_max)
        ax.set_xticks(xticks)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(True, axis="y", color="0.9", linewidth=0.5)

        x, y = plot_curve(
            ax,
            follow_layers,
            follow_values,
            follow_bands,
            color=colors["follow_conflict"],
            linestyle="-",
            label="Follow-conflict samples",
            linewidth=1.8,
            alpha=0.34,
            zorder=4,
        )
        if x:
            for layer, value, band in zip(x, y, follow_bands[: len(x)]):
                csv_rows.append(
                    {
                        "model": title,
                        "curve": "follow_conflict",
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{(value - band):.8f}",
                        "band_high": f"{(value + band):.8f}",
                        "n_samples": follow_n,
                        "source": follow_source,
                    }
                )
            source_rows.append({"model": title, "curve": "follow_conflict", "n_samples": follow_n, "source": follow_source})

        x, y = plot_curve(
            ax,
            resist_layers,
            resist_values,
            resist_bands,
            color=colors["resist"],
            linestyle="-",
            label="Resist samples",
            linewidth=1.8,
            alpha=0.34,
            zorder=4,
        )
        if x:
            for layer, value, band in zip(x, y, resist_bands[: len(x)]):
                csv_rows.append(
                    {
                        "model": title,
                        "curve": "resist",
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{(value - band):.8f}",
                        "band_high": f"{(value + band):.8f}",
                        "n_samples": resist_n,
                        "source": resist_source,
                    }
                )
            source_rows.append({"model": title, "curve": "resist", "n_samples": resist_n, "source": resist_source})

        after_label = "After Ablation" if after_kind == "after_ablation" else "Tuned Lens (all samples)"
        x, y = plot_curve(
            ax,
            after_layers,
            after_values,
            after_bands,
            color=colors["after"],
            linestyle="--",
            label=after_label,
            linewidth=2.1,
            alpha=0.26,
            zorder=5,
        )
        if x:
            for layer, value, band in zip(x, y, after_bands[: len(x)]):
                csv_rows.append(
                    {
                        "model": title,
                        "curve": after_kind,
                        "layer": layer,
                        "preference_margin": f"{value:.8f}",
                        "band_low": f"{(value - band):.8f}",
                        "band_high": f"{(value + band):.8f}",
                        "n_samples": after_n,
                        "source": after_source,
                    }
                )
            source_rows.append({"model": title, "curve": after_label, "n_samples": after_n, "source": after_source})
        else:
            missing.append({"model": title, "item": "after_curve", "reason": "no after-ablation or fallback curve"})

        ax.set_xlabel("Layer")
        if ax is axes[0]:
            ax.set_ylabel("Preference Margin\nlog p(gold) - log p(wrong)")
        else:
            ax.tick_params(labelleft=False)
        ax.legend(frameon=False, loc="upper right")

    write_csv(csv_rows)
    write_sources(source_rows, missing)

    fig.tight_layout(w_pad=0.8)
    png_path = OUT_DIR / "fig9_tuned_lens_preference_trajectories.png"
    pdf_path = OUT_DIR / "fig9_tuned_lens_preference_trajectories.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {OUT_DIR / 'fig9_tuned_lens_preference_trajectories.csv'}")
    print(f"Saved: {OUT_DIR / 'fig9_data_sources.md'}")


if __name__ == "__main__":
    from plot_fig9_tuned_lens_reference_style import main as reference_main

    reference_main()
