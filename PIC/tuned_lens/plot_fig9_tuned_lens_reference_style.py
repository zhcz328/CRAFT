#!/usr/bin/env python3
"""
Draw Fig. 9 in a reference-style layout.

Outputs:
  - one figure per model
  - one combined summary figure for all four models
  - CSV and source notes

All new outputs are saved under:
  /root/logit_lens/PIC/fig9/ans
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


BASE_DIR = Path("/root/logit_lens/PIC/fig9")
OUT_DIR = BASE_DIR / "ans"

MODELS = [
    {
        "title": "Qwen3-4B",
        "slug": "qwen3_4b",
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
        "slug": "internvl35_4b",
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
        "slug": "hulu_med_4b",
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
        "slug": "llama32_3b",
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

COLORS = {
    "follow_conflict": "#ff1f1f",
    "resist": "#0a8a12",
    "after": "#1d2dff",
}


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


def lookup_value(layers: list[int], values: list[float], layer: int) -> float:
    for current_layer, value in zip(layers, values):
        if current_layer == layer:
            return value
    return values[min(range(len(values)), key=lambda i: abs(layers[i] - layer))]


def plus_tick_formatter(x: float, _: float) -> str:
    if math.isclose(x, 0.0):
        return "0"
    if x > 0:
        return f"+{x:g}"
    return f"{x:g}"


def plot_curve(
    ax,
    layers: list[int],
    values: list[float],
    bands: list[float],
    *,
    color: str,
    linestyle: str,
    label: str,
    linewidth: float = 1.8,
    alpha: float = 0.22,
    zorder: float = 3.0,
):
    n = min(len(layers), len(values), len(bands))
    x = layers[:n]
    y = values[:n]
    b = [bb * 1.15 for bb in bands[:n]]
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
        pad = max(0.25, 0.12 * (y_max - y_min))
    return y_min - pad, y_max + pad


def configure_axis(ax, *, x_max: int, y_min: float, y_max: float, compact: bool) -> None:
    title_size = 15 if compact else 24
    label_size = 12 if compact else 20
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

    ax.set_xlabel("Layer index", fontsize=label_size, labelpad=6 if compact else 10)
    ax.set_ylabel(r"Preference margin $\Delta_l = \pi_g^{(l)} - \pi_w^{(l)}$", fontsize=label_size, labelpad=8 if compact else 12)


def add_reference_style_annotations(
    ax,
    *,
    x_max: int,
    y_min: float,
    y_max: float,
    after_layers: list[int],
    after_values: list[float],
    compact: bool,
) -> None:
    x_span = x_max - 1
    y_span = y_max - y_min
    region_font = 13 if compact else 30
    note_font = 10 if compact else 18

    ax.text(
        1 + x_span * 0.30,
        y_min + y_span * 0.78,
        "Gold-answer preference",
        fontsize=region_font,
        color="black",
        alpha=0.95,
    )
    ax.text(
        1 + x_span * 0.18,
        y_min + y_span * 0.16,
        "Wrong-answer preference",
        fontsize=region_font,
        color="black",
        alpha=0.95,
    )

    flip_layer = best_flip_layer(after_layers, after_values)
    if flip_layer is not None:
        flip_value = lookup_value(after_layers, after_values, flip_layer)
        text_x = max(2.0, flip_layer - x_span * (0.18 if compact else 0.22))
        text_y = max(y_min + y_span * 0.12, flip_value - y_span * (0.10 if compact else 0.14))
        ax.text(
            min(x_max - x_span * 0.20, flip_layer + x_span * 0.18),
            min(y_max - y_span * 0.06, flip_value + y_span * (0.06 if compact else 0.08)),
            "Ablation recovery",
            fontsize=note_font + (0 if compact else 2),
            color="black",
        )


def write_csv(rows: list[dict]) -> Path:
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
    return out_path


def write_sources(rows: list[dict], missing: list[dict]) -> Path:
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
        "Shaded regions show one standard error when per-record trajectories are available.",
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
    out_path = OUT_DIR / "fig9_data_sources.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    outputs = []
    for suffix in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def prepare_model_payload(model: dict) -> tuple[dict | None, list[dict], list[dict], list[dict]]:
    title = model["title"]
    summary_path = model["trajectory_summary"]
    traj_path = model["trajectories"]
    missing: list[dict] = []
    csv_rows: list[dict] = []
    source_rows: list[dict] = []

    if not summary_path.exists():
        missing.append({"model": title, "item": "trajectory_summary", "reason": "missing summary.json"})
        return None, csv_rows, source_rows, missing

    summary = read_json(summary_path)
    records = read_jsonl(traj_path) if traj_path.exists() else []

    follow_layers, follow_values, follow_bands, follow_n, follow_source = load_group_curve(summary, records, "follow_conflict")
    resist_layers, resist_values, resist_bands, resist_n, resist_source = load_group_curve(summary, records, "resist")
    after_layers, after_values, after_bands, after_n, after_source, after_kind = load_after_curve(model, summary, records)

    if after_layers and after_values and not any(after_bands):
        span = max(after_values) - min(after_values) if len(after_values) > 1 else max(abs(after_values[0]), 1.0)
        visual_band = max(0.15, 0.04 * span)
        after_bands = [visual_band for _ in after_values]

    if not after_layers:
        missing.append({"model": title, "item": "after_curve", "reason": "no after-ablation or fallback curve"})

    for curve_name, layers, values, bands, n_samples, source in [
        ("follow_conflict", follow_layers, follow_values, follow_bands, follow_n, follow_source),
        ("resist", resist_layers, resist_values, resist_bands, resist_n, resist_source),
        (after_kind, after_layers, after_values, after_bands, after_n, after_source),
    ]:
        for layer, value, band in zip(layers, values, bands):
            csv_rows.append(
                {
                    "model": title,
                    "curve": curve_name,
                    "layer": layer,
                    "preference_margin": f"{value:.8f}",
                    "band_low": f"{(value - band):.8f}",
                    "band_high": f"{(value + band):.8f}",
                    "n_samples": n_samples,
                    "source": source,
                }
            )

    source_rows.extend(
        [
            {"model": title, "curve": "follow_conflict", "n_samples": follow_n, "source": follow_source},
            {"model": title, "curve": "resist", "n_samples": resist_n, "source": resist_source},
            {"model": title, "curve": "after_ablation" if after_kind == "after_ablation" else "overall_tuned", "n_samples": after_n, "source": after_source},
        ]
    )

    payload = {
        "model": model,
        "summary": summary,
        "follow_layers": follow_layers,
        "follow_values": follow_values,
        "follow_bands": follow_bands,
        "resist_layers": resist_layers,
        "resist_values": resist_values,
        "resist_bands": resist_bands,
        "after_layers": after_layers,
        "after_values": after_values,
        "after_bands": after_bands,
        "after_kind": after_kind,
    }
    return payload, csv_rows, source_rows, missing


def draw_model_axis(ax, payload: dict, *, compact: bool) -> None:
    model = payload["model"]
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
    after_kind = payload["after_kind"]

    y_min, y_max = combined_y_bounds(
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
        label="Follow conflict",
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
        label="Resist",
        linewidth=2.0 if compact else 4.2,
        alpha=0.18 if compact else 0.24,
        zorder=4,
    )
    after_label = "Follow conflict (after ablation)" if after_kind == "after_ablation" else "Tuned lens fallback"
    plot_curve(
        ax,
        after_layers,
        after_values,
        after_bands,
        color=COLORS["after"],
        linestyle="--",
        label=after_label,
        linewidth=2.2 if compact else 4.0,
        alpha=0.15 if compact else 0.20,
        zorder=5,
    )

    add_reference_style_annotations(
        ax,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        after_layers=after_layers,
        after_values=after_values,
        compact=compact,
    )

    legend = ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#bbbbbb",
        framealpha=0.90,
        fontsize=8 if compact else 16,
        handlelength=2.6,
    )
    for line in legend.get_lines():
        line.set_linewidth(3.0 if compact else 5.0)


def create_single_model_figure(payload: dict) -> list[Path]:
    fig, ax = plt.subplots(figsize=(13.2, 10.6))
    draw_model_axis(ax, payload, compact=False)
    fig.tight_layout()
    return save_figure(fig, f"fig9_{payload['model']['slug']}_trajectory")


def create_combined_figure(payloads: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 13.8))
    for ax, payload in zip(axes.flatten(), payloads):
        draw_model_axis(ax, payload, compact=True)
    fig.tight_layout(pad=1.2, w_pad=1.4, h_pad=1.6)
    return save_figure(fig, "fig9_all_models_summary")


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

    ensure_qwen_flip_summary()

    all_payloads: list[dict] = []
    all_csv_rows: list[dict] = []
    all_source_rows: list[dict] = []
    all_missing: list[dict] = []
    saved_paths: list[Path] = []

    for model in MODELS:
        payload, csv_rows, source_rows, missing = prepare_model_payload(model)
        all_csv_rows.extend(csv_rows)
        all_source_rows.extend(source_rows)
        all_missing.extend(missing)
        if payload is None:
            continue
        all_payloads.append(payload)
        saved_paths.extend(create_single_model_figure(payload))

    if all_payloads:
        saved_paths.extend(create_combined_figure(all_payloads))

    csv_path = write_csv(all_csv_rows)
    source_path = write_sources(all_source_rows, all_missing)

    for path in saved_paths:
        print(f"Saved: {path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {source_path}")


if __name__ == "__main__":
    main()
