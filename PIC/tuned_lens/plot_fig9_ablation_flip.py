#!/usr/bin/env python3
"""
Build Fig. 9 outputs for the Qwen ConflictMedQA ablation-flip analysis.

Behavior:
  - Reuse an existing summary.json if it already exists.
  - Otherwise run the analyzer logic to produce the flip summary first.
  - Write the figure outputs and provenance files into /root/logit_lens/PIC/fig9.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "fig9"
PROJECT_ROOT = ROOT / "conflictmedqa" / "Qwen3-4B_exp"
SUMMARY_DIR = PROJECT_ROOT / "analyze_probe_lens" / "ablation_flip_before_question_follow"
SUMMARY_PATH = SUMMARY_DIR / "summary.json"
ANALYSIS_BATCH_SIZE = 1
FORCE_CPU = os.environ.get("FIG9_FORCE_CPU", "0") == "1"

ABLATION_JSON = PROJECT_ROOT / "result" / "conflict_retest_ablated_heads_inf_before_question.jsonl"
BASELINE_JSON = PROJECT_ROOT / "result_all_positions" / "conflict_positions.jsonl"
PAIRS_JSON = PROJECT_ROOT / "result" / "kept_pairs_a12_b12.jsonl"
MODEL_PATH = Path("/root/autodl-tmp/qwen3-4B")
PROBE_DIR = Path(
    "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
    "before_question/results/follow_linear_before_question"
)
LENS_CKPT = Path(
    "/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/"
    "before_question/results/train_before_question/tuned_lens.pt"
)
HEAD_GROUPS_FALLBACK = PROJECT_ROOT / "result" / "headscan_rounds_v2" / "head_groups.json"


def add_project_paths() -> None:
    for path in [PROJECT_ROOT, PROJECT_ROOT / "probe", PROJECT_ROOT / "tuned_lens"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


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


def load_ablation_records(path: Path) -> tuple[dict, list[dict]]:
    meta = {}
    records = []
    for row in read_jsonl(path):
        if "_meta" in row:
            meta = row["_meta"]
            continue
        records.append(row)
    return meta, records


def build_baseline_map(path: Path) -> dict[tuple[int, str, str], dict]:
    baseline = {}
    for row in read_jsonl(path):
        key = (int(row["pair_id"]), str(row["side"]), str(row["position"]))
        baseline[key] = row
    return baseline


def build_pairs_map(path: Path) -> dict[int, dict]:
    pairs = {}
    for row in read_jsonl(path):
        pairs[int(row["pair_id"])] = row
    return pairs


def select_flip_samples(
    ablation_records: list[dict],
    baseline_map: dict[tuple[int, str, str], dict],
    pairs_map: dict[int, dict],
    position: str,
) -> list[dict]:
    selected = []
    for row in ablation_records:
        if str(row.get("position")) != position:
            continue
        key = (int(row["pair_id"]), str(row["side"]), str(row["position"]))
        baseline_row = baseline_map.get(key)
        pair_row = pairs_map.get(int(row["pair_id"]))
        if baseline_row is None or pair_row is None:
            continue

        baseline_conflict_pred = baseline_row["conflict"]["pred"]
        ablated_conflict_pred = row["conflict"]["pred"]
        gold = str(row["gold"])
        if baseline_conflict_pred == gold:
            continue
        if ablated_conflict_pred != gold:
            continue

        side = str(row["side"])
        base_prompt = pair_row[side]["prompt"]
        selected.append(
            {
                "pair_id": int(row["pair_id"]),
                "side": side,
                "position": position,
                "gold_answer": gold,
                "wrong_answer": str(row["conflict_label"]),
                "base_prompt": base_prompt,
                "baseline_conflict_pred": baseline_conflict_pred,
                "ablated_conflict_pred": ablated_conflict_pred,
            }
        )
    return selected


def ensure_summary() -> tuple[dict, Path]:
    if SUMMARY_PATH.exists():
        return read_json(SUMMARY_PATH), SUMMARY_PATH

    add_project_paths()
    import analyze_ablation_flip_with_probe_and_lens as analyzer

    meta, ablation_records = load_ablation_records(ABLATION_JSON)
    positions = meta.get("positions") or []
    if len(positions) == 1:
        position = str(positions[0])
    elif ablation_records:
        position = str(ablation_records[0]["position"])
    else:
        raise RuntimeError("Cannot infer analysis position from the ablation JSON.")

    baseline_map = build_baseline_map(BASELINE_JSON)
    pairs_map = build_pairs_map(PAIRS_JSON)
    selected_samples = select_flip_samples(ablation_records, baseline_map, pairs_map, position)
    if not selected_samples:
        raise RuntimeError("No wrong->correct flip samples were found.")

    tokenizer = analyzer.ensure_padding_token(
        analyzer.AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    )
    if FORCE_CPU:
        model_dtype = analyzer.torch.float32
        device_map = "cpu"
    else:
        model_dtype = analyzer.lens_modeling.resolve_dtype("bfloat16")
        device_map = "auto"
    model = analyzer.AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        dtype=model_dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    probe_specs = analyzer.load_probe_specs(PROBE_DIR)
    first_device = next(model.parameters()).device
    lens_ckpt, translators = analyzer.load_tuned_lens(LENS_CKPT, first_device)

    before_summary = analyzer.collect_condition_outputs(
        samples=selected_samples,
        tokenizer=tokenizer,
        model=model,
        probe_specs=probe_specs,
        score_type="logit",
        lens_ckpt=lens_ckpt,
        translators=translators,
        batch_size=ANALYSIS_BATCH_SIZE,
        enable_thinking=False,
    )

    head_groups_path = str(meta.get("head_groups_path") or "")
    resolved_head_groups = None
    if head_groups_path and Path(head_groups_path).exists():
        resolved_head_groups = Path(head_groups_path)
    elif HEAD_GROUPS_FALLBACK.exists():
        resolved_head_groups = HEAD_GROUPS_FALLBACK
    if resolved_head_groups is None:
        raise RuntimeError("Cannot resolve head_groups.json for the ablation analysis.")

    layer2heads, _ = analyzer.ablate_mod.load_ablation_heads(str(resolved_head_groups))
    handles = analyzer.ablate_mod.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_summary = analyzer.collect_condition_outputs(
            samples=selected_samples,
            tokenizer=tokenizer,
            model=model,
            probe_specs=probe_specs,
            score_type="logit",
            lens_ckpt=lens_ckpt,
            translators=translators,
            batch_size=ANALYSIS_BATCH_SIZE,
            enable_thinking=False,
        )
    finally:
        analyzer.ablate_mod.remove_hooks(handles)

    diff_summary = analyzer.summarize_ablation_effect(before_summary, after_summary)

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ablation_json": str(ABLATION_JSON),
        "baseline_json": str(BASELINE_JSON),
        "pairs": str(PAIRS_JSON),
        "head_groups": str(resolved_head_groups),
        "position": position,
        "score_type": "logit",
        "n_flip_correct_samples": len(selected_samples),
        "selected_samples": [
            {
                "pair_id": sample["pair_id"],
                "side": sample["side"],
                "gold_answer": sample["gold_answer"],
                "wrong_answer": sample["wrong_answer"],
                "baseline_conflict_pred": sample["baseline_conflict_pred"],
                "ablated_conflict_pred": sample["ablated_conflict_pred"],
            }
            for sample in selected_samples
        ],
        "before_ablation": before_summary,
        "after_ablation": after_summary,
        "ablation_effect_summary": diff_summary,
        "saved_plots": {
            "probe_before_after": str(SUMMARY_DIR / "probe_before_after_ablation_logit.png"),
            "tuned_lens_before": str(SUMMARY_DIR / "tuned_lens_before_ablation.png"),
            "tuned_lens_after": str(SUMMARY_DIR / "tuned_lens_after_ablation.png"),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, SUMMARY_PATH


def write_curve_csv(summary: dict) -> Path:
    csv_path = OUT_DIR / "fig9_ablation_flip_curve_data.csv"
    before = summary["before_ablation"]
    after = summary["after_ablation"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_type",
                "layer",
                "before",
                "after",
                "delta",
            ],
        )
        writer.writeheader()
        for idx, layer in enumerate(before["probe_layer_indices"]):
            before_value = float(before["probe_mean_scores_by_layer"][idx])
            after_value = float(after["probe_mean_scores_by_layer"][idx])
            writer.writerow(
                {
                    "layer_type": "probe",
                    "layer": int(layer),
                    "before": before_value,
                    "after": after_value,
                    "delta": after_value - before_value,
                }
            )
        for idx, layer in enumerate(before["lens_layer_indices"]):
            writer.writerow(
                {
                    "layer_type": "tuned_lens_gold",
                    "layer": int(layer),
                    "before": float(before["tuned_lens_mean_gold_logprob_by_layer"][idx]),
                    "after": float(after["tuned_lens_mean_gold_logprob_by_layer"][idx]),
                    "delta": float(after["tuned_lens_mean_gold_logprob_by_layer"][idx])
                    - float(before["tuned_lens_mean_gold_logprob_by_layer"][idx]),
                }
            )
            writer.writerow(
                {
                    "layer_type": "tuned_lens_wrong",
                    "layer": int(layer),
                    "before": float(before["tuned_lens_mean_wrong_logprob_by_layer"][idx]),
                    "after": float(after["tuned_lens_mean_wrong_logprob_by_layer"][idx]),
                    "delta": float(after["tuned_lens_mean_wrong_logprob_by_layer"][idx])
                    - float(before["tuned_lens_mean_wrong_logprob_by_layer"][idx]),
                }
            )
    return csv_path


def write_sources_md(summary_path: Path, curve_csv: Path) -> Path:
    md_path = OUT_DIR / "fig9_data_sources.md"
    lines = [
        "# Fig. 9 Data Sources",
        "",
        "This figure is built from the Qwen ConflictMedQA ablation-flip summary.",
        "",
        f"- Summary source: `{summary_path}`",
        f"- Curve data export: `{curve_csv}`",
        f"- Raw ablation input: `{ABLATION_JSON}`",
        f"- Baseline input: `{BASELINE_JSON}`",
        f"- Pair input: `{PAIRS_JSON}`",
        f"- Probe directory: `{PROBE_DIR}`",
        f"- Tuned-lens checkpoint: `{LENS_CKPT}`",
        f"- Head-groups fallback: `{HEAD_GROUPS_FALLBACK}`",
        "",
        "The source summary is used directly when it already exists. When it is missing, "
        "the script reconstructs it with the same analyzer logic used in the project code.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def plot_probe_before_after(summary: dict) -> tuple[Path, Path]:
    before = summary["before_ablation"]
    after = summary["after_ablation"]
    png_path = OUT_DIR / "fig9_probe_before_after_ablation_logit.png"
    pdf_path = OUT_DIR / "fig9_probe_before_after_ablation_logit.pdf"
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        before["probe_layer_indices"],
        before["probe_mean_scores_by_layer"],
        marker="o",
        linewidth=2.2,
        color="#d04f3e",
        label="before ablation",
    )
    ax.plot(
        after["probe_layer_indices"],
        after["probe_mean_scores_by_layer"],
        marker="o",
        linewidth=2.2,
        color="#2f6db3",
        label="after ablation",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean probe logit")
    ax.set_title(f"Probe before vs after ablation\nwrong->correct samples (n={before['probe_count']})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_tuned_lens(summary: dict, which: str) -> tuple[Path, Path]:
    payload = summary[which]
    png_path = OUT_DIR / f"fig9_tuned_lens_{which.split('_')[0]}_ablation.png"
    pdf_path = OUT_DIR / f"fig9_tuned_lens_{which.split('_')[0]}_ablation.pdf"
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(
        payload["lens_layer_indices"],
        payload["tuned_lens_mean_gold_logprob_by_layer"],
        marker="o",
        linewidth=2.2,
        color="#2f6db3",
        label="gold",
    )
    ax.plot(
        payload["lens_layer_indices"],
        payload["tuned_lens_mean_wrong_logprob_by_layer"],
        marker="o",
        linewidth=2.2,
        color="#d04f3e",
        label="wrong",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean tuned-lens logprob")
    ax.set_title(
        f"Tuned lens {which.split('_')[0]} ablation\nwrong->correct samples (n={payload['tuned_lens_count']})"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def write_figure_bundle(summary: dict) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    out_paths.extend(plot_probe_before_after(summary))
    out_paths.extend(plot_tuned_lens(summary, "before_ablation"))
    out_paths.extend(plot_tuned_lens(summary, "after_ablation"))
    return out_paths


def write_overview(summary: dict) -> Path:
    before = summary["before_ablation"]
    after = summary["after_ablation"]
    overview_png = OUT_DIR / "fig9_ablation_flip_overview.png"
    overview_pdf = OUT_DIR / "fig9_ablation_flip_overview.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    ax.plot(
        before["probe_layer_indices"],
        before["probe_mean_scores_by_layer"],
        marker="o",
        linewidth=2.0,
        color="#d04f3e",
        label="before ablation",
    )
    ax.plot(
        after["probe_layer_indices"],
        after["probe_mean_scores_by_layer"],
        marker="o",
        linewidth=2.0,
        color="#2f6db3",
        label="after ablation",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    ax.set_title("Probe")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean probe logit")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        before["lens_layer_indices"],
        before["tuned_lens_mean_gold_logprob_by_layer"],
        marker="o",
        linewidth=2.0,
        color="#2f6db3",
        label="gold before",
    )
    ax.plot(
        before["lens_layer_indices"],
        before["tuned_lens_mean_wrong_logprob_by_layer"],
        marker="o",
        linewidth=2.0,
        color="#d04f3e",
        label="wrong before",
    )
    ax.plot(
        after["lens_layer_indices"],
        after["tuned_lens_mean_gold_logprob_by_layer"],
        marker="o",
        linewidth=1.8,
        color="#2f6db3",
        linestyle="--",
        label="gold after",
    )
    ax.plot(
        after["lens_layer_indices"],
        after["tuned_lens_mean_wrong_logprob_by_layer"],
        marker="o",
        linewidth=1.8,
        color="#d04f3e",
        linestyle="--",
        label="wrong after",
    )
    ax.set_title("Tuned lens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean tuned-lens logprob")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Qwen3-4B ConflictMedQA ablation flip\n"
        f"wrong->correct samples (n={before['probe_count']})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(overview_png, dpi=220, bbox_inches="tight")
    fig.savefig(overview_pdf, bbox_inches="tight")
    plt.close(fig)
    return overview_png


def main() -> None:
    summary, summary_path = ensure_summary()
    curve_csv = write_curve_csv(summary)
    write_sources_md(summary_path, curve_csv)
    write_figure_bundle(summary)
    write_overview(summary)

    print(f"saved_summary={summary_path}")
    print(f"saved_curve_csv={curve_csv}")
    print(f"saved_outputs_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
