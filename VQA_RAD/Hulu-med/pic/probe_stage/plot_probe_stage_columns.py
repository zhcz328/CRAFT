import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_stage_curve(summary, epoch, metric_key):
    xs = []
    ys = []
    epoch_key = str(epoch)
    for row in summary["metrics_by_layer"]:
        xs.append(int(row["layer_idx"]))
        stage_metrics = row.get("stage_val_metrics", {})
        if epoch_key in stage_metrics:
            ys.append(float(stage_metrics[epoch_key][metric_key]))
        else:
            ys.append(float(row["val_metrics"][metric_key]))
    return xs, ys


def mean(values):
    return sum(values) / max(1, len(values))


def metric_name(metric_key):
    return "AUROC" if metric_key == "auroc" else "Macro-F1"


def draw_overlay_axis(ax, summary, stage_epochs, metric_key, title):
    stage_colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    legend_labels = []
    bests = []
    means = []

    for idx, epoch in enumerate(stage_epochs):
        x_vals, y_vals = extract_stage_curve(summary, epoch, metric_key)
        color = stage_colors[idx % len(stage_colors)]
        ax.plot(x_vals, y_vals, color=color, linewidth=2.0, alpha=0.95)
        legend_labels.append(f"Stage {idx + 1} (epoch {epoch})")
        bests.append(max(y_vals))
        means.append(mean(y_vals))

    trend_text = ""
    if len(bests) >= 2:
        trend_text = f"\nTrend: best {bests[0]:.3f}→{bests[-1]:.3f}, mean {means[0]:.3f}→{means[-1]:.3f}"

    ax.set_title(f"{title} ({metric_name(metric_key)}){trend_text}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_name(metric_key))
    ax.grid(alpha=0.25)
    ax.legend(legend_labels, loc="best", fontsize=9, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser(description="Plot 3-stage probe metric curves in 2 columns.")
    ap.add_argument(
        "--conflict_summary",
        default="/root/autodl-tmp/Hulumed/probe_vqa_rad/results/conflict_linear_before_question_ok/summary.json",
    )
    ap.add_argument(
        "--follow_summary",
        default="/root/autodl-tmp/Hulumed/probe_vqa_rad/results/follow_linear_before_question_ok/summary.json",
    )
    ap.add_argument(
        "--out_dir",
        default="/root/logit_lens/VQA_RAD/Hulu-med/pic/probe_stage",
    )
    args = ap.parse_args()

    conflict_summary = load_summary(Path(args.conflict_summary))
    follow_summary = load_summary(Path(args.follow_summary))

    conflict_stages = [int(x) for x in conflict_summary.get("stage_epochs", [])]
    follow_stages = [int(x) for x in follow_summary.get("stage_epochs", [])]
    if not conflict_stages or not follow_stages:
        raise ValueError("Missing stage_epochs in one or both summaries.")

    # Align by stage index (stage1/stage2/stage3), not by identical epoch numbers.
    stage_count = min(len(conflict_stages), len(follow_stages))
    conflict_stages = conflict_stages[:stage_count]
    follow_stages = follow_stages[:stage_count]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

    draw_overlay_axis(
        axes[0],
        conflict_summary,
        conflict_stages,
        metric_key="auroc",
        title="Probe A (conflict)",
    )
    draw_overlay_axis(
        axes[1],
        follow_summary,
        follow_stages,
        metric_key="macro_f1",
        title="Probe B (follow_conflict)",
    )

    fig.suptitle("Probe Stage Trends (3 stages overlaid per probe)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = out_dir / "probe_stage_overlay_1x2.png"
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    output_files = [str(out_path)]

    summary = {
        "conflict_summary": str(Path(args.conflict_summary)),
        "follow_summary": str(Path(args.follow_summary)),
        "aligned_by": "stage_index",
        "stage_count": stage_count,
        "conflict_stage_epochs": conflict_stages,
        "follow_stage_epochs": follow_stages,
        "outputs": output_files,
    }
    summary_path = out_dir / "probe_stage_plot_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved_summary={summary_path}")
    for p in output_files:
        print(f"saved_plot={p}")


if __name__ == "__main__":
    main()
