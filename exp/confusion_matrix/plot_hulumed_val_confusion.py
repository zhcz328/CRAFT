from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot two binary yes/no confusion matrices with percentages only."
    )
    parser.add_argument(
        "--input-json",
        default=(
            "/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
            "ablate_selected_heads_all_hulumed4b_val.json"
        ),
        help="Path to the ablation val result json.",
    )
    parser.add_argument(
        "--pred-scope",
        choices=["ctx", "nc"],
        default="ctx",
        help="Use context-conditioned predictions (`ctx`) or no-context predictions (`nc`).",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/logit_lens/exp/confusion_matrix",
        help="Directory to save confusion matrix figures.",
    )
    parser.add_argument(
        "--output-prefix",
        default="hulumed4b_val_ctx_binary",
        help="Filename prefix for saved figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG dpi.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["yes_no", "unknown_nonunknown"],
        default="yes_no",
        help="Binary label mapping to use for the confusion matrix.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_predicted_answer(record: dict, pred_token: str) -> str:
    if pred_token == "gold":
        return str(record["gold"])
    if pred_token == "wrong":
        return str(record["wrong"])
    return str(pred_token)


def map_binary_label(answer: str, label_mode: str) -> str | None:
    answer = str(answer).strip().lower()
    if label_mode == "yes_no":
        return answer if answer in {"yes", "no"} else None
    if label_mode == "unknown_nonunknown":
        return "unknown" if answer == "unknown" else "nonunknown"
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def build_binary_confusion_matrix(
    records: list[dict],
    pred_key: str,
    label_mode: str,
) -> tuple[np.ndarray, int]:
    if label_mode == "yes_no":
        labels = {"yes": 0, "no": 1}
    else:
        labels = {"unknown": 0, "nonunknown": 1}
    matrix = np.zeros((2, 2), dtype=np.int32)
    kept = 0

    for record in records:
        gold = map_binary_label(record["gold"], label_mode)
        pred = map_binary_label(resolve_predicted_answer(record, str(record[pred_key])), label_mode)
        if gold is None or pred is None:
            continue
        matrix[labels[gold], labels[pred]] += 1
        kept += 1

    return matrix, kept


def draw_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.1), constrained_layout=True)
    percent_matrix = np.zeros_like(matrix, dtype=np.float64)
    for i in range(matrix.shape[0]):
        row_total = int(matrix[i].sum())
        if row_total:
            percent_matrix[i] = (matrix[i] / row_total) * 100.0

    im = ax.imshow(
        percent_matrix,
        cmap="Blues",
        interpolation="nearest",
        vmin=0.0,
        vmax=100.0,
    )

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(["", ""], fontsize=10)
    ax.set_yticklabels(["", ""], fontsize=10)

    threshold = 50.0
    for i in range(2):
        row_total = int(matrix[i].sum())
        for j in range(2):
            value = int(matrix[i, j])
            percent = (100.0 * value / row_total) if row_total else 0.0
            color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                f"{percent:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=14,
            )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return im.cmap, im.norm


def draw_colorbar_legend(cmap, norm, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(1.4, 4.8), constrained_layout=True)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical",
    )
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontsize(tick_label.get_fontsize() * 2.25)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    payload = load_json(Path(args.input_json))
    records = payload["records"]

    base_key = f"base_{args.pred_scope}_pred"
    ab_key = f"ab_{args.pred_scope}_pred"

    before_matrix, _ = build_binary_confusion_matrix(records, base_key, args.label_mode)
    after_matrix, _ = build_binary_confusion_matrix(records, ab_key, args.label_mode)

    output_dir = Path(args.output_dir)
    before_path = output_dir / f"{args.output_prefix}_before_ablation.png"
    after_path = output_dir / f"{args.output_prefix}_after_ablation.png"
    legend_path = output_dir / f"{args.output_prefix}_colorbar_legend.png"

    cmap, norm = draw_confusion_matrix(before_matrix, before_path, args.dpi)
    draw_confusion_matrix(after_matrix, after_path, args.dpi)
    draw_colorbar_legend(cmap, norm, legend_path, args.dpi)

    print(f"Saved: {before_path}")
    print(f"Saved: {after_path}")
    print(f"Saved: {legend_path}")


if __name__ == "__main__":
    main()
