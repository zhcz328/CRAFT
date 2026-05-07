import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


BASE_DIR = Path("/root/logit_lens/exp/Head overlap Venn")

MODEL_SPECS = [
    {
        "model": "hulumed4b",
        "left_label": "Image conflict",
        "right_label": "Text conflict",
        "left_path": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
            "result_image_conflict_slake/selected_heads_core_layers.json"
        ),
        "right_path": Path(
            "/root/logit_lens/Slake_vqa/Hulu-med/text_conflict/"
            "result_slake_hulumed4b_before_question/"
            "headscan_slake_mm_accel_hulumed4b_96g/"
            "selected_heads_stable_hulumed4b.json"
        ),
    },
    {
        "model": "internvl35_4b",
        "left_label": "Image conflict",
        "right_label": "Text conflict",
        "left_path": Path(
            "/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/"
            "result_image_conflict_slake/selected_heads_core_layers.json"
        ),
        "right_path": Path(
            "/root/logit_lens/Slake_vqa/text_conflict/internvl35_4b/"
            "result_before_question_slake/selected_heads_merged_unique_layers.json"
        ),
    },
]


def load_selected_heads(path: Path) -> set[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {(item["layer"], item["head"]) for item in data["selected"]}


def compute_summary(left_heads: set[tuple[int, int]], right_heads: set[tuple[int, int]]) -> dict:
    overlap = left_heads & right_heads
    union = left_heads | right_heads
    left_only = left_heads - right_heads
    right_only = right_heads - left_heads

    return {
        "left_count": len(left_heads),
        "right_count": len(right_heads),
        "overlap_count": len(overlap),
        "union_count": len(union),
        "jaccard_index": len(overlap) / len(union) if union else 0.0,
        "overlap_coefficient": len(overlap) / min(len(left_heads), len(right_heads))
        if left_heads and right_heads
        else 0.0,
        "left_only": sorted(left_only),
        "right_only": sorted(right_only),
        "overlap": sorted(overlap),
    }


def draw_venn(model: str, left_label: str, right_label: str, summary: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfcfc")

    left_center = (0.42, 0.55)
    right_center = (0.72, 0.55)
    radius = 0.22

    left_circle = Circle(left_center, radius, color="#4C78A8", alpha=0.45, lw=2)
    right_circle = Circle(right_center, radius, color="#E45756", alpha=0.45, lw=2)
    ax.add_patch(left_circle)
    ax.add_patch(right_circle)

    ax.text(left_center[0] - 0.09, left_center[1], str(len(summary["left_only"])),
            ha="center", va="center", fontsize=22, fontweight="bold", color="#16324F")
    ax.text((left_center[0] + right_center[0]) / 2, left_center[1], str(summary["overlap_count"]),
            ha="center", va="center", fontsize=22, fontweight="bold", color="#5B3A29")
    ax.text(right_center[0] + 0.09, right_center[1], str(len(summary["right_only"])),
            ha="center", va="center", fontsize=22, fontweight="bold", color="#5A1F20")

    ax.text(left_center[0], 0.24, left_label, ha="center", va="center", fontsize=13, color="#16324F")
    ax.text(right_center[0], 0.24, right_label, ha="center", va="center", fontsize=13, color="#5A1F20")

    stats_text = (
        f"Image heads: {summary['left_count']}\n"
        f"Text heads: {summary['right_count']}\n"
        f"Overlap: {summary['overlap_count']}\n"
        f"Union: {summary['union_count']}\n"
        f"Jaccard: {summary['jaccard_index']:.3f}\n"
        f"Overlap coeff.: {summary['overlap_coefficient']:.3f}"
    )
    ax.text(
        0.06,
        0.92,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#ffffff", "ec": "#d9d9d9"},
    )

    ax.set_title(f"{model} selected_heads overlap", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlim(0.08, 1.0)
    ax.set_ylim(0.12, 0.98)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_markdown(results: list[dict], out_path: Path) -> None:
    lines = ["# Head overlap summary", ""]
    for result in results:
        summary = result["summary"]
        lines.extend(
            [
                f"## {result['model']}",
                "",
                f"- Image selected heads: {summary['left_count']}",
                f"- Text selected heads: {summary['right_count']}",
                f"- Overlap heads: {summary['overlap_count']}",
                f"- Union heads: {summary['union_count']}",
                f"- Jaccard index: {summary['jaccard_index']:.3f}",
                f"- Overlap coefficient: {summary['overlap_coefficient']:.3f}",
                f"- Overlap list: {summary['overlap']}",
                f"- Image only: {summary['left_only']}",
                f"- Text only: {summary['right_only']}",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for spec in MODEL_SPECS:
        left_heads = load_selected_heads(spec["left_path"])
        right_heads = load_selected_heads(spec["right_path"])
        summary = compute_summary(left_heads, right_heads)

        png_path = BASE_DIR / f"{spec['model']}_head_overlap_venn.png"
        draw_venn(spec["model"], spec["left_label"], spec["right_label"], summary, png_path)

        results.append(
            {
                "model": spec["model"],
                "left_label": spec["left_label"],
                "right_label": spec["right_label"],
                "left_path": str(spec["left_path"]),
                "right_path": str(spec["right_path"]),
                "venn_png": str(png_path),
                "summary": summary,
            }
        )

    summary_json = BASE_DIR / "head_overlap_summary.json"
    summary_md = BASE_DIR / "head_overlap_summary.md"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, summary_md)


if __name__ == "__main__":
    main()
