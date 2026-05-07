#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_legend(handles: list[Line2D], labels: list[str], out_base: Path, ncol: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 1.5))
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.4,
        handletextpad=0.7,
        fontsize=20,
    )
    fig.tight_layout(pad=0.1)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    text_dir = Path("/root/logit_lens/PIC/tuned_lens/text_lens")
    image_dir = Path("/root/logit_lens/PIC/tuned_lens/image_lens")

    text_handles = [
        Line2D([0], [0], color="#ff1f1f", lw=4.2, linestyle="-"),
        Line2D([0], [0], color="#0a8a12", lw=4.2, linestyle="-"),
        Line2D([0], [0], color="#1d2dff", lw=4.0, linestyle="--"),
    ]
    text_labels = [
        "Follow conflict",
        "Resist",
        "After ablation",
    ]
    save_legend(text_handles, text_labels, text_dir / "text_lens_hulu_med_4b_legend", ncol=3)

    image_handles = [
        Line2D([0], [0], color="#ff1f1f", lw=4.2, linestyle="-"),
        Line2D([0], [0], color="#0a8a12", lw=4.2, linestyle="-"),
        Line2D([0], [0], color="#1d2dff", lw=4.0, linestyle="--"),
    ]
    image_labels = [
        "Hallucination before",
        "Non-hallucination before",
        "After ablation",
    ]
    save_legend(image_handles, image_labels, image_dir / "image_lens_grouped_hulumed4b_legend", ncol=2)


if __name__ == "__main__":
    main()
