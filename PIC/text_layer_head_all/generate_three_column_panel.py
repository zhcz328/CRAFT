#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "text_layer_head_all"


def crop_panel_from_row(
    image: Image.Image,
    panel_idx: int,
    n_panels: int,
    left: float,
    right: float,
    wspace: float,
    y0_frac: float,
    y1_frac: float,
) -> Image.Image:
    w, h = image.size
    total_units = n_panels + wspace * (n_panels - 1)
    axw = (right - left) / total_units
    gap = wspace * axw

    x0 = left + panel_idx * (axw + gap)
    x1 = x0 + axw

    px0 = int(max(0, round(x0 * w)))
    px1 = int(min(w, round(x1 * w)))
    py0 = int(max(0, round(y0_frac * h)))
    py1 = int(min(h, round(y1_frac * h)))
    return image.crop((px0, py0, px1, py1))


def compose_grid(
    fig4_hulu: Image.Image,
    fig4_internvl: Image.Image,
    fig5_hulu: Image.Image,
    fig5_internvl: Image.Image,
    tlh_hulu: Image.Image,
    tlh_internvl: Image.Image,
    out_png: Path,
    out_pdf: Path,
) -> None:
    # 2 rows x 3 cols:
    # col1 fig4 (hulu/internvl), col2 fig5 (hulu/internvl), col3 text_layer_head (hulu/internvl)
    grid = [
        [fig4_hulu, fig5_hulu, tlh_hulu],
        [fig4_internvl, fig5_internvl, tlh_internvl],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=220)
    for r in range(2):
        for c in range(3):
            axes[r, c].imshow(grid[r][c])
            axes[r, c].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02, hspace=0.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compose 3-column panel from fig4/fig5/text_layer_head.")
    ap.add_argument("--out-prefix", default="fig4_fig5_textlayerhead_3col", help="Output file prefix.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    fig4_img = Image.open(ROOT / "PIC" / "fig4" / "fig4_layer_trace.png").convert("RGB")
    fig5_img = Image.open(ROOT / "PIC" / "fig5" / "fig5_bcp_cer_scatter.png").convert("RGB")

    tlh_hulu = Image.open(ROOT / "PIC" / "text_layer_head" / "text_layer_head_hulumed4b.png").convert("RGB")
    tlh_internvl = Image.open(ROOT / "PIC" / "text_layer_head" / "text_layer_head_internvl35_4b_filled.png").convert("RGB")

    # Panel index: 0-based -> 2 is Hulu-med-4B, 3 is InternVL3.5-4B in both fig4/fig5.
    # fig4 uses tight bbox in source script, so we use broad y-crop that keeps panel titles/axes.
    fig4_hulu = crop_panel_from_row(
        fig4_img, panel_idx=2, n_panels=4, left=0.055, right=0.985, wspace=0.30, y0_frac=0.14, y1_frac=0.94
    )
    fig4_internvl = crop_panel_from_row(
        fig4_img, panel_idx=3, n_panels=4, left=0.055, right=0.985, wspace=0.30, y0_frac=0.14, y1_frac=0.94
    )

    # fig5 has no tight bbox in source script.
    fig5_hulu = crop_panel_from_row(
        fig5_img, panel_idx=2, n_panels=4, left=0.055, right=0.995, wspace=0.26, y0_frac=0.09, y1_frac=0.93
    )
    fig5_internvl = crop_panel_from_row(
        fig5_img, panel_idx=3, n_panels=4, left=0.055, right=0.995, wspace=0.26, y0_frac=0.09, y1_frac=0.93
    )

    out_png = OUT_DIR / f"{args.out_prefix}.png"
    out_pdf = OUT_DIR / f"{args.out_prefix}.pdf"
    compose_grid(
        fig4_hulu=fig4_hulu,
        fig4_internvl=fig4_internvl,
        fig5_hulu=fig5_hulu,
        fig5_internvl=fig5_internvl,
        tlh_hulu=tlh_hulu,
        tlh_internvl=tlh_internvl,
        out_png=out_png,
        out_pdf=out_pdf,
    )
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

