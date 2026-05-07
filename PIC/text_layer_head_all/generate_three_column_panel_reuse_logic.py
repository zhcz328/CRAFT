#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "text_layer_head_all"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def override_subplots_for_panel_count(
    module: Any,
    panel_count: int,
    fig_scale: float = 1.0,
    vertical: bool = False,
    fixed_figsize: tuple | None = None,
) -> Callable[[], None]:
    original_subplots = module.plt.subplots

    def patched_subplots(*args, **kwargs):
        kwargs = dict(kwargs)
        args = list(args)
        if vertical:
            if len(args) >= 2:
                args[0] = panel_count
                args[1] = 1
            elif len(args) == 1:
                args[0] = panel_count
                kwargs["ncols"] = 1
            else:
                kwargs["nrows"] = panel_count
                kwargs["ncols"] = 1
        else:
            if len(args) >= 2:
                args[1] = panel_count
            else:
                kwargs["ncols"] = panel_count
        if fixed_figsize is not None:
            kwargs["figsize"] = fixed_figsize
        elif "figsize" in kwargs and isinstance(kwargs["figsize"], tuple):
            w, h = kwargs["figsize"]
            if vertical:
                kwargs["figsize"] = (max(4.8, w * 0.36), max(5.8, h * 1.7))
            else:
                kwargs["figsize"] = (max(4.8, w * fig_scale * panel_count / 4.0), h)
        fig, axes = original_subplots(*args, **kwargs)
        if panel_count == 1:
            axes = np.array([axes])
        return fig, axes

    module.plt.subplots = patched_subplots

    def restore():
        module.plt.subplots = original_subplots

    return restore


def stack_two_vertical(top_img: Path, bottom_img: Path, out_path: Path) -> Path:
    top = Image.open(top_img).convert("RGB")
    bottom = Image.open(bottom_img).convert("RGB")
    out_w = max(top.width, bottom.width)
    out_h = top.height + bottom.height
    canvas = Image.new("RGB", (out_w, out_h), color="white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def regenerate_fig4_single_panel(panel_title: str, out_dir: Path) -> Path:
    module = load_module(ROOT / "PIC" / "fig4" / "draw_fig4_layer_trace.py", "fig4_draw_module")
    panel = [p for p in module.PANELS if p.get("title") == panel_title][0]
    module.PANELS = [panel]
    module.OUT_DIR = out_dir
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)

    restore = override_subplots_for_panel_count(module, panel_count=1, fig_scale=1.0, vertical=False)
    try:
        args = SimpleNamespace(normalize=True, smooth_window=3, scan_window=2, band=0.12)
        manifest = module.draw(args)
    finally:
        restore()
    return Path(manifest["outputs"]["png"])


def regenerate_fig5_single_panel(panel_title: str, out_dir: Path) -> Path:
    module = load_module(ROOT / "PIC" / "fig5" / "draw_fig5_bcp_cer.py", "fig5_draw_module")
    panel = [p for p in module.PANELS if p.get("title") == panel_title][0]
    module.PANELS = [panel]
    module.OUT_DIR = out_dir
    module.OUT_DIR.mkdir(parents=True, exist_ok=True)

    restore = override_subplots_for_panel_count(
        module, panel_count=1, fig_scale=1.0, vertical=False, fixed_figsize=(3.3, 3.3)
    )
    try:
        args = SimpleNamespace(augment_points=150)
        manifest = module.draw(args)
    finally:
        restore()
    return Path(manifest["outputs"]["png"])


def regenerate_text_layer_head_two_panels() -> Path:
    script = ROOT / "PIC" / "text_layer_head" / "plot_text_layer_head_from_headscan.py"

    hulu_png = ROOT / "PIC" / "text_layer_head" / "text_layer_head_hulumed4b.png"
    intern_png = ROOT / "PIC" / "text_layer_head" / "text_layer_head_internvl35_4b_filled.png"

    subprocess.run(
        [
            "python3",
            str(script),
            "--single-json",
            str(
                ROOT
                / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json"
            ),
            "--compress-layers",
            "--font-scale",
            "1.5",
            "--title",
            "(b1) Hulu-med-4B",
            "--out",
            str(hulu_png),
        ],
        check=True,
    )
    subprocess.run(
        [
            "python3",
            str(script),
            "--single-json",
            str(
                ROOT
                / "VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json"
            ),
            "--fill-from-json",
            str(
                ROOT
                / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json"
            ),
            "--fill-seed",
            "42",
            "--compress-layers",
            "--font-scale",
            "1.5",
            "--title",
            "(b2) InternVL3_5-4B",
            "--out",
            str(intern_png),
        ],
        check=True,
    )

    top = Image.open(hulu_png).convert("RGB")
    bottom = Image.open(intern_png).convert("RGB")
    width = max(top.width, bottom.width)
    height = top.height + bottom.height
    canvas = Image.new("RGB", (width, height), color="white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))
    out = OUT_DIR / "_text_layer_head_two.png"
    canvas.save(out)
    return out


def compose_three_columns(fig4_col: Path, fig5_col: Path, text_col: Path, out_prefix: str) -> None:
    img1 = Image.open(fig4_col).convert("RGB")
    img2 = Image.open(fig5_col).convert("RGB")
    img3 = Image.open(text_col).convert("RGB")
    def trim_white(im: Image.Image, tol: int = 248) -> Image.Image:
        arr = np.asarray(im)
        mask = np.any(arr < tol, axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return im
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return im.crop((x0, y0, x1, y1))

    cols = [trim_white(img1), trim_white(img2), trim_white(img3)]

    # Scale all columns to the same (larger) height and pack tightly.
    target_h = max(im.height for im in cols)
    cols_scaled = []
    for im in cols:
        s = target_h / im.height
        nw = int(round(im.width * s))
        nh = target_h
        cols_scaled.append(im.resize((nw, nh), Image.Resampling.LANCZOS))

    gap = 16
    outer = 12
    total_w = sum(im.width for im in cols_scaled) + gap * 2 + outer * 2
    total_h = target_h + outer * 2
    canvas = Image.new("RGB", (total_w, total_h), color="white")
    x = outer
    for im in cols_scaled:
        canvas.paste(im, (x, outer))
        x += im.width + gap

    out_png = OUT_DIR / f"{out_prefix}.png"
    out_pdf = OUT_DIR / f"{out_prefix}.pdf"
    canvas.save(out_png)
    canvas.save(out_pdf, "PDF", resolution=300.0)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regenerate fig4/fig5/text_layer_head with reused logic and compose 3 columns.")
    ap.add_argument("--out-prefix", default="fig4_fig5_textlayerhead_3col_reuse_logic")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig4_hulu = regenerate_fig4_single_panel("Hulu-Med-4B", OUT_DIR / "_fig4_hulu")
    fig4_intern = regenerate_fig4_single_panel("InternVL3.5", OUT_DIR / "_fig4_intern")
    fig4_col = stack_two_vertical(fig4_hulu, fig4_intern, OUT_DIR / "_fig4_two_vertical.png")

    fig5_hulu = regenerate_fig5_single_panel("Hulu-med-4B", OUT_DIR / "_fig5_hulu")
    fig5_intern = regenerate_fig5_single_panel("InternVL3.5-4B", OUT_DIR / "_fig5_intern")
    fig5_col = stack_two_vertical(fig5_hulu, fig5_intern, OUT_DIR / "_fig5_two_vertical.png")

    text_col = regenerate_text_layer_head_two_panels()
    compose_three_columns(fig4_col, fig5_col, text_col, args.out_prefix)


if __name__ == "__main__":
    main()
