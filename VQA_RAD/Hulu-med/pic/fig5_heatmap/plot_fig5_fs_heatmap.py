from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Fig.5 attention-head FS heatmap (layer x head)."
    )
    parser.add_argument(
        "--input-json",
        default=(
            "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
            "headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json"
        ),
        help="Path to head_scan_merged_unique_layers.json",
    )
    parser.add_argument(
        "--selected-json",
        default=(
            "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
            "headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json"
        ),
        help="Path to selected_heads_stable_hulumed4b.json",
    )
    parser.add_argument(
        "--label",
        default="Hulu-Med-4B (before_question)",
        help="Subplot label.",
    )
    parser.add_argument(
        "--compare-input-json",
        default="",
        help="Optional second head-scan json for side-by-side comparison.",
    )
    parser.add_argument(
        "--compare-selected-json",
        default="",
        help="Optional second selected-heads json.",
    )
    parser.add_argument(
        "--compare-label",
        default="",
        help="Optional second subplot label.",
    )
    parser.add_argument(
        "--layers-mode",
        choices=["scan", "all"],
        default="scan",
        help="Use only scanned layers or all layers.",
    )
    parser.add_argument(
        "--output-dir",
        default="VQA_RAD/Hulu-med/pic/fig5_heatmap",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--output-name",
        default="fig5_fs_heatmap_before_question_scan_layers",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG dpi.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(scan_json: Path, selected_json: Path, label: str) -> dict:
    scan = load_json(scan_json)
    selected = load_json(selected_json)

    n_layers = int(scan["n_layers"])
    n_heads = int(scan["n_heads"])
    heat = np.zeros((n_heads, n_layers), dtype=np.float32)

    for item in scan["results"]:
        layer = int(item["layer"])
        head = int(item["head"])
        eff = float(item.get("mean_abs_effect_reduction", 0.0))
        base = float(item.get("mean_abs_base_change", 0.0))
        fs = eff / (base + EPS)
        heat[head, layer] = fs

    selected_pairs = {
        (int(h["head"]), int(h["layer"])) for h in selected.get("selected", [])
    }

    scan_layers = scan.get("scan_layers") or []
    scan_layers = sorted({int(x) for x in scan_layers})

    return {
        "label": label,
        "heat": heat,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "selected_pairs": selected_pairs,
        "scan_layers": scan_layers,
    }


def choose_layers(payload: dict, mode: str) -> list[int]:
    if mode == "all":
        return list(range(payload["n_layers"]))
    if payload["scan_layers"]:
        return payload["scan_layers"]
    return list(range(payload["n_layers"]))


def plot_one_panel(
    ax: plt.Axes,
    payload: dict,
    layers_to_show: list[int],
    vmax: float,
    cmap: LinearSegmentedColormap,
):
    heat = payload["heat"][:, layers_to_show]
    im = ax.imshow(
        heat,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    n_show = len(layers_to_show)
    tick_step = 1 if n_show <= 20 else 2 if n_show <= 40 else 4
    x_ticks = list(range(0, n_show, tick_step))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(layers_to_show[i]) for i in x_ticks], fontsize=8)

    y_ticks = list(range(0, payload["n_heads"], 2))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(y) for y in y_ticks], fontsize=8)

    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_ylabel("Head Index", fontsize=10)
    ax.set_title(payload["label"], fontsize=11)

    layer_to_x = {layer: idx for idx, layer in enumerate(layers_to_show)}
    for head, layer in payload["selected_pairs"]:
        if layer not in layer_to_x:
            continue
        x = layer_to_x[layer]
        rect = Rectangle(
            (x - 0.5, head - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=1.1,
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.5, len(layers_to_show) - 0.5)
    ax.set_ylim(payload["n_heads"] - 0.5, -0.5)
    ax.grid(False)
    ax.set_box_aspect(1)
    return im


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = build_payload(
        scan_json=Path(args.input_json),
        selected_json=Path(args.selected_json),
        label=args.label,
    )

    payloads = [primary]

    if args.compare_input_json:
        compare_selected = (
            args.compare_selected_json if args.compare_selected_json else args.selected_json
        )
        compare_label = args.compare_label if args.compare_label else "Model B"
        secondary = build_payload(
            scan_json=Path(args.compare_input_json),
            selected_json=Path(compare_selected),
            label=compare_label,
        )
        payloads.append(secondary)

    vmax = max(float(np.max(p["heat"])) for p in payloads)
    vmax = max(vmax, 1e-6)
    cmap = LinearSegmentedColormap.from_list("white_to_darkred", ["#ffffff", "#7f0000"])

    n_cols = len(payloads)
    fig_w = 8 if n_cols == 1 else 16
    fig_h = 8
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    ims = []
    layers_meta = []
    for ax, payload in zip(axes, payloads):
        layers_to_show = choose_layers(payload, args.layers_mode)
        layers_meta.append(layers_to_show)
        ims.append(plot_one_panel(ax, payload, layers_to_show, vmax, cmap))

    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("FS score", fontsize=10)
    fig.suptitle("Fig.5 - Attention Head FS Heatmap (Selected Layers)", fontsize=13)

    png_path = output_dir / f"{args.output_name}.png"
    pdf_path = output_dir / f"{args.output_name}.pdf"
    meta_path = output_dir / f"{args.output_name}_meta.json"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "input_json": args.input_json,
        "selected_json": args.selected_json,
        "compare_input_json": args.compare_input_json,
        "compare_selected_json": args.compare_selected_json,
        "layers_mode": args.layers_mode,
        "layers_shown_per_panel": layers_meta,
        "vmax": vmax,
        "figure_size": [fig_w, fig_h],
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {pdf_path}")
    print(f"[OK] Saved: {meta_path}")


if __name__ == "__main__":
    main()
