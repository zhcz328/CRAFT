from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURE_ROOT, MANIFEST_ROOT, ensure_result_dirs, get_trace_spec_map, load_curve, load_json


LOCAL_STYLE = {
    "color": "#ff7f0e",
    "linestyle": "-",
    "marker": "o",
    "linewidth": 2.2,
    "markersize": 3.6,
}
ALL_TOKEN_STYLE = {
    "color": "#1f77b4",
    "linestyle": "--",
    "marker": "o",
    "linewidth": 1.9,
    "markersize": 3.3,
}


def select_entries(manifest: Dict[str, Any], model_key: str, variants: Tuple[str, str]) -> Dict[str, Dict[str, Any]]:
    wanted = set(variants)
    picked: Dict[str, Dict[str, Any]] = {}
    for entry in manifest["experiments"]:
        if entry["model_key"] == model_key and entry["variant"] in wanted:
            picked[entry["variant"]] = entry
    return picked


def plot_single_panel(
    model_key: str,
    manifest: Dict[str, Any],
    out_path: Path,
    panel_title: str,
    use_reference_caption: bool,
) -> None:
    spec = get_trace_spec_map()[model_key]
    compare_variant = spec.compare_variant
    picked = select_entries(manifest, model_key, (compare_variant, "all_token"))
    missing = [variant for variant in [compare_variant, "all_token"] if variant not in picked]
    if missing:
        raise FileNotFoundError(f"Missing manifest entries for {model_key}: {missing}")

    local_entry = picked[compare_variant]
    all_entry = picked["all_token"]
    local_curve = load_curve(Path(local_entry["trace_json"]), local_entry.get("curve_metric"))
    all_curve = load_curve(Path(all_entry["trace_json"]), all_entry.get("curve_metric"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x_local = np.arange(len(local_curve))
    x_all = np.arange(len(all_curve))

    ax.plot(
        x_all,
        all_curve,
        label="Full-sequence patching",
        **ALL_TOKEN_STYLE,
    )
    ax.plot(
        x_local,
        local_curve,
        label=f"Local-k patching (k={local_entry['patch_k']})",
        **LOCAL_STYLE,
    )

    if use_reference_caption:
        ax.set_title(panel_title)
    else:
        ax.set_title(f"{panel_title}: {spec.display_name}")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Patch score")
    ax.grid(True, color="#cfd6e0", linewidth=0.6, alpha=0.65)
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(-0.5, max(len(local_curve), len(all_curve)) - 0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined(manifest: Dict[str, Any], out_path: Path) -> None:
    model_order = ["qwen3_4b", "hulumed4b"]
    titles = {
        "qwen3_4b": "(a) Text-only setting",
        "hulumed4b": "(b) Multimodal setting",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), sharey=False)
    for ax, model_key in zip(axes, model_order):
        spec = get_trace_spec_map()[model_key]
        compare_variant = spec.compare_variant
        picked = select_entries(manifest, model_key, (compare_variant, "all_token"))
        local_entry = picked[compare_variant]
        all_entry = picked["all_token"]
        local_curve = load_curve(Path(local_entry["trace_json"]), local_entry.get("curve_metric"))
        all_curve = load_curve(Path(all_entry["trace_json"]), all_entry.get("curve_metric"))

        ax.plot(np.arange(len(all_curve)), all_curve, label="Full-sequence patching", **ALL_TOKEN_STYLE)
        ax.plot(
            np.arange(len(local_curve)),
            local_curve,
            label=f"Local-k patching (k={local_entry['patch_k']})",
            **LOCAL_STYLE,
        )
        ax.set_title(f"{titles[model_key]}\n{spec.display_name}")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Patch score")
        ax.grid(True, color="#cfd6e0", linewidth=0.6, alpha=0.65)
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlim(-0.5, max(len(local_curve), len(all_curve)) - 0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot patch_k vs all_token comparison curves.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_ROOT / "patch_trace_manifest.json",
        help="Patch trace manifest produced by generate_run_manifest.py",
    )
    args = parser.parse_args()

    ensure_result_dirs()
    manifest = load_json(args.manifest)

    plot_single_panel(
        model_key="qwen3_4b",
        manifest=manifest,
        out_path=FIGURE_ROOT / "qwen3_4b_patch_compare.png",
        panel_title="(a) Text-only setting",
        use_reference_caption=True,
    )
    plot_single_panel(
        model_key="hulumed4b",
        manifest=manifest,
        out_path=FIGURE_ROOT / "hulumed4b_patch_compare.png",
        panel_title="(b) Multimodal setting",
        use_reference_caption=True,
    )
    plot_combined(manifest, FIGURE_ROOT / "patch_compare_panel.png")
    print(f"Saved patch comparison figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()

