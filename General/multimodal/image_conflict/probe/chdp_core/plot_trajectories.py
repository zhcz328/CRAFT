import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def compute_curves(bundle):
    d_h = torch.linalg.norm(bundle["nc_hidden_states"].float() - bundle["ic_hidden_states"].float(), dim=-1)
    m_gc_ic = bundle["ic_score_gold"].float() - bundle["ic_score_conflict"].float()
    return {
        "d_h": d_h,
        "A_img_ic": bundle["ic_attn_img"].float(),
        "A_ctx_ic": bundle["ic_attn_ctx"].float(),
        "M_gc_ic": m_gc_ic,
    }


def choose_sample_index(bundle, label_value, strategy):
    indices = [idx for idx, label in enumerate(bundle["labels"].tolist()) if int(label) == label_value]
    if not indices:
        raise RuntimeError(f"No samples found for label={label_value}.")
    if strategy == "first":
        return indices[0]
    curves = compute_curves(bundle)
    score = curves["d_h"].mean(dim=1)
    if strategy == "highest_divergence":
        return max(indices, key=lambda idx: float(score[idx].item()))
    median_value = float(torch.median(score[indices]).item())
    return min(indices, key=lambda idx: abs(float(score[idx].item()) - median_value))


def plot_single_sample(curves, bundle, sample_idx, out_path):
    layers = list(range(curves["d_h"].shape[1]))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    series = [
        ("d_h", "NC/IC hidden divergence", "#d04f3e"),
        ("A_img_ic", "IC attention on image", "#2f6db3"),
        ("A_ctx_ic", "IC attention on conflict evidence", "#3d8f57"),
        ("M_gc_ic", "IC gold-conflict margin", "#6a4c93"),
    ]
    for ax, (key, title, color) in zip(axes.flatten(), series):
        values = curves[key][sample_idx].tolist()
        ax.plot(layers, values, marker="o", linewidth=2.0, color=color)
        if key == "M_gc_ic":
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"CHDP single-sample trajectory | {bundle['label_names'][sample_idx]} | sample_id={bundle['sample_ids'][sample_idx]}",
        fontsize=13,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_group_means(curves, bundle, out_path):
    layers = list(range(curves["d_h"].shape[1]))
    labels = bundle["labels"].tolist()
    resist_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    hijack_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    series = [
        ("d_h", "Mean hidden divergence", "#d04f3e"),
        ("A_img_ic", "Mean IC attention on image", "#2f6db3"),
        ("A_ctx_ic", "Mean IC attention on conflict evidence", "#3d8f57"),
        ("M_gc_ic", "Mean IC gold-conflict margin", "#6a4c93"),
    ]
    for ax, (key, title, color) in zip(axes.flatten(), series):
        resist_mean = curves[key][resist_idx].mean(dim=0).tolist() if resist_idx else [0.0] * len(layers)
        hijack_mean = curves[key][hijack_idx].mean(dim=0).tolist() if hijack_idx else [0.0] * len(layers)
        ax.plot(layers, resist_mean, marker="o", linewidth=2.0, color="#2f6db3", label="Resist")
        ax.plot(layers, hijack_mean, marker="o", linewidth=2.0, color=color, label="Hijack")
        if key == "M_gc_ic":
            ax.axhline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("CHDP class-mean trajectories", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot CHDP trajectories from atomic feature dumps.")
    ap.add_argument("--atomic_bundle", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_id", default="")
    ap.add_argument("--sample_strategy", default="highest_divergence", choices=["highest_divergence", "median_divergence", "first"])
    args = ap.parse_args()

    bundle = torch.load(args.atomic_bundle, map_location="cpu")
    curves = compute_curves(bundle)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_id:
        try:
            sample_idx = bundle["sample_ids"].index(args.sample_id)
        except ValueError as exc:
            raise RuntimeError(f"sample_id not found: {args.sample_id}") from exc
    else:
        strategy = "median_divergence" if args.sample_strategy == "median_divergence" else args.sample_strategy
        sample_idx = choose_sample_index(bundle, label_value=1, strategy=strategy)

    single_path = out_dir / "single_sample_trajectory.png"
    mean_path = out_dir / "class_mean_trajectory.png"
    plot_single_sample(curves, bundle, sample_idx, single_path)
    plot_group_means(curves, bundle, mean_path)

    summary = {
        "atomic_bundle": args.atomic_bundle,
        "single_sample_id": bundle["sample_ids"][sample_idx],
        "single_sample_label": bundle["label_names"][sample_idx],
        "single_sample_plot": str(single_path),
        "group_mean_plot": str(mean_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_single_plot={single_path}")
    print(f"saved_group_plot={mean_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
