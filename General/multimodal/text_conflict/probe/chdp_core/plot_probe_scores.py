import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from train_probe import CHDPBiGRU, FlattenMLP


def load_feature_bundle(path):
    bundle = torch.load(path, map_location="cpu")
    return {
        "x": bundle["features"].float(),
        "labels": bundle["labels"].long(),
        "label_names": bundle["label_names"],
        "sample_ids": bundle["sample_ids"],
        "feature_names": bundle["feature_names"],
        "feature_groups": bundle["feature_groups"],
        "n_layers": int(bundle["n_layers"]),
        "feature_dim": int(bundle["feature_dim"]),
    }


def select_feature_indices(bundle, selected_names):
    name_to_idx = {name: idx for idx, name in enumerate(bundle["feature_names"])}
    return [name_to_idx[name] for name in selected_names]


def build_model(ckpt, n_layers, feature_dim):
    model_type = ckpt["model_type"]
    if model_type == "bigru":
        state_dict = ckpt["state_dict"]
        hidden_dim = int(state_dict["encoder.0.weight"].shape[0])
        model = CHDPBiGRU(input_dim=feature_dim, layer_hidden_dim=hidden_dim, dropout=0.0)
    elif model_type == "flat_mlp":
        state_dict = ckpt["state_dict"]
        hidden_dim = int(state_dict["net.0.weight"].shape[0])
        model = FlattenMLP(num_layers=n_layers, input_dim=feature_dim, hidden_dim=hidden_dim, dropout=0.0)
    else:
        raise ValueError(f"Unsupported model_type for plot_probe_scores.py: {model_type}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def normalize_inputs(x, ckpt):
    mean = ckpt["mean"].float()
    std = ckpt["std"].float().clamp_min(1e-6)
    return (x - mean) / std


def compute_prefix_scores(model, x):
    scores = []
    with torch.no_grad():
        for layer_idx in range(x.shape[1]):
            prefix_x = x.clone()
            if layer_idx + 1 < x.shape[1]:
                prefix_x[:, layer_idx + 1 :, :] = 0.0
            logits = model(prefix_x)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.append(probs.cpu())
    return torch.stack(scores, dim=1)


def find_sample_index(bundle, sample_id, label_value):
    if sample_id:
        for idx, sid in enumerate(bundle["sample_ids"]):
            if str(sid) == str(sample_id):
                return idx
        raise RuntimeError(f"sample_id not found: {sample_id}")
    for idx, label in enumerate(bundle["labels"].tolist()):
        if int(label) == label_value:
            return idx
    raise RuntimeError(f"No sample found for label={label_value}")


def plot_sample_scores(prefix_scores, bundle, pos_idx, neg_idx, out_path):
    layers = list(range(prefix_scores.shape[1]))
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(
        layers,
        prefix_scores[pos_idx].tolist(),
        marker="o",
        linewidth=2.2,
        color="#d04f3e",
        label=f"Hijack ({bundle['sample_ids'][pos_idx]})",
    )
    plt.plot(
        layers,
        prefix_scores[neg_idx].tolist(),
        marker="o",
        linewidth=2.2,
        color="#2f6db3",
        label=f"Resist ({bundle['sample_ids'][neg_idx]})",
    )
    plt.axhline(0.5, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Probe risk score")
    plt.title("Figure 3: Probe prefix risk score by layer")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_group_scores(prefix_scores, bundle, out_path):
    layers = list(range(prefix_scores.shape[1]))
    labels = bundle["labels"].tolist()
    resist_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    hijack_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]

    resist_mean = prefix_scores[resist_idx].mean(dim=0).tolist() if resist_idx else [0.0] * len(layers)
    hijack_mean = prefix_scores[hijack_idx].mean(dim=0).tolist() if hijack_idx else [0.0] * len(layers)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(layers, hijack_mean, marker="o", linewidth=2.2, color="#d04f3e", label=f"Hijack mean (n={len(hijack_idx)})")
    plt.plot(layers, resist_mean, marker="o", linewidth=2.2, color="#2f6db3", label=f"Resist mean (n={len(resist_idx)})")
    plt.axhline(0.5, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel("Mean probe risk score")
    plt.title("Figure 3: Probe prefix risk score by layer (class mean)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_earliest_warning(prefix_scores, bundle, threshold, out_path):
    labels = bundle["labels"].tolist()
    hijack_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    earliest = []
    for idx in hijack_idx:
        values = prefix_scores[idx].tolist()
        hit = next((layer for layer, score in enumerate(values) if score >= threshold), None)
        if hit is not None:
            earliest.append(hit)
    plt.figure(figsize=(7.2, 4.2))
    if earliest:
        bins = list(range(0, prefix_scores.shape[1] + 1))
        plt.hist(earliest, bins=bins, color="#d04f3e", alpha=0.85, rwidth=0.9)
    plt.xlabel("Earliest warning layer")
    plt.ylabel("Hijack sample count")
    plt.title(f"Figure 3: Earliest warning layer distribution (threshold={threshold:.2f})")
    plt.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return earliest


def main():
    ap = argparse.ArgumentParser(description="Plot Figure 3: probe risk score by layer.")
    ap.add_argument("--feature_bundle", required=True)
    ap.add_argument("--probe_ckpt", required=True, help="Path to best.pt from CHDP/train_probe.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--positive_sample_id", default="")
    ap.add_argument("--negative_sample_id", default="")
    ap.add_argument("--warning_threshold", type=float, default=0.5)
    args = ap.parse_args()

    bundle = load_feature_bundle(args.feature_bundle)
    ckpt = torch.load(args.probe_ckpt, map_location="cpu")
    selected_idx = select_feature_indices(bundle, ckpt["selected_features"])
    x = bundle["x"][:, :, selected_idx]
    x = normalize_inputs(x, ckpt)

    model = build_model(ckpt, n_layers=x.shape[1], feature_dim=x.shape[2])
    prefix_scores = compute_prefix_scores(model, x)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_idx = find_sample_index(bundle, args.positive_sample_id, label_value=1)
    neg_idx = find_sample_index(bundle, args.negative_sample_id, label_value=0)

    sample_plot = out_dir / "probe_score_sample.png"
    mean_plot = out_dir / "probe_score_mean.png"
    warning_plot = out_dir / "probe_earliest_warning_hist.png"

    plot_sample_scores(prefix_scores, bundle, pos_idx, neg_idx, sample_plot)
    plot_group_scores(prefix_scores, bundle, mean_plot)
    earliest = plot_earliest_warning(prefix_scores, bundle, args.warning_threshold, warning_plot)

    summary = {
        "feature_bundle": args.feature_bundle,
        "probe_ckpt": args.probe_ckpt,
        "selected_features": ckpt["selected_features"],
        "positive_sample_id": bundle["sample_ids"][pos_idx],
        "negative_sample_id": bundle["sample_ids"][neg_idx],
        "warning_threshold": args.warning_threshold,
        "saved_plots": {
            "sample": str(sample_plot),
            "mean": str(mean_plot),
            "earliest_warning_hist": str(warning_plot),
        },
        "earliest_warning_layers": earliest,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_sample_plot={sample_plot}")
    print(f"saved_mean_plot={mean_plot}")
    print(f"saved_warning_plot={warning_plot}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
