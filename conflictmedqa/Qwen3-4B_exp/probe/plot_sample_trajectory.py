import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_model(probe_type, input_dim, state_dict):
    if probe_type == "linear":
        return LinearProbe(input_dim)
    if probe_type == "mlp":
        hidden_dim = state_dict["net.0.weight"].shape[0]
        return MLPProbe(input_dim, hidden_dim)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def load_task_bundle(path, task):
    bundle = torch.load(path, map_location="cpu")
    x = bundle["hidden_states"].float()
    prompt_types = bundle["prompt_types"]

    if task == "conflict":
        y = bundle["conflict_targets"].long()
        mask = torch.ones(len(prompt_types), dtype=torch.bool)
        label_names = {0: "support", 1: "conflict"}
    elif task == "follow_conflict":
        y = bundle["follow_targets"].long()
        mask = torch.tensor([pt == "conflict" for pt in prompt_types], dtype=torch.bool)
        mask &= (y == 0) | (y == 1)
        label_names = {0: "resist", 1: "follow_conflict"}
    else:
        raise ValueError(f"Unknown task: {task}")

    keep = mask.tolist()
    return {
        "x": x[mask],
        "y": y[mask],
        "sample_ids": [sid for sid, flag in zip(bundle["sample_ids"], keep) if flag],
        "pair_ids": [sid for sid, flag in zip(bundle["pair_ids"], keep) if flag],
        "prompt_types": [sid for sid, flag in zip(bundle["prompt_types"], keep) if flag],
        "label_names": label_names,
        "n_layers": x.shape[1],
    }


def load_probe_ckpt(probe_dir, layer_idx):
    ckpt_path = Path(probe_dir) / f"layer_{layer_idx:02d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = make_model(
        probe_type=ckpt["probe_type"],
        input_dim=ckpt["mean"].shape[-1],
        state_dict=ckpt["state_dict"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return ckpt, model


def score_hidden(hidden_state, ckpt, model, score_type):
    mean = ckpt["mean"].float()
    std = ckpt["std"].float().clamp_min(1e-6)
    x = (hidden_state.unsqueeze(0).float() - mean) / std
    with torch.no_grad():
        logit = model(x).squeeze(0).item()
    if score_type == "logit":
        return logit
    return torch.sigmoid(torch.tensor(logit)).item()


def choose_representative_index(bundle, label_value, probe_dir, score_type, strategy):
    indices = [idx for idx, label in enumerate(bundle["y"].tolist()) if int(label) == label_value]
    if not indices:
        raise RuntimeError(f"No samples found for label={label_value}.")
    if strategy == "first":
        return indices[0]

    summary = json.loads((Path(probe_dir) / "summary.json").read_text(encoding="utf-8"))
    best_layer = summary["best_layer"]
    ckpt, model = load_probe_ckpt(probe_dir, best_layer)
    scores = [score_hidden(bundle["x"][idx, best_layer, :], ckpt, model, score_type) for idx in indices]

    if strategy == "highest":
        best_pos = max(range(len(indices)), key=lambda i: scores[i])
        return indices[best_pos]

    median_score = torch.median(torch.tensor(scores)).item()
    best_pos = min(range(len(indices)), key=lambda i: abs(scores[i] - median_score))
    return indices[best_pos]


def build_layer_scores(bundle, sample_index, probe_dir, score_type):
    scores = []
    for layer_idx in range(bundle["n_layers"]):
        ckpt, model = load_probe_ckpt(probe_dir, layer_idx)
        scores.append(score_hidden(bundle["x"][sample_index, layer_idx, :], ckpt, model, score_type))
    return scores


def build_group_scores(bundle, label_value, probe_dir, score_type):
    indices = [idx for idx, label in enumerate(bundle["y"].tolist()) if int(label) == label_value]
    if not indices:
        raise RuntimeError(f"No samples found for label={label_value}.")

    per_layer_scores = []
    for layer_idx in range(bundle["n_layers"]):
        ckpt, model = load_probe_ckpt(probe_dir, layer_idx)
        layer_scores = [score_hidden(bundle["x"][idx, layer_idx, :], ckpt, model, score_type) for idx in indices]
        per_layer_scores.append(layer_scores)

    means = [float(torch.tensor(scores).mean().item()) for scores in per_layer_scores]
    stds = [float(torch.tensor(scores).std(unbiased=False).item()) for scores in per_layer_scores]
    return {"count": len(indices), "means": means, "stds": stds}


def plot_scores(positive_scores, negative_scores, positive_label, negative_label, score_type, title, out_path):
    xs = list(range(len(positive_scores)))

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, positive_scores, marker="o", linewidth=2.2, color="#d04f3e", label=positive_label)
    plt.plot(xs, negative_scores, marker="o", linewidth=2.2, color="#2f6db3", label=negative_label)
    if score_type == "prob":
        plt.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.ylim(-0.02, 1.02)
        plt.ylabel("Probe probability")
    else:
        plt.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.ylabel("Probe logit")
    plt.xlabel("Layer")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_group_scores(positive_means, negative_means, positive_label, negative_label, score_type, title, out_path):
    xs = list(range(len(positive_means)))

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, positive_means, marker="o", linewidth=2.2, color="#d04f3e", label=positive_label)
    plt.plot(xs, negative_means, marker="o", linewidth=2.2, color="#2f6db3", label=negative_label)
    if score_type == "prob":
        plt.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.ylim(-0.02, 1.02)
        plt.ylabel("Mean probe probability")
    else:
        plt.axhline(0.0, linestyle="--", linewidth=1, color="gray", alpha=0.7)
        plt.ylabel("Mean probe logit")
    plt.xlabel("Layer")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def with_score_suffix(path, suffix):
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def main():
    ap = argparse.ArgumentParser(description="Plot per-layer probe scores for one positive and one negative sample.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--task", required=True, choices=["conflict", "follow_conflict"])
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--score_type", default="prob", choices=["prob", "logit", "both"])
    ap.add_argument("--plot_mode", default="sample", choices=["sample", "mean"])
    ap.add_argument("--selection_strategy", default="median", choices=["median", "highest", "first"])
    ap.add_argument("--model", default="")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--position", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_png = Path(args.out_png)
    out_json = Path(args.out_json) if args.out_json else None
    score_types = ["logit", "prob"] if args.score_type == "both" else [args.score_type]
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_task_bundle(args.features, args.task)
    label_names = bundle["label_names"]
    saved_plots = []
    payload = {"task": args.task, "plot_mode": args.plot_mode, "score_type": args.score_type}

    if args.plot_mode == "mean":
        group_payload = {}
        for score_type in score_types:
            positive_group = build_group_scores(bundle, 1, args.probe_dir, score_type)
            negative_group = build_group_scores(bundle, 0, args.probe_dir, score_type)
            title = (
                f"{args.task} class-mean trajectories ({score_type})\n"
                f"{label_names[1]} (n={positive_group['count']}) vs {label_names[0]} (n={negative_group['count']})"
            )
            target_png = with_score_suffix(out_png, score_type) if args.score_type == "both" else out_png
            plot_group_scores(
                positive_means=positive_group["means"],
                negative_means=negative_group["means"],
                positive_label=f"{label_names[1]} mean",
                negative_label=f"{label_names[0]} mean",
                score_type=score_type,
                title=title,
                out_path=target_png,
            )
            saved_plots.append(str(target_png))
            group_payload[score_type] = {
                "positive_group": positive_group,
                "negative_group": negative_group,
            }
        payload["by_score_type"] = group_payload
    else:
        pos_idx = choose_representative_index(bundle, 1, args.probe_dir, "prob", args.selection_strategy)
        neg_idx = choose_representative_index(bundle, 0, args.probe_dir, "prob", args.selection_strategy)
        score_payload = {}
        for score_type in score_types:
            positive_scores = build_layer_scores(bundle, pos_idx, args.probe_dir, score_type)
            negative_scores = build_layer_scores(bundle, neg_idx, args.probe_dir, score_type)
            title = (
                f"{args.task} sample trajectories ({score_type})\n"
                f"{label_names[1]} id={bundle['sample_ids'][pos_idx]} vs {label_names[0]} id={bundle['sample_ids'][neg_idx]}"
            )
            target_png = with_score_suffix(out_png, score_type) if args.score_type == "both" else out_png
            plot_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                positive_label=f"{label_names[1]} ({bundle['sample_ids'][pos_idx]})",
                negative_label=f"{label_names[0]} ({bundle['sample_ids'][neg_idx]})",
                score_type=score_type,
                title=title,
                out_path=target_png,
            )
            saved_plots.append(str(target_png))
            score_payload[score_type] = {
                "positive_scores_by_layer": positive_scores,
                "negative_scores_by_layer": negative_scores,
            }
        payload["positive_sample_id"] = bundle["sample_ids"][pos_idx]
        payload["negative_sample_id"] = bundle["sample_ids"][neg_idx]
        payload["by_score_type"] = score_payload

    payload["saved_plots"] = saved_plots
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={out_json}")
    for saved_plot in saved_plots:
        print(f"saved_plot={saved_plot}")


if __name__ == "__main__":
    main()
