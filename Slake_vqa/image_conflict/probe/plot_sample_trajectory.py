import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from resume_utils import ResumeTracker, build_resume_dir


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
        label_names = {0: "nc", 1: "ic"}
    elif task in {"follow_conflict", "hallucination"}:
        target_key = "hallucination_targets" if "hallucination_targets" in bundle else "follow_targets"
        y = bundle[target_key].long()
        mask = torch.tensor([pt == "ic" for pt in prompt_types], dtype=torch.bool)
        mask &= (y == 0) | (y == 1)
        label_names = {0: "unknown", 1: "hallucination"}
    else:
        raise ValueError(f"Unknown task: {task}")

    keep = mask.tolist()
    return {
        "x": x[mask],
        "y": y[mask],
        "sample_ids": [sid for sid, flag in zip(bundle["sample_ids"], keep) if flag],
        "sample_keys": [sid for sid, flag in zip(bundle["sample_keys"], keep) if flag],
        "pair_ids": [sid for sid, flag in zip(bundle["pair_ids"], keep) if flag],
        "img_ids": [sid for sid, flag in zip(bundle["img_ids"], keep) if flag],
        "prompt_types": [sid for sid, flag in zip(bundle["prompt_types"], keep) if flag],
        "gold_answers": [sid for sid, flag in zip(bundle["gold_answers"], keep) if flag],
        "wrong_answers": [sid for sid, flag in zip(bundle["wrong_answers"], keep) if flag],
        "splits": [sid for sid, flag in zip(bundle["splits"], keep) if flag],
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


def find_index_by_sample_id(sample_ids, wanted_id):
    wanted = str(wanted_id)
    for idx, sample_id in enumerate(sample_ids):
        if str(sample_id) == wanted:
            return idx
    raise ValueError(f"sample_id not found: {wanted_id}")


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
    return {
        "indices": indices,
        "count": len(indices),
        "means": means,
        "stds": stds,
    }


def make_sample_record(bundle, sample_index, label_name):
    return {
        "sample_index": sample_index,
        "sample_id": bundle["sample_ids"][sample_index],
        "sample_key": bundle["sample_keys"][sample_index],
        "pair_id": bundle["pair_ids"][sample_index],
        "img_id": bundle["img_ids"][sample_index],
        "prompt_type": bundle["prompt_types"][sample_index],
        "gold_answer": bundle["gold_answers"][sample_index],
        "wrong_answer": bundle["wrong_answers"][sample_index],
        "split": bundle["splits"][sample_index],
        "label_name": label_name,
    }


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


def plot_group_scores(
    positive_means,
    negative_means,
    positive_label,
    negative_label,
    score_type,
    title,
    out_path,
):
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
    ap.add_argument("--task", required=True, choices=["conflict", "follow_conflict", "hallucination"])
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--score_type", default="prob", choices=["prob", "logit", "both"])
    ap.add_argument("--plot_mode", default="sample", choices=["sample", "mean"])
    ap.add_argument("--positive_sample_id", default="")
    ap.add_argument("--negative_sample_id", default="")
    ap.add_argument(
        "--selection_strategy",
        default="median",
        choices=["median", "highest", "first"],
        help="Used only when sample ids are not provided.",
    )
    ap.add_argument("--model", default="")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--position", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_png = Path(args.out_png)
    out_json = Path(args.out_json) if args.out_json else None
    expected_score_types = ["logit", "prob"] if args.score_type == "both" else [args.score_type]
    expected_plot_paths = [
        with_score_suffix(out_png, score_type) if args.score_type == "both" else out_png
        for score_type in expected_score_types
    ]
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            Path(__file__).resolve().parents[1],
            task_name="probe_plot_trajectory",
            model_name=args.model_name,
            model=args.model,
            position=args.position,
        ),
        enabled=args.resume,
    )
    tracker.start(
        features=args.features,
        probe_dir=args.probe_dir,
        task=args.task,
        plot_mode=args.plot_mode,
        score_type=args.score_type,
        out_png=str(out_png),
        out_json=str(out_json) if out_json else "",
        model=args.model,
        model_name=args.model_name,
        position=args.position,
    )
    if tracker.is_done(str(out_png)) and all(path.exists() for path in expected_plot_paths) and (out_json is None or out_json.exists()):
        for saved_plot in expected_plot_paths:
            print(f"saved_plot={saved_plot}")
        if out_json is not None:
            print(f"saved_json={out_json}")
        tracker.finish(out_png=str(out_png), out_json=str(out_json) if out_json else "")
        return

    bundle = load_task_bundle(args.features, args.task)
    display_task = "hallucination" if args.task in {"follow_conflict", "hallucination"} else args.task
    label_names = bundle["label_names"]
    score_types = expected_score_types

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_mode == "mean":
        group_payload = {}
        saved_plots = []
        for score_type in score_types:
            positive_group = build_group_scores(bundle, 1, args.probe_dir, score_type)
            negative_group = build_group_scores(bundle, 0, args.probe_dir, score_type)
            title = (
                f"{display_task} class-mean trajectories ({score_type})\n"
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
                "positive_group": {
                    "label_name": label_names[1],
                    "count": positive_group["count"],
                    "scores_mean_by_layer": positive_group["means"],
                    "scores_std_by_layer": positive_group["stds"],
                },
                "negative_group": {
                    "label_name": label_names[0],
                    "count": negative_group["count"],
                    "scores_mean_by_layer": negative_group["means"],
                    "scores_std_by_layer": negative_group["stds"],
                },
            }
        payload = {
            "task": display_task,
            "plot_mode": args.plot_mode,
            "score_type": args.score_type,
            "probe_dir": args.probe_dir,
            "features": args.features,
            "saved_plots": saved_plots,
            "by_score_type": group_payload,
        }
    else:
        if args.positive_sample_id:
            pos_idx = find_index_by_sample_id(bundle["sample_ids"], args.positive_sample_id)
        else:
            pos_idx = choose_representative_index(bundle, 1, args.probe_dir, args.score_type, args.selection_strategy)

        if args.negative_sample_id:
            neg_idx = find_index_by_sample_id(bundle["sample_ids"], args.negative_sample_id)
        else:
            neg_idx = choose_representative_index(bundle, 0, args.probe_dir, args.score_type, args.selection_strategy)

        if int(bundle["y"][pos_idx].item()) != 1:
            raise RuntimeError("positive_sample_id does not belong to the positive class.")
        if int(bundle["y"][neg_idx].item()) != 0:
            raise RuntimeError("negative_sample_id does not belong to the negative class.")

        pos_record = make_sample_record(bundle, pos_idx, label_names[1])
        neg_record = make_sample_record(bundle, neg_idx, label_names[0])
        score_payload = {}
        saved_plots = []
        for score_type in score_types:
            positive_scores = build_layer_scores(bundle, pos_idx, args.probe_dir, score_type)
            negative_scores = build_layer_scores(bundle, neg_idx, args.probe_dir, score_type)
            title = (
                f"{display_task} sample trajectories ({score_type})\n"
                f"{label_names[1]} id={pos_record['sample_id']} vs {label_names[0]} id={neg_record['sample_id']}"
            )
            target_png = with_score_suffix(out_png, score_type) if args.score_type == "both" else out_png
            plot_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                positive_label=f"{label_names[1]} ({pos_record['sample_id']})",
                negative_label=f"{label_names[0]} ({neg_record['sample_id']})",
                score_type=score_type,
                title=title,
                out_path=target_png,
            )
            saved_plots.append(str(target_png))
            score_payload[score_type] = {
                "positive_scores_by_layer": positive_scores,
                "negative_scores_by_layer": negative_scores,
            }
        payload = {
            "task": display_task,
            "plot_mode": args.plot_mode,
            "score_type": args.score_type,
            "probe_dir": args.probe_dir,
            "features": args.features,
            "saved_plots": saved_plots,
            "positive_sample": pos_record,
            "negative_sample": neg_record,
            "by_score_type": score_payload,
        }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.score_type == "both":
        for saved_plot in payload["saved_plots"]:
            print(f"saved_plot={saved_plot}")
    else:
        print(f"saved_plot={out_png}")
    if args.plot_mode == "mean":
        first_score_type = score_types[0]
        print(f"positive_count={payload['by_score_type'][first_score_type]['positive_group']['count']}")
        print(f"negative_count={payload['by_score_type'][first_score_type]['negative_group']['count']}")
    else:
        print(f"positive_sample_id={pos_record['sample_id']}")
        print(f"negative_sample_id={neg_record['sample_id']}")
    if args.out_json:
        print(f"saved_json={args.out_json}")
    tracker.mark_done(str(out_png), {"task": args.task, "plot_mode": args.plot_mode, "score_type": args.score_type})
    tracker.finish(out_png=str(out_png), out_json=str(out_json) if args.out_json else "")


if __name__ == "__main__":
    main()
