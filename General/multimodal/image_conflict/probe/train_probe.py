import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from resume_utils import ResumeTracker, build_resume_dir_from_output


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


def binary_confusion(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return tp, tn, fp, fn, probs


def safe_div(numer, denom):
    if denom == 0:
        return 0.0
    return numer / denom


def binary_auroc(probs, labels):
    labels = labels.float()
    pos = labels.sum().item()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    tpr = torch.cat([torch.tensor([0.0]), tp / pos, torch.tensor([1.0])])
    fpr = torch.cat([torch.tensor([0.0]), fp / neg, torch.tensor([1.0])])
    return torch.trapz(tpr, fpr).item()


def binary_auprc(probs, labels):
    labels = labels.float()
    pos = labels.sum().item()
    if pos == 0:
        return 0.0
    order = torch.argsort(probs, descending=True)
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / pos
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    return torch.trapz(precision, recall).item()


def compute_metrics(logits, labels):
    tp, tn, fp, fn, probs = binary_confusion(logits, labels)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision_pos = safe_div(tp, tp + fp)
    recall_pos = safe_div(tp, tp + fn)
    f1_pos = safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = safe_div(tn, tn + fn)
    recall_neg = safe_div(tn, tn + fp)
    f1_neg = safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    return {
        "accuracy": acc,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "macro_f1": macro_f1,
        "auroc": binary_auroc(probs.cpu(), labels.cpu()),
        "auprc": binary_auprc(probs.cpu(), labels.cpu()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_margin": (probs - 0.5).abs().mean().item(),
    }


def load_task_bundle(path, task):
    bundle = torch.load(path, map_location="cpu")
    x = bundle["hidden_states"].float()
    prompt_types = bundle["prompt_types"]

    if task == "conflict":
        y = bundle["conflict_targets"].long()
        mask = torch.ones(len(prompt_types), dtype=torch.bool)
    elif task in {"follow_conflict", "hallucination"}:
        target_key = "hallucination_targets" if "hallucination_targets" in bundle else "follow_targets"
        y = bundle[target_key].long()
        mask = torch.tensor([pt == "ic" for pt in prompt_types], dtype=torch.bool)
        mask &= (y == 0) | (y == 1)
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "x": x[mask],
        "y": y[mask],
        "sample_ids": [sid for sid, keep in zip(bundle["sample_ids"], mask.tolist()) if keep],
        "n_layers": x.shape[1],
        "hidden_dim": x.shape[2],
    }


def make_model(probe_type, input_dim, hidden_dim):
    if probe_type == "linear":
        return LinearProbe(input_dim)
    if probe_type == "mlp":
        return MLPProbe(input_dim, hidden_dim)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def evaluate(model, x, y, criterion, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).cpu()
        loss = criterion(logits, y.float()).item()
    metrics = compute_metrics(logits, y)
    metrics["loss"] = loss
    return metrics, logits


def main():
    ap = argparse.ArgumentParser(description="Train one probe per layer.")
    ap.add_argument("--train_features", required=True)
    ap.add_argument("--val_features", required=True)
    ap.add_argument("--task", default="conflict", choices=["conflict", "follow_conflict", "hallucination"])
    ap.add_argument("--probe_type", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--mlp_hidden_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    train_bundle = load_task_bundle(args.train_features, args.task)
    val_bundle = load_task_bundle(args.val_features, args.task)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    position = out_dir.parent.name if out_dir.parent.name else ""
    tracker = ResumeTracker(
        resume_dir=build_resume_dir_from_output(
            project_root=PARENT_DIR,
            task_name="train_probe",
            output_path=args.out_dir,
            position=position,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="train_probe",
        out_dir=str(out_dir),
        task_label=args.task,
        probe_type=args.probe_type,
    )

    metrics_by_layer = tracker.read_records("metrics_by_layer.jsonl")
    done_layers = {int(item["layer_idx"]) for item in metrics_by_layer if "layer_idx" in item}
    best_layer = None
    best_metric = None
    primary_metric = "auroc" if args.task == "conflict" else "macro_f1"
    for item in metrics_by_layer:
        val_metrics = item.get("val_metrics", {})
        score = float(val_metrics.get(primary_metric, 0.0))
        if best_metric is None or score > best_metric:
            best_metric = score
            best_layer = int(item["layer_idx"])

    layer_progress = tqdm(
        range(train_bundle["n_layers"]),
        desc="train probe layers",
        unit="layer",
    )

    for layer_idx in layer_progress:
        ckpt_path = out_dir / f"layer_{layer_idx:02d}.pt"
        if args.resume and layer_idx in done_layers and ckpt_path.exists():
            layer_progress.set_postfix(best_layer=best_layer, best=f"{(best_metric or 0.0):.4f}", resumed=layer_idx)
            continue
        x_train = train_bundle["x"][:, layer_idx, :]
        y_train = train_bundle["y"]
        x_val = val_bundle["x"][:, layer_idx, :]
        y_val = val_bundle["y"]

        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std

        dataset = TensorDataset(x_train, y_train.float())
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = make_model(args.probe_type, train_bundle["hidden_dim"], args.mlp_hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_score = None
        bad_epochs = 0

        epoch_progress = tqdm(
            range(args.epochs),
            desc=f"layer {layer_idx}",
            unit="epoch",
            leave=False,
        )

        for _ in epoch_progress:
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb.to(device))
                loss = criterion(logits, yb.to(device))
                loss.backward()
                optimizer.step()

            val_metrics, _ = evaluate(model, x_val, y_val, criterion, device)
            current_score = val_metrics[primary_metric]
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_state = {
                    "model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "metrics": val_metrics,
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    break

            epoch_progress.set_postfix(best=f"{best_val_score:.4f}", bad_epochs=bad_epochs)

        model.load_state_dict(best_state["model"])
        train_metrics, _ = evaluate(model, x_train, y_train, criterion, device)
        val_metrics = best_state["metrics"]

        layer_metrics = {
            "layer_idx": layer_idx,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        metrics_by_layer.append(layer_metrics)

        ckpt = {
            "layer_idx": layer_idx,
            "task": args.task,
            "probe_type": args.probe_type,
            "state_dict": best_state["model"],
            "mean": best_state["mean"],
            "std": best_state["std"],
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, ckpt_path)
        tracker.append_record("metrics_by_layer.jsonl", layer_metrics)
        tracker.mark_done(f"layer:{layer_idx}", {"layer_idx": int(layer_idx)})

        if best_metric is None or val_metrics[primary_metric] > best_metric:
            best_metric = val_metrics[primary_metric]
            best_layer = layer_idx
        tracker.update(best_layer=best_layer, best_metric=best_metric, primary_metric=primary_metric)

        layer_progress.set_postfix(best_layer=best_layer, best=f"{best_metric:.4f}")

    summary = {
        "task": args.task,
        "probe_type": args.probe_type,
        "primary_metric": primary_metric,
        "best_layer": best_layer,
        "best_metric": best_metric,
        "metrics_by_layer": metrics_by_layer,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tracker.finish(best_layer=best_layer, best_metric=best_metric, primary_metric=primary_metric)

    print(f"saved_summary={out_dir / 'summary.json'}")
    print(f"best_layer={best_layer}")
    print(f"best_{primary_metric}={best_metric:.6f}")


if __name__ == "__main__":
    main()
