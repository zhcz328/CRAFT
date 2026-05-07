import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CHDPBiGRU(nn.Module):
    def __init__(self, input_dim, layer_hidden_dim=64, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_hidden_dim, layer_hidden_dim),
        )
        self.gru = nn.GRU(
            input_size=layer_hidden_dim,
            hidden_size=layer_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(layer_hidden_dim * 2, layer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_hidden_dim, 2),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        gru_out, _ = self.gru(encoded)
        pooled = gru_out.mean(dim=1)
        return self.classifier(pooled)


class FlattenMLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        flat_dim = num_layers * input_dim
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


class SingleLayerLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)


def safe_div(numer, denom):
    return 0.0 if denom == 0 else float(numer / denom)


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
    return float(torch.trapz(tpr, fpr).item())


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
    return float(torch.trapz(precision, recall).item())


def compute_metrics(logits, labels):
    probs = torch.softmax(logits, dim=-1)[:, 1]
    preds = torch.argmax(logits, dim=-1)
    labels = labels.long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": binary_auroc(probs.cpu(), labels.cpu()),
        "auprc": binary_auprc(probs.cpu(), labels.cpu()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_selected_features(path, feature_set):
    bundle = torch.load(path, map_location="cpu")
    feature_names = bundle["feature_names"]
    feature_groups = bundle["feature_groups"]
    if feature_set in feature_groups:
        selected_names = feature_groups[feature_set]
    else:
        selected_names = [name.strip() for name in feature_set.split(",") if name.strip()]
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    selected_idx = [name_to_idx[name] for name in selected_names]
    x = bundle["features"][:, :, selected_idx].float()
    return {
        "x": x,
        "y": bundle["labels"].long(),
        "sample_ids": bundle["sample_ids"],
        "selected_names": selected_names,
        "all_feature_names": feature_names,
        "n_layers": int(bundle["n_layers"]),
        "feature_dim": int(len(selected_idx)),
    }


def evaluate_model(model, x, y, criterion, device):
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).cpu()
        loss = float(criterion(logits, y).item())
    metrics = compute_metrics(logits, y)
    metrics["loss"] = loss
    return metrics, logits


def standardize_seq(train_x, val_x, test_x=None):
    mean = train_x.mean(dim=(0, 1), keepdim=True)
    std = train_x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    train_norm = (train_x - mean) / std
    val_norm = (val_x - mean) / std
    test_norm = None if test_x is None else (test_x - mean) / std
    return train_norm, val_norm, test_norm, mean, std


def standardize_layer(train_x, val_x, test_x=None):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_norm = (train_x - mean) / std
    val_norm = (val_x - mean) / std
    test_norm = None if test_x is None else (test_x - mean) / std
    return train_norm, val_norm, test_norm, mean, std


def train_epochs(model, train_x, train_y, val_x, val_y, epochs, batch_size, lr, weight_decay, patience, selection_metric, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    best_state = None
    best_metrics = None
    best_score = None
    bad_epochs = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            optimizer.step()

        val_metrics, _ = evaluate_model(model, val_x, val_y, criterion, device)
        current_score = float(val_metrics[selection_metric])
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_metrics, _ = evaluate_model(model, val_x, val_y, criterion, device)
    model.load_state_dict(best_state)
    train_metrics, _ = evaluate_model(model, train_x, train_y, criterion, device)
    return {
        "state_dict": best_state,
        "train_metrics": train_metrics,
        "val_metrics": best_metrics,
    }


def run_sequence_model(args, train_bundle, val_bundle, test_bundle, out_dir, device):
    train_x, val_x, test_x, mean, std = standardize_seq(
        train_bundle["x"],
        val_bundle["x"],
        None if test_bundle is None else test_bundle["x"],
    )
    train_y = train_bundle["y"]
    val_y = val_bundle["y"]
    test_y = None if test_bundle is None else test_bundle["y"]

    if args.model_type == "bigru":
        model = CHDPBiGRU(
            input_dim=train_bundle["feature_dim"],
            layer_hidden_dim=args.layer_hidden_dim,
            dropout=args.dropout,
        ).to(device)
    else:
        model = FlattenMLP(
            num_layers=train_bundle["n_layers"],
            input_dim=train_bundle["feature_dim"],
            hidden_dim=args.flat_hidden_dim,
            dropout=args.dropout,
        ).to(device)

    trained = train_epochs(
        model=model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        selection_metric=args.selection_metric,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    test_metrics = None
    if test_bundle is not None:
        test_metrics, _ = evaluate_model(model, test_x, test_y, criterion, device)

    ckpt = {
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "selected_features": train_bundle["selected_names"],
        "state_dict": trained["state_dict"],
        "mean": mean,
        "std": std,
        "train_metrics": trained["train_metrics"],
        "val_metrics": trained["val_metrics"],
        "test_metrics": test_metrics,
    }
    torch.save(ckpt, out_dir / "best.pt")
    summary = {
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "selected_features": train_bundle["selected_names"],
        "selection_metric": args.selection_metric,
        "train_metrics": trained["train_metrics"],
        "val_metrics": trained["val_metrics"],
        "test_metrics": test_metrics,
    }
    return summary


def run_single_layer_linear(args, train_bundle, val_bundle, test_bundle, out_dir, device):
    layer_summaries = []
    best_layer = None
    best_score = None
    criterion = nn.CrossEntropyLoss()

    for layer_idx in range(train_bundle["n_layers"]):
        train_x, val_x, test_x, mean, std = standardize_layer(
            train_bundle["x"][:, layer_idx, :],
            val_bundle["x"][:, layer_idx, :],
            None if test_bundle is None else test_bundle["x"][:, layer_idx, :],
        )
        train_y = train_bundle["y"]
        val_y = val_bundle["y"]
        test_y = None if test_bundle is None else test_bundle["y"]
        model = SingleLayerLinear(train_bundle["feature_dim"]).to(device)

        trained = train_epochs(
            model=model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            selection_metric=args.selection_metric,
            device=device,
        )
        test_metrics = None
        if test_bundle is not None:
            test_metrics, _ = evaluate_model(model, test_x, test_y, criterion, device)

        ckpt = {
            "model_type": args.model_type,
            "layer_idx": layer_idx,
            "feature_set": args.feature_set,
            "selected_features": train_bundle["selected_names"],
            "state_dict": trained["state_dict"],
            "mean": mean,
            "std": std,
            "train_metrics": trained["train_metrics"],
            "val_metrics": trained["val_metrics"],
            "test_metrics": test_metrics,
        }
        torch.save(ckpt, out_dir / f"layer_{layer_idx:02d}.pt")

        record = {
            "layer_idx": layer_idx,
            "train_metrics": trained["train_metrics"],
            "val_metrics": trained["val_metrics"],
            "test_metrics": test_metrics,
        }
        layer_summaries.append(record)
        score = float(trained["val_metrics"][args.selection_metric])
        if best_score is None or score > best_score:
            best_score = score
            best_layer = layer_idx

    return {
        "model_type": args.model_type,
        "feature_set": args.feature_set,
        "selected_features": train_bundle["selected_names"],
        "selection_metric": args.selection_metric,
        "best_layer": best_layer,
        "best_score": best_score,
        "metrics_by_layer": layer_summaries,
    }


def main():
    ap = argparse.ArgumentParser(description="Train CHDP probes and baselines.")
    ap.add_argument("--train_bundle", required=True)
    ap.add_argument("--val_bundle", required=True)
    ap.add_argument("--test_bundle", default="")
    ap.add_argument("--feature_set", default="chdp_minimal")
    ap.add_argument("--model_type", default="bigru", choices=["bigru", "flat_mlp", "single_layer_linear"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--layer_hidden_dim", type=int, default=64)
    ap.add_argument("--flat_hidden_dim", type=int, default=256)
    ap.add_argument("--selection_metric", default="auprc", choices=["accuracy", "f1", "auroc", "auprc"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    train_bundle = load_selected_features(args.train_bundle, args.feature_set)
    val_bundle = load_selected_features(args.val_bundle, args.feature_set)
    test_bundle = load_selected_features(args.test_bundle, args.feature_set) if args.test_bundle else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type in {"bigru", "flat_mlp"}:
        summary = run_sequence_model(args, train_bundle, val_bundle, test_bundle, out_dir, device)
    else:
        summary = run_single_layer_linear(args, train_bundle, val_bundle, test_bundle, out_dir, device)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved_summary={out_dir / 'summary.json'}")
    if "best_layer" in summary:
        print(f"best_layer={summary['best_layer']}")
        print(f"best_{args.selection_metric}={summary['best_score']:.6f}")
    else:
        print(f"best_{args.selection_metric}={summary['val_metrics'][args.selection_metric]:.6f}")


if __name__ == "__main__":
    main()
