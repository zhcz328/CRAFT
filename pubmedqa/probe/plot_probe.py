import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_probe(summary_path, metric_key, title, out_path):
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    xs = [row["layer_idx"] for row in data["metrics_by_layer"]]
    ys = [row["val_metrics"][metric_key] for row in data["metrics_by_layer"]]
    best_layer = data["best_layer"]
    best_metric = data["best_metric"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.axvline(best_layer, linestyle="--", alpha=0.6)
    plt.title(f"{title}\nbest layer={best_layer}, best {metric_key}={best_metric:.4f}")
    plt.xlabel("Layer")
    plt.ylabel(metric_key)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot a per-layer probe metric curve from train_probe.py summary.json.")
    ap.add_argument("--summary_path", required=True)
    ap.add_argument("--metric_key", required=True)
    ap.add_argument("--title", default="Probe")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--position", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    plot_probe(args.summary_path, args.metric_key, args.title, args.out_path)
    print(f"saved_plot={args.out_path}")


if __name__ == "__main__":
    main()
