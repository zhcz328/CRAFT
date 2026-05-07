import json
from pathlib import Path
import matplotlib.pyplot as plt

def plot_probe(summary_path, metric_key, title, out_path):
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    xs = [row["layer_idx"] for row in data["metrics_by_layer"]]
    ys = [row["val_metrics"][metric_key] for row in data["metrics_by_layer"]]
    best_layer = data["best_layer"]
    best_metric = data["best_metric"]

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

plot_probe(
    "probe/results/conflict_linear_prefix_ok/summary.json",
    "auroc",
    "Probe: conflict",
    "probe/results/conflict_linear_prefix_ok/auroc_curve.png",
)

plot_probe(
    "probe/results/follow_linear_prefix_ok/summary.json",
    "macro_f1",
    "Probe: follow_conflict",
    "probe/results/follow_linear_prefix_ok/macro_f1_curve.png",
)
