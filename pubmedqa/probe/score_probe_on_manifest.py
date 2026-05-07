import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import read_jsonl, wrap_as_chat


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


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def make_model(probe_type, input_dim, state_dict):
    if probe_type == "linear":
        return LinearProbe(input_dim)
    if probe_type == "mlp":
        hidden_dim = state_dict["net.0.weight"].shape[0]
        return MLPProbe(input_dim, hidden_dim)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def load_probe_specs(probe_dir):
    probe_dir = Path(probe_dir)
    layer_paths = sorted(probe_dir.glob("layer_*.pt"))
    if not layer_paths:
        raise RuntimeError(f"No probe checkpoints found in: {probe_dir}")
    probe_specs = {}
    for layer_path in layer_paths:
        ckpt = torch.load(layer_path, map_location="cpu")
        model = make_model(
            probe_type=ckpt["probe_type"],
            input_dim=ckpt["mean"].shape[-1],
            state_dict=ckpt["state_dict"],
        )
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        probe_specs[int(ckpt["layer_idx"])] = {
            "model": model,
            "mean": ckpt["mean"].float(),
            "std": ckpt["std"].float().clamp_min(1e-6),
        }
    return probe_specs


def apply_probe_batch(hidden_batch, probe_spec, score_type):
    x = (hidden_batch.float() - probe_spec["mean"]) / probe_spec["std"]
    with torch.no_grad():
        logits = probe_spec["model"](x).cpu()
    if score_type == "logit":
        return logits
    return torch.sigmoid(logits)


def filter_rows(rows, prompt_types, group_field, group_values, constant_group):
    if prompt_types:
        wanted_prompt_types = {item.strip() for item in prompt_types.split(",") if item.strip()}
        rows = [row for row in rows if row.get("prompt_type") in wanted_prompt_types]
    if group_values and group_field != "none":
        wanted_group_values = {item.strip() for item in group_values.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get(group_field, "")) in wanted_group_values]
    if group_field == "none":
        rows = [dict(row, __group_name__=constant_group) for row in rows]
    return rows


def plot_group_means(layer_indices, group_means, score_type, title, out_path):
    colors = ["#d04f3e", "#2f6db3", "#2a9d8f", "#e9c46a", "#7b61ff", "#111827"]
    plt.figure(figsize=(8.6, 4.8))
    for idx, (group_name, values) in enumerate(group_means.items()):
        plt.plot(layer_indices, values, marker="o", linewidth=2.2, color=colors[idx % len(colors)], label=group_name)
    if score_type == "prob":
        plt.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7)
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


def main():
    ap = argparse.ArgumentParser(description="Apply trained probes to manifest prompts and plot mean layer trajectories.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--probe_dir", required=True, nargs="+")
    ap.add_argument("--probe_label", nargs="*", default=[])
    ap.add_argument("--group_field", default="follow_label")
    ap.add_argument("--group_values", default="")
    ap.add_argument("--group_name", default="all")
    ap.add_argument("--prompt_types", default="base")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--score_type", default="logit", choices=["logit", "prob"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
    rows = filter_rows(rows, args.prompt_types, args.group_field, args.group_values, args.group_name)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows left after filtering.")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    first_device = next(model.parameters()).device
    probe_labels = args.probe_label if args.probe_label else [Path(p).name for p in args.probe_dir]
    if len(probe_labels) != len(args.probe_dir):
        raise ValueError("--probe_label count must match --probe_dir count.")

    probe_specs_by_label = {}
    layer_indices = None
    for probe_dir, probe_label in zip(args.probe_dir, probe_labels):
        specs = load_probe_specs(probe_dir)
        current_layers = sorted(specs.keys())
        if layer_indices is None:
            layer_indices = current_layers
        elif current_layers != layer_indices:
            raise RuntimeError("All probes must cover the same layer indices for joint plotting.")
        probe_specs_by_label[probe_label] = specs
    assert layer_indices is not None

    group_key = "__group_name__" if args.group_field == "none" else args.group_field
    base_groups = sorted({str(row.get(group_key, "")) for row in rows})
    series_names = []
    for probe_label in probe_labels:
        for base_group in base_groups:
            series_name = probe_label if args.group_field == "none" else f"{probe_label} | {base_group}"
            series_names.append(series_name)
    group_sums = {series_name: [0.0 for _ in layer_indices] for series_name in series_names}
    group_counts = defaultdict(int)

    for batch_rows in chunked(rows, args.batch_size):
        prompts = [wrap_as_chat(tok, row["prompt_text"], enable_thinking=args.enable_thinking) for row in batch_rows]
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        positions = enc["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states[1:]
        base_group_names = [str(row.get(group_key, "")) for row in batch_rows]
        for probe_label, probe_specs in probe_specs_by_label.items():
            for pos, layer_idx in enumerate(layer_indices):
                hs_tensor = hidden_states[layer_idx]
                batch_index = torch.arange(len(batch_rows), device=hs_tensor.device)
                local_positions = positions.to(hs_tensor.device)
                gathered = hs_tensor[batch_index, local_positions].cpu()
                scores = apply_probe_batch(gathered, probe_specs[layer_idx], args.score_type)
                for item_idx, score in enumerate(scores.tolist()):
                    base_group_name = base_group_names[item_idx]
                    series_name = probe_label if args.group_field == "none" else f"{probe_label} | {base_group_name}"
                    group_sums[series_name][pos] += float(score)
            for base_group_name in base_group_names:
                series_name = probe_label if args.group_field == "none" else f"{probe_label} | {base_group_name}"
                group_counts[series_name] += 1

    group_means = {
        group_name: [value / max(1, group_counts[group_name]) for value in values]
        for group_name, values in group_sums.items()
    }

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = f"Probe on manifest ({args.score_type})\nprompt_types={args.prompt_types or 'all'}, group_field={args.group_field}"
    plot_group_means(layer_indices, group_means, args.score_type, title, out_png)

    payload = {
        "manifest": args.manifest,
        "probe_dirs": args.probe_dir,
        "probe_labels": probe_labels,
        "group_field": args.group_field,
        "group_values": args.group_values,
        "group_name": args.group_name,
        "prompt_types": args.prompt_types,
        "score_type": args.score_type,
        "image_size": args.image_size,
        "layer_indices": layer_indices,
        "group_counts": dict(group_counts),
        "group_mean_scores_by_layer": group_means,
        "saved_plot": str(out_png),
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={out_json}")

    print(f"saved_plot={out_png}")
    for group_name, count in group_counts.items():
        print(f"group_count[{group_name}]={count}")


if __name__ == "__main__":
    main()
