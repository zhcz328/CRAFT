import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import read_jsonl
from extract_features import (
    build_batch_inputs,
    ensure_video_import_compat,
    load_mm_model,
    move_to_device,
    resolve_dtype,
)
from model_utils import load_processor_with_compat, prefix_model_relative_path, resolve_model_selection
from resume_utils import ResumeTracker, build_resume_dir, build_resume_scope


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
        layer_idx = int(ckpt["layer_idx"])
        model = make_model(
            probe_type=ckpt["probe_type"],
            input_dim=ckpt["mean"].shape[-1],
            state_dict=ckpt["state_dict"],
        )
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        probe_specs[layer_idx] = {
            "model": model,
            "mean": ckpt["mean"].float(),
            "std": ckpt["std"].float().clamp_min(1e-6),
            "task": ckpt["task"],
            "probe_type": ckpt["probe_type"],
        }
    return probe_specs


def resolve_probe_labels(probe_dirs, probe_labels):
    if probe_labels:
        if len(probe_labels) != len(probe_dirs):
            raise ValueError("--probe_label count must match --probe_dir count.")
        return probe_labels
    labels = []
    for probe_dir in probe_dirs:
        labels.append(Path(probe_dir).name)
    return labels


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


def infer_position_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    for candidate in ("before_question", "before_answer", "prefix"):
        if stem.endswith(candidate):
            return candidate
    return ""


def plot_group_means(layer_indices, group_means, score_type, title, out_path):
    colors = ["#d04f3e", "#2f6db3", "#2a9d8f", "#e9c46a", "#7b61ff", "#111827"]
    plt.figure(figsize=(8.6, 4.8))
    for idx, (group_name, values) in enumerate(group_means.items()):
        plt.plot(
            layer_indices,
            values,
            marker="o",
            linewidth=2.2,
            color=colors[idx % len(colors)],
            label=group_name,
        )
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
    ap = argparse.ArgumentParser(description="Apply a trained probe to manifest prompts and plot mean layer trajectories.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--probe_dir", required=True, nargs="+")
    ap.add_argument("--probe_label", nargs="*", default=[])
    ap.add_argument("--group_field", default="follow_label")
    ap.add_argument("--group_values", default="")
    ap.add_argument("--group_name", default="all", help="Used when --group_field none.")
    ap.add_argument("--prompt_types", default="base")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--score_type", default="logit", choices=["logit", "prob"])
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_png = prefix_model_relative_path(args.out_png, model_name=args.model_name, model=args.model)
    if args.out_json:
        args.out_json = prefix_model_relative_path(args.out_json, model_name=args.model_name, model=args.model)

    ensure_video_import_compat()
    manifest_path = Path(args.manifest)
    position = infer_position_from_manifest(manifest_path)
    resume_scope = build_resume_scope(manifest_path)
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=PARENT_DIR,
            task_name="probe_score_manifest",
            model_name=args.model_name,
            model=args.model,
            position=position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="probe_score_manifest",
        manifest=args.manifest,
        out_png=args.out_png,
        out_json=args.out_json,
        position=position,
    )
    rows = read_jsonl(args.manifest)
    rows = filter_rows(rows, args.prompt_types, args.group_field, args.group_values, args.group_name)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows left after filtering.")

    processor = load_processor_with_compat(args.model)
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    project_root = Path(__file__).resolve().parents[1]
    probe_labels = resolve_probe_labels(args.probe_dir, args.probe_label)
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
    for batch_stat in tracker.read_records("batch_group_stats.jsonl"):
        for group_name, values in batch_stat.get("group_sums", {}).items():
            if group_name not in group_sums:
                group_sums[group_name] = [0.0 for _ in layer_indices]
            for pos, value in enumerate(values):
                group_sums[group_name][pos] += float(value)
        for group_name, count in batch_stat.get("group_counts", {}).items():
            group_counts[group_name] += int(count)

    batches = list(chunked(rows, args.batch_size))
    pbar = tqdm(list(enumerate(batches)), total=len(batches), desc="score probe on manifest", unit="batch")
    for batch_idx, batch_rows in pbar:
        batch_key = f"batch:{batch_idx}"
        if args.resume and tracker.is_done(batch_key):
            pbar.set_postfix(resumed=batch_idx, scored=sum(group_counts.values()))
            continue
        batch_inputs = build_batch_inputs(processor, batch_rows, project_root, image_size=args.image_size)
        batch_inputs.pop("token_type_ids", None)
        batch_inputs = move_to_device(batch_inputs, first_device, model_dtype)
        positions = batch_inputs["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states[1:]
        base_group_names = [str(row.get(group_key, "")) for row in batch_rows]
        batch_group_sums = {series_name: [0.0 for _ in layer_indices] for series_name in series_names}
        batch_group_counts = defaultdict(int)
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
                    batch_group_sums[series_name][pos] += float(score)
            for base_group_name in base_group_names:
                series_name = probe_label if args.group_field == "none" else f"{probe_label} | {base_group_name}"
                group_counts[series_name] += 1
                batch_group_counts[series_name] += 1
        tracker.append_record(
            "batch_group_stats.jsonl",
            {
                "batch_idx": int(batch_idx),
                "group_sums": batch_group_sums,
                "group_counts": dict(batch_group_counts),
            },
        )
        tracker.mark_done(batch_key, {"batch_idx": int(batch_idx)})
        tracker.update(completed_batches=tracker.completed_count, total_batches=len(batches))
        pbar.set_postfix(scored=sum(group_counts.values()))

    group_means = {
        group_name: [value / max(1, group_counts[group_name]) for value in values]
        for group_name, values in group_sums.items()
    }

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    title = (
        f"Probe on manifest ({args.score_type})\n"
        f"prompt_types={args.prompt_types or 'all'}, group_field={args.group_field}"
    )
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
    tracker.finish(saved_plot=str(out_png), saved_json=args.out_json or "")

    print(f"saved_plot={out_png}")
    for group_name, count in group_counts.items():
        print(f"group_count[{group_name}]={count}")
    if args.out_json:
        print(f"saved_json={args.out_json}")


if __name__ == "__main__":
    main()
