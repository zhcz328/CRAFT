import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import (
    build_batch_inputs,
    ensure_video_import_compat,
    get_output_head,
    load_mm_model,
    move_to_device,
    read_jsonl,
)
from modeling import build_translators, resolve_dtype
from model_utils import load_processor_with_compat, prefix_model_relative_path, resolve_model_selection
from resume_utils import ResumeTracker, build_resume_dir, build_resume_scope


RAW_COLOR = "#ef553b"
ALIGNED_COLOR = "#1f77ff"


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pick_device(name, fallback):
    if name == "auto":
        return fallback
    return torch.device(name)


def filter_rows(rows, prompt_types):
    wanted = {item.strip() for item in prompt_types.split(",") if item.strip()}
    return [row for row in rows if row["prompt_type"] in wanted]


def infer_position_from_manifest(path_str: str) -> str:
    stem = Path(path_str).stem
    for candidate in ("before_question", "before_answer", "prefix"):
        if stem.endswith(candidate):
            return candidate
    return ""


def kl_divergence(p, q):
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def plot_kl_by_layer(layer_ids, raw_kl_means, aligned_kl_means, out_path, model_label, alignment_label):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    x = np.arange(len(layer_ids))
    xtick_labels = [str(layer + 1) for layer in layer_ids]

    ax.plot(
        x,
        raw_kl_means,
        color=RAW_COLOR,
        marker="s",
        markersize=4.5,
        linewidth=1.5,
        label="Logit lens",
    )
    ax.plot(
        x,
        aligned_kl_means,
        color=ALIGNED_COLOR,
        marker="o",
        markersize=3.8,
        linewidth=1.5,
        label=alignment_label,
    )

    ax.set_xlabel("Layer", fontsize=16)
    ax.set_ylabel("KL (nats)", fontsize=16)
    ax.set_title(f"Layer-to-Final KL Alignment ({model_label})", fontsize=18, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(frameon=True, fontsize=12, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot mean KL-to-final by layer for multimodal raw logit lens vs tuned lens.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="conflict")
    ap.add_argument("--alignment_mode", default="tuned_lens", choices=["tuned_lens"])
    ap.add_argument("--lens_ckpt", required=True)
    ap.add_argument("--model_label", default="")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_dir = prefix_model_relative_path(args.out_dir, model_name=args.model_name, model=args.model)
    position = infer_position_from_manifest(args.manifest)
    resume_scope = build_resume_scope(args.manifest, args.lens_ckpt)
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=PARENT_DIR,
            task_name="tuned_lens_plot_alignment",
            model_name=args.model_name,
            model=args.model,
            position=position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="tuned_lens_plot_alignment",
        manifest=args.manifest,
        out_dir=args.out_dir,
        position=position,
    )

    ensure_video_import_compat()
    rows = filter_rows(read_jsonl(args.manifest), args.prompt_types)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows left after prompt_types filtering.")

    processor = load_processor_with_compat(args.model)
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_head = get_output_head(model)
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lens_device = pick_device(args.lens_device, first_device)

    ckpt = torch.load(args.lens_ckpt, map_location="cpu")
    layer_ids = ckpt["layer_indices"]
    translators = build_translators(
        layer_indices=layer_ids,
        hidden_dim=ckpt["hidden_dim"],
        dtype=resolve_dtype(ckpt["translator_dtype"]),
        rank=ckpt["translator_rank"],
    ).to(lens_device)
    translators.load_state_dict(ckpt["state_dict"], strict=True)
    translators.eval()

    raw_kl_sums = np.zeros(len(layer_ids), dtype=np.float64)
    aligned_kl_sums = np.zeros(len(layer_ids), dtype=np.float64)
    layer_counts = np.zeros(len(layer_ids), dtype=np.int64)
    for batch_stat in tracker.read_records("batch_kl_stats.jsonl"):
        raw_kl_sums += np.array(batch_stat["raw_kl_sums"], dtype=np.float64)
        aligned_kl_sums += np.array(batch_stat["aligned_kl_sums"], dtype=np.float64)
        layer_counts += np.array(batch_stat["layer_counts"], dtype=np.int64)
    project_root = Path(__file__).resolve().parents[1]
    lm_device = output_head.weight.device

    batches = list(chunked(rows, args.batch_size))
    pbar = tqdm(list(enumerate(batches)), total=len(batches), desc="compute KL by layer", unit="batch")
    for batch_idx, batch_rows in pbar:
        batch_key = f"batch:{batch_idx}"
        if args.resume and tracker.is_done(batch_key):
            pbar.set_postfix(resumed=batch_idx)
            continue
        batch_inputs = build_batch_inputs(processor, batch_rows, project_root, image_size=args.image_size)
        batch_inputs.pop("token_type_ids", None)
        batch_inputs = move_to_device(batch_inputs, first_device, model_dtype)
        positions = batch_inputs["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states[1:]
        logits_batch_index = torch.arange(len(batch_rows), device=out.logits.device)
        logits_positions = positions.to(out.logits.device)
        final_logits = out.logits[logits_batch_index, logits_positions].to(
            device=lm_device, dtype=output_head.weight.dtype
        )
        target_probs = torch.softmax(final_logits / args.temperature, dim=-1).detach().float().cpu().numpy()

        batch_raw_kl_sums = np.zeros(len(layer_ids), dtype=np.float64)
        batch_aligned_kl_sums = np.zeros(len(layer_ids), dtype=np.float64)
        batch_layer_counts = np.zeros(len(layer_ids), dtype=np.int64)
        for pos, layer_idx in enumerate(layer_ids):
            hs_tensor = hidden_states[layer_idx]
            batch_index_hidden = torch.arange(len(batch_rows), device=hs_tensor.device)
            pos_hidden = positions.to(hs_tensor.device)
            gathered_hidden = hs_tensor[batch_index_hidden, pos_hidden]

            raw_logits = output_head(gathered_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
            raw_probs = torch.softmax(raw_logits / args.temperature, dim=-1).detach().float().cpu().numpy()
            raw_kl = kl_divergence(raw_probs, target_probs)

            translator = translators[str(layer_idx)]
            translator_dtype = next(translator.parameters()).dtype
            aligned_hidden = translator(gathered_hidden.to(device=lens_device, dtype=translator_dtype))
            aligned_logits = output_head(aligned_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
            aligned_probs = torch.softmax(aligned_logits / args.temperature, dim=-1).detach().float().cpu().numpy()
            aligned_kl = kl_divergence(aligned_probs, target_probs)

            raw_sum = float(raw_kl.sum())
            aligned_sum = float(aligned_kl.sum())
            raw_kl_sums[pos] += raw_sum
            aligned_kl_sums[pos] += aligned_sum
            layer_counts[pos] += len(batch_rows)
            batch_raw_kl_sums[pos] += raw_sum
            batch_aligned_kl_sums[pos] += aligned_sum
            batch_layer_counts[pos] += len(batch_rows)
        tracker.append_record(
            "batch_kl_stats.jsonl",
            {
                "batch_idx": int(batch_idx),
                "raw_kl_sums": batch_raw_kl_sums.tolist(),
                "aligned_kl_sums": batch_aligned_kl_sums.tolist(),
                "layer_counts": batch_layer_counts.tolist(),
            },
        )
        tracker.mark_done(batch_key, {"batch_idx": int(batch_idx)})

    raw_kl_means = raw_kl_sums / np.clip(layer_counts, 1, None)
    aligned_kl_means = aligned_kl_sums / np.clip(layer_counts, 1, None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = args.model_label or Path(str(args.model)).name
    fig_path = out_dir / "kl_by_layer_tuned_lens.png"
    plot_kl_by_layer(
        layer_ids=layer_ids,
        raw_kl_means=raw_kl_means,
        aligned_kl_means=aligned_kl_means,
        out_path=fig_path,
        model_label=model_label,
        alignment_label="Tuned lens",
    )

    summary = {
        "manifest": args.manifest,
        "model": args.model,
        "alignment_mode": args.alignment_mode,
        "lens_ckpt": args.lens_ckpt,
        "prompt_types": args.prompt_types,
        "image_size": args.image_size,
        "temperature": args.temperature,
        "n_samples": int(layer_counts[0]) if len(layer_counts) > 0 else 0,
        "layer_indices": layer_ids,
        "layers": [layer + 1 for layer in layer_ids],
        "mean_kl_logit_lens": raw_kl_means.tolist(),
        "mean_kl_tuned_lens": aligned_kl_means.tolist(),
        "saved_figure": str(fig_path),
    }
    summary_path = out_dir / "kl_by_layer_tuned_lens.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tracker.finish(saved_figure=str(fig_path), saved_summary=str(summary_path))

    print(f"saved_figure={fig_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
