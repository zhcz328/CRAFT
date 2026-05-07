import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import read_jsonl, wrap_as_chat
from modeling import build_translators, resolve_dtype


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
    ap = argparse.ArgumentParser(description="Plot mean KL-to-final by layer for raw logit lens vs tuned lens.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="conflict")
    ap.add_argument("--alignment_mode", default="tuned_lens", choices=["tuned_lens", "ridge_io"])
    ap.add_argument("--lens_ckpt", default="")
    ap.add_argument("--model_label", default="")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    rows = filter_rows(read_jsonl(args.manifest), args.prompt_types)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No rows left after prompt_types filtering.")

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

    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("Model must expose output embeddings.")

    first_device = next(model.parameters()).device
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        raise RuntimeError("Could not infer num_hidden_layers from model config.")
    layer_ids = list(range(n_layers))

    lens_device = pick_device(args.lens_device, first_device)
    translators = None
    if args.alignment_mode == "tuned_lens":
        if not args.lens_ckpt:
            raise RuntimeError("--lens_ckpt is required when --alignment_mode tuned_lens.")
        ckpt = torch.load(args.lens_ckpt, map_location="cpu")
        translators = build_translators(
            layer_indices=ckpt["layer_indices"],
            hidden_dim=ckpt["hidden_dim"],
            dtype=resolve_dtype(ckpt["translator_dtype"]),
            rank=ckpt["translator_rank"],
        ).to(lens_device)
        translators.load_state_dict(ckpt["state_dict"], strict=True)
        translators.eval()
        missing_layers = [layer for layer in layer_ids if str(layer) not in translators]
        if missing_layers:
            raise RuntimeError(f"Tuned lens checkpoint is missing layers: {missing_layers}")
    else:
        raise RuntimeError("Only --alignment_mode tuned_lens is supported in this KL-by-layer plot.")

    raw_kl_sums = np.zeros(n_layers, dtype=np.float64)
    aligned_kl_sums = np.zeros(n_layers, dtype=np.float64)
    layer_counts = np.zeros(n_layers, dtype=np.int64)
    lm_device = output_head.weight.device

    for batch_rows in chunked(rows, args.batch_size):
        prompts = [
            wrap_as_chat(tok, row["prompt_text"], enable_thinking=args.enable_thinking)
            for row in batch_rows
        ]
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        positions = enc["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states[1:]
        logits_batch_index = torch.arange(len(batch_rows), device=out.logits.device)
        logits_positions = positions.to(out.logits.device)
        final_logits = out.logits[logits_batch_index, logits_positions].to(
            device=lm_device, dtype=output_head.weight.dtype
        )
        target_probs = torch.softmax(final_logits / args.temperature, dim=-1).detach().float().cpu().numpy()

        for layer_idx in layer_ids:
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

            raw_kl_sums[layer_idx] += float(raw_kl.sum())
            aligned_kl_sums[layer_idx] += float(aligned_kl.sum())
            layer_counts[layer_idx] += len(batch_rows)

    raw_kl_means = raw_kl_sums / np.clip(layer_counts, 1, None)
    aligned_kl_means = aligned_kl_sums / np.clip(layer_counts, 1, None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_label = args.model_label or Path(str(args.model)).name
    fig_path = out_dir / f"kl_by_layer_{args.alignment_mode}.png"
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
        "temperature": args.temperature,
        "n_samples": int(layer_counts[0]) if len(layer_counts) > 0 else 0,
        "layers": [layer + 1 for layer in layer_ids],
        "mean_kl_logit_lens": raw_kl_means.tolist(),
        "mean_kl_tuned_lens": aligned_kl_means.tolist(),
        "saved_figure": str(fig_path),
    }
    summary_path = out_dir / f"kl_by_layer_{args.alignment_mode}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved_figure={fig_path}")
    print(f"saved_summary={summary_path}")
    print(json.dumps(summary["mean_kl_tuned_lens"][:5], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
