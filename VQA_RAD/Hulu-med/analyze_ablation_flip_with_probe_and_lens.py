import argparse
import csv
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor


ROOT_DIR = Path(__file__).resolve().parent


def load_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


probe_extract = load_module("hulumed_probe_extract", ROOT_DIR / "probe" / "extract_features.py")
lens_common = load_module("hulumed_lens_common", ROOT_DIR / "tuned_lens" / "common.py")
lens_modeling = load_module("hulumed_lens_modeling", ROOT_DIR / "tuned_lens" / "modeling.py")
ablate_mod = load_module("hulumed_ablate_head", ROOT_DIR / "ablate_head.py")


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


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_model(probe_type, input_dim, state_dict):
    if probe_type == "linear":
        return LinearProbe(input_dim)
    if probe_type == "mlp":
        hidden_dim = state_dict["net.0.weight"].shape[0]
        return MLPProbe(input_dim, hidden_dim)
    raise ValueError(f"Unknown probe_type: {probe_type}")


def load_probe_specs(probe_dir):
    probe_specs = {}
    for layer_path in sorted(Path(probe_dir).glob("layer_*.pt")):
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
    if not probe_specs:
        raise RuntimeError(f"No probe checkpoints found in {probe_dir}")
    return probe_specs


def apply_probe_batch(hidden_batch, probe_spec, score_type):
    x = (hidden_batch.float() - probe_spec["mean"]) / probe_spec["std"]
    with torch.no_grad():
        logits = probe_spec["model"](x).cpu()
    if score_type == "logit":
        return logits
    return torch.sigmoid(logits)


def resolve_ablation_meta(ablation_json, data_csv, selected_heads, position):
    payload = json.loads(Path(ablation_json).read_text(encoding="utf-8"))
    config = payload.get("config", {})
    data_csv = data_csv or config.get("data_csv", "")
    selected_heads = selected_heads or config.get("selected_heads", "")
    position = position or config.get("position", "before_question")
    if not data_csv or not selected_heads:
        raise RuntimeError("Need data_csv and selected_heads, either from args or ablation json config.")
    image_root = str(config.get("image_root", "."))
    trace_mode = str(config.get("trace_mode", "conflict"))
    keep_mode = str(config.get("keep_mode", "self"))
    max_image_side = int(config.get("max_image_side", 672))
    return payload, str(data_csv), str(selected_heads), str(position), image_root, trace_mode, keep_mode, max_image_side, config


def select_flip_rows(ablation_payload, csv_rows):
    selected = []
    for record in ablation_payload.get("records", []):
        if record.get("base_ctx_pred") != "wrong":
            continue
        if record.get("ab_ctx_pred") != "gold":
            continue
        row_idx = int(record["row_idx"])
        if row_idx < 0 or row_idx >= len(csv_rows):
            continue
        source = csv_rows[row_idx]
        question = source.get("question", "")
        gold = source.get("gold_norm") or source.get("gold") or record.get("gold", "")
        wrong = source.get("wrong_norm") or source.get("wrong") or record.get("wrong", "")
        selected.append(
            {
                "row_idx": row_idx,
                "sample_id": str(source.get("id", row_idx)),
                "sample_key": lens_common.sample_key(source.get("img_id", ""), question),
                "img_id": source.get("img_id", ""),
                "image_path": source.get("image_path", ""),
                "question": question,
                "gold_answer": gold,
                "wrong_answer": wrong,
            }
        )
    return selected


def build_ablate_args(data_csv, image_root, position, trace_mode, max_image_side):
    return Namespace(
        data_csv=data_csv,
        image_root=image_root,
        max_examples=-1,
        max_image_side=max_image_side,
        position=position,
        trace_mode=trace_mode,
    )


def select_flip_samples_from_built(ablation_payload, built_samples):
    flip_map = {
        int(record["row_idx"]): record
        for record in ablation_payload.get("records", [])
        if record.get("base_ctx_pred") == "wrong" and record.get("ab_ctx_pred") == "gold"
    }
    selected = []
    for sample in built_samples:
        row_idx = int(sample["row_idx"])
        if row_idx not in flip_map:
            continue
        record = flip_map[row_idx]
        item = dict(sample)
        item["ablation_record"] = record
        selected.append(item)
    return selected


def load_tuned_lens(lens_ckpt_path, device):
    ckpt = torch.load(lens_ckpt_path, map_location="cpu")
    translators = lens_modeling.build_translators(
        layer_indices=ckpt["layer_indices"],
        hidden_dim=ckpt["hidden_dim"],
        dtype=lens_modeling.resolve_dtype(ckpt["translator_dtype"]),
        rank=ckpt["translator_rank"],
    ).to(device)
    translators.load_state_dict(ckpt["state_dict"], strict=True)
    translators.eval()
    return ckpt, translators


def load_model_like_ablate_head(model_name, dtype, device):
    load_errs = []
    for cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
        try:
            model = cls.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            return model.to(device)
        except Exception as exc:
            load_errs.append(f"{cls.__name__}: {repr(exc)}")
    raise RuntimeError("Failed to load model. " + " | ".join(load_errs))


def max_abs_diff(xs, ys):
    if len(xs) != len(ys):
        raise ValueError("Cannot compare sequences with different lengths.")
    if not xs:
        return 0.0
    return max(abs(float(x) - float(y)) for x, y in zip(xs, ys))


def is_numeric_like(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def summarize_ablation_effect(before_summary, after_summary):
    probe_diff = max_abs_diff(
        before_summary["probe_mean_scores_by_layer"],
        after_summary["probe_mean_scores_by_layer"],
    )
    gold_diff = max_abs_diff(
        before_summary["tuned_lens_mean_gold_logprob_by_layer"],
        after_summary["tuned_lens_mean_gold_logprob_by_layer"],
    )
    wrong_diff = max_abs_diff(
        before_summary["tuned_lens_mean_wrong_logprob_by_layer"],
        after_summary["tuned_lens_mean_wrong_logprob_by_layer"],
    )
    return {
        "probe_max_abs_diff": probe_diff,
        "gold_max_abs_diff": gold_diff,
        "wrong_max_abs_diff": wrong_diff,
    }


def verify_ablation_changes_original_path(model, sample, selected_heads_path, keep_mode, trace_mode):
    tokenizer = sample["ctx_inputs"].get("tokenizer", None)
    if tokenizer is None:
        tokenizer = None
    device = next(model.parameters()).device

    before_cache = ablate_mod.build_prompt_cache_from_inputs(model, sample["ctx_inputs"])
    before_scores = ablate_mod.score_answer_candidates_with_cache(
        model, sample.get("tokenizer", None) or sample["_tokenizer"], before_cache, sample["gold"], sample["wrong"], trace_mode, device
    )

    selected_pairs = ablate_mod.parse_selected_heads(selected_heads_path)
    layer_to_heads = ablate_mod.pack_layer_to_heads(selected_pairs)
    handles = ablate_mod.install_head_mask_hooks(model, layer_to_heads, keep_mode=keep_mode)
    try:
        after_cache = ablate_mod.build_prompt_cache_from_inputs(model, sample["ctx_inputs"])
        after_scores = ablate_mod.score_answer_candidates_with_cache(
            model, sample.get("tokenizer", None) or sample["_tokenizer"], after_cache, sample["gold"], sample["wrong"], trace_mode, device
        )
    finally:
        for handle in handles:
            handle.remove()

    keys = sorted(set(before_scores.keys()) | set(after_scores.keys()))
    max_diff = 0.0
    numeric_before = {}
    numeric_after = {}
    for key in keys:
        before_value = before_scores.get(key, 0.0)
        after_value = after_scores.get(key, 0.0)
        if not (is_numeric_like(before_value) and is_numeric_like(after_value)):
            continue
        before_value = float(before_value)
        after_value = float(after_value)
        numeric_before[key] = before_value
        numeric_after[key] = after_value
        max_diff = max(max_diff, abs(before_value - after_value))
    return {
        "row_idx": sample["row_idx"],
        "max_abs_score_diff": max_diff,
        "before_scores": numeric_before,
        "after_scores": numeric_after,
        "n_selected_heads": len(selected_pairs),
    }


def plot_single_line(layer_indices, values, ylabel, title, out_path, color, zero_line):
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(layer_indices, values, marker="o", linewidth=2.2, color=color)
    plt.axhline(zero_line, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_gold_wrong(layer_indices, gold_values, wrong_values, title, out_path):
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(layer_indices, gold_values, marker="o", linewidth=2.2, color="#2f6db3", label="gold")
    plt.plot(layer_indices, wrong_values, marker="o", linewidth=2.2, color="#d04f3e", label="wrong")
    plt.xlabel("Layer")
    plt.ylabel("Mean tuned-lens logprob")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_probe_before_after(layer_indices, before_values, after_values, ylabel, title, out_path, zero_line):
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(
        layer_indices,
        before_values,
        marker="o",
        linewidth=2.2,
        linestyle="-",
        color="#d04f3e",
        label="before ablation",
    )
    plt.plot(
        layer_indices,
        after_values,
        marker="o",
        linewidth=2.2,
        linestyle="-",
        color="#2f6db3",
        label="after ablation",
    )
    plt.axhline(zero_line, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_tuned_lens_before_after(
    layer_indices,
    before_gold_values,
    before_wrong_values,
    after_gold_values,
    after_wrong_values,
    title,
    out_path,
):
    plt.figure(figsize=(8.8, 5.0))
    plt.plot(
        layer_indices,
        before_gold_values,
        marker="o",
        linewidth=2.0,
        linestyle="-",
        color="#2f6db3",
        label="before gold",
    )
    plt.plot(
        layer_indices,
        before_wrong_values,
        marker="o",
        linewidth=2.0,
        linestyle="-",
        color="#d04f3e",
        label="before wrong",
    )
    plt.plot(
        layer_indices,
        after_gold_values,
        marker="o",
        linewidth=2.2,
        linestyle="-",
        color="#1f77b4",
        label="after gold",
    )
    plt.plot(
        layer_indices,
        after_wrong_values,
        marker="o",
        linewidth=2.2,
        linestyle="-",
        color="#c62828",
        label="after wrong",
    )
    plt.xlabel("Layer")
    plt.ylabel("Mean tuned-lens logprob")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def collect_condition_outputs(
    samples,
    model,
    first_device,
    model_dtype,
    output_head,
    probe_specs,
    score_type,
    translators,
    lens_ckpt,
    image_size,
    batch_size,
):
    layer_indices = sorted(probe_specs.keys())
    probe_sums = [0.0 for _ in layer_indices]
    probe_count = 0

    lens_layer_indices = lens_ckpt["layer_indices"]
    gold_sums = [0.0 for _ in lens_layer_indices]
    wrong_sums = [0.0 for _ in lens_layer_indices]
    lens_count = 0

    lm_device = output_head.weight.device
    usable_samples = []
    for sample in samples:
        item = dict(sample)
        usable_samples.append(item)

    batches = list(chunked(usable_samples, batch_size))
    pbar = tqdm(batches, total=len(batches), desc="collect condition outputs", unit="batch", leave=False)
    for batch_rows in pbar:
        batch_inputs = lens_common.collate_processor_outputs(
            [sample["ctx_inputs"] for sample in batch_rows],
            pad_token_id=0,
        )
        batch_inputs.pop("token_type_ids", None)
        batch_inputs = lens_common.move_to_device(batch_inputs, first_device, model_dtype)
        positions = batch_inputs["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            # Match the original ablation script more closely: it ablates along the
            # prompt-cache path with use_cache=True. Some model implementations route
            # attention differently when cache is disabled.
            out = model(**batch_inputs, output_hidden_states=True, use_cache=True, return_dict=True)

        hidden_states = out.hidden_states[1:]
        past_key_values = out.past_key_values
        for pos, layer_idx in enumerate(layer_indices):
            hs_tensor = hidden_states[layer_idx]
            batch_index = torch.arange(len(batch_rows), device=hs_tensor.device)
            local_positions = positions.to(hs_tensor.device)
            gathered = hs_tensor[batch_index, local_positions].cpu()
            scores = apply_probe_batch(gathered, probe_specs[layer_idx], score_type)
            probe_sums[pos] += float(scores.sum().item())
        probe_count += len(batch_rows)

        for layer_pos, layer_idx in enumerate(lens_layer_indices):
            hs_tensor = hidden_states[layer_idx]
            batch_index = torch.arange(len(batch_rows), device=hs_tensor.device)
            local_positions = positions.to(hs_tensor.device)
            gathered = hs_tensor[batch_index, local_positions]
            translator = translators[str(layer_idx)]
            translator_dtype = next(translator.parameters()).dtype
            translator_device = next(translator.parameters()).device
            tuned_hidden = translator(gathered.to(device=translator_device, dtype=translator_dtype))
            tuned_logits = output_head(tuned_hidden.to(device=lm_device, dtype=output_head.weight.dtype))

            for item_idx, sample in enumerate(batch_rows):
                sample_past = tuple(
                    tuple(state[item_idx : item_idx + 1] for state in layer_states)
                    for layer_states in past_key_values
                )
                prompt_cache = {
                    "past_key_values": sample_past,
                    "prompt_last_logits": tuned_logits[item_idx : item_idx + 1],
                }
                gold_score = ablate_mod.score_continuation_with_cache(
                    model,
                    sample["_tokenizer"],
                    prompt_cache,
                    sample["gold"],
                    str(first_device),
                )
                wrong_score = ablate_mod.score_continuation_with_cache(
                    model,
                    sample["_tokenizer"],
                    prompt_cache,
                    sample["wrong"],
                    str(first_device),
                )
                gold_sums[layer_pos] += float(gold_score)
                wrong_sums[layer_pos] += float(wrong_score)
                if layer_pos == 0:
                    lens_count += 1
        pbar.set_postfix(probe_count=probe_count, lens_count=lens_count)

    probe_means = [value / max(1, probe_count) for value in probe_sums]
    gold_means = [value / max(1, lens_count) for value in gold_sums]
    wrong_means = [value / max(1, lens_count) for value in wrong_sums]
    return {
        "probe_layer_indices": layer_indices,
        "probe_mean_scores_by_layer": probe_means,
        "probe_count": probe_count,
        "lens_layer_indices": lens_layer_indices,
        "tuned_lens_mean_gold_logprob_by_layer": gold_means,
        "tuned_lens_mean_wrong_logprob_by_layer": wrong_means,
        "tuned_lens_count": lens_count,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze wrong->correct ablation flips with probe and tuned lens trajectories.")
    ap.add_argument("--ablation_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--lens_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_csv", default="./data/nc_cc_both_correct_rerun_tmp_val.csv")
    ap.add_argument("--selected_heads", default="")
    ap.add_argument("--position", default="")
    ap.add_argument("--score_type", default="logit", choices=["logit", "prob"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    ablation_payload, data_csv, selected_heads_path, position, image_root, trace_mode, keep_mode, max_image_side, ablation_config = resolve_ablation_meta(
        args.ablation_json,
        args.data_csv,
        args.selected_heads,
        args.position,
    )
    csv_rows = read_csv_rows(data_csv)

    lens_common.ensure_video_import_compat()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if args.device_map != "auto":
        device = torch.device(args.device_map)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_like_ablate_head(
        args.model,
        dtype=lens_modeling.resolve_dtype(args.dtype),
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_head = lens_common.get_output_head(model)
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    probe_specs = load_probe_specs(args.probe_dir)
    lens_ckpt, translators = load_tuned_lens(args.lens_ckpt, first_device)
    tokenizer = processor.tokenizer

    ablate_args = build_ablate_args(
        data_csv=data_csv,
        image_root=image_root,
        position=position,
        trace_mode=trace_mode,
        max_image_side=max_image_side,
    )
    df = ablate_mod.pd.read_csv(data_csv)
    built_samples = ablate_mod.build_samples(ablate_args, df, processor, tokenizer, model, str(first_device))
    for sample in built_samples:
        sample["_tokenizer"] = tokenizer
    selected_samples = select_flip_samples_from_built(ablation_payload, built_samples)
    if args.max_samples > 0:
        selected_samples = selected_samples[: args.max_samples]
    if not selected_samples:
        raise RuntimeError("No wrong->correct samples found after aligning ablation results with built samples.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_path_check = verify_ablation_changes_original_path(
        model=model,
        sample=selected_samples[0],
        selected_heads_path=selected_heads_path,
        keep_mode=keep_mode,
        trace_mode=trace_mode,
    )
    if original_path_check["max_abs_score_diff"] == 0.0:
        raise RuntimeError(
            "Ablation is already a no-op on the original ablate_head.py scoring path in this environment. "
            "This likely means the current Hulu-Med/transformers runtime does not expose the same attention-mask "
            "hooking path as the environment that produced the saved ablation JSON."
        )

    before_summary = collect_condition_outputs(
        samples=selected_samples,
        model=model,
        first_device=first_device,
        model_dtype=model_dtype,
        output_head=output_head,
        probe_specs=probe_specs,
        score_type=args.score_type,
        translators=translators,
        lens_ckpt=lens_ckpt,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    selected_pairs = ablate_mod.parse_selected_heads(selected_heads_path)
    layer_to_heads = ablate_mod.pack_layer_to_heads(selected_pairs)
    handles = ablate_mod.install_head_mask_hooks(model, layer_to_heads, keep_mode=keep_mode)
    try:
        after_summary = collect_condition_outputs(
            samples=selected_samples,
            model=model,
            first_device=first_device,
            model_dtype=model_dtype,
            output_head=output_head,
            probe_specs=probe_specs,
            score_type=args.score_type,
            translators=translators,
            lens_ckpt=lens_ckpt,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
    finally:
        for handle in handles:
            handle.remove()

    diff_summary = summarize_ablation_effect(before_summary, after_summary)
    if (
        diff_summary["probe_max_abs_diff"] == 0.0
        and diff_summary["gold_max_abs_diff"] == 0.0
        and diff_summary["wrong_max_abs_diff"] == 0.0
    ):
        raise RuntimeError(
            "Ablation had no measurable effect in this analysis path: before/after probe and "
            "tuned-lens trajectories are exactly identical. The hooks likely did not affect the "
            "current forward path. Please verify the hook target or model forward mode."
        )

    probe_compare_png = out_dir / f"probe_before_after_ablation_{args.score_type}.png"
    lens_before_png = out_dir / "tuned_lens_before_ablation.png"
    lens_after_png = out_dir / "tuned_lens_after_ablation.png"

    zero_line = 0.5 if args.score_type == "prob" else 0.0
    ylabel = "Mean probe probability" if args.score_type == "prob" else "Mean probe logit"
    plot_probe_before_after(
        before_summary["probe_layer_indices"],
        before_summary["probe_mean_scores_by_layer"],
        after_summary["probe_mean_scores_by_layer"],
        ylabel=ylabel,
        title=f"Probe before vs after ablation ({args.score_type})\nwrong->correct samples (n={before_summary['probe_count']})",
        out_path=probe_compare_png,
        zero_line=zero_line,
    )
    plot_gold_wrong(
        before_summary["lens_layer_indices"],
        before_summary["tuned_lens_mean_gold_logprob_by_layer"],
        before_summary["tuned_lens_mean_wrong_logprob_by_layer"],
        title=f"Tuned lens before ablation\nwrong->correct samples (n={before_summary['tuned_lens_count']})",
        out_path=lens_before_png,
    )
    plot_gold_wrong(
        after_summary["lens_layer_indices"],
        after_summary["tuned_lens_mean_gold_logprob_by_layer"],
        after_summary["tuned_lens_mean_wrong_logprob_by_layer"],
        title=f"Tuned lens after ablation\nwrong->correct samples (n={after_summary['tuned_lens_count']})",
        out_path=lens_after_png,
    )

    payload = {
        "ablation_json": args.ablation_json,
        "data_csv": data_csv,
        "selected_heads": selected_heads_path,
        "position": position,
        "keep_mode": keep_mode,
        "score_type": args.score_type,
        "n_selected_heads": len(selected_pairs),
        "n_installed_hooks": len(handles),
        "original_path_check": original_path_check,
        "n_flip_correct_samples": len(selected_samples),
        "selected_samples": [
            {
                "row_idx": sample["row_idx"],
                "question": sample["question"],
                "gold": sample["gold"],
                "wrong": sample["wrong"],
                "base_ctx_pred": sample["base_ctx_pred"],
                "ab_ctx_pred_from_json": sample["ablation_record"]["ab_ctx_pred"],
            }
            for sample in selected_samples
        ],
        "before_ablation": before_summary,
        "after_ablation": after_summary,
        "ablation_effect_summary": diff_summary,
        "saved_plots": {
            "probe_before_after": str(probe_compare_png),
            "tuned_lens_before": str(lens_before_png),
            "tuned_lens_after": str(lens_after_png),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_probe_before_after={probe_compare_png}")
    print(f"saved_tuned_lens_before={lens_before_png}")
    print(f"saved_tuned_lens_after={lens_after_png}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
