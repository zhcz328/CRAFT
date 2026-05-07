import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR / "probe") not in sys.path:
    sys.path.append(str(ROOT_DIR / "probe"))
if str(ROOT_DIR / "tuned_lens") not in sys.path:
    sys.path.append(str(ROOT_DIR / "tuned_lens"))

import ablate_head_inf as ablate_mod
import modeling as lens_modeling


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


def ensure_padding_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token or eos_token to use for padding.")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_probe_specs(probe_dir):
    probe_specs = {}
    for layer_path in sorted(Path(probe_dir).glob("layer_*.pt")):
        ckpt = torch.load(layer_path, map_location="cpu")
        if ckpt["probe_type"] == "linear":
            model = LinearProbe(ckpt["mean"].shape[-1])
        elif ckpt["probe_type"] == "mlp":
            hidden_dim = ckpt["state_dict"]["net.0.weight"].shape[0]
            model = MLPProbe(ckpt["mean"].shape[-1], hidden_dim)
        else:
            raise ValueError(f"Unknown probe_type: {ckpt['probe_type']}")
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


def apply_probe_batch(hidden_batch, probe_spec, score_type):
    x = (hidden_batch.float() - probe_spec["mean"]) / probe_spec["std"]
    with torch.no_grad():
        logits = probe_spec["model"](x).cpu()
    if score_type == "logit":
        return logits
    return torch.sigmoid(logits)


def resolve_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError('" Yes"/" No" must be single-token for this analyzer.')
    return {"yes": yes_ids[0], "no": no_ids[0]}


def load_ablation_records(path):
    rows = read_jsonl(path)
    meta = {}
    records = []
    for row in rows:
        if "_meta" in row:
            meta = row["_meta"]
            continue
        records.append(row)
    return meta, records


def build_baseline_map(path):
    baseline = {}
    for row in read_jsonl(path):
        key = (int(row["pair_id"]), str(row["side"]), str(row["position"]))
        baseline[key] = row
    return baseline


def build_pairs_map(path):
    pairs = {}
    for row in read_jsonl(path):
        pairs[int(row["pair_id"])] = row
    return pairs


def extract_pred(record, *field_names):
    for field_name in field_names:
        node = record.get(field_name)
        if isinstance(node, dict) and "pred" in node:
            return str(node["pred"])
    raise KeyError(f"None of {field_names} found with a 'pred' field in record keys={list(record.keys())}")


def resolve_head_groups_path(args_head_groups, meta_head_groups_path):
    candidates = [args_head_groups, meta_head_groups_path]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return str(candidate_path)

        # The metadata often stores an absolute path from another machine.
        # If the tail points into this repo's result/ tree, map it back locally.
        parts = candidate_path.parts
        if "result" in parts:
            local_path = ROOT_DIR / Path(*parts[parts.index("result") :])
            if local_path.exists():
                return str(local_path)
            if local_path.name == "head_groups.json":
                selected_path = local_path.with_name("selected_heads.json")
                if selected_path.exists():
                    return str(selected_path)

    # Final fallback for the current llama32_3b experiment layout.
    for rel_path in (
        ROOT_DIR / "llama32_3b" / "result" / "before_question" / "headscan_rounds_top30_inf" / "selected_heads.json",
        ROOT_DIR / "llama32_3b" / "result" / "before_answer" / "headscan_rounds_top30_inf" / "selected_heads.json",
        ROOT_DIR / "llama32_3b" / "result" / "prefix" / "headscan_rounds_top30_inf" / "selected_heads.json",
        ROOT_DIR / "result" / "before_question" / "headscan_rounds_top30_inf" / "selected_heads.json",
        ROOT_DIR / "result" / "before_answer" / "headscan_rounds_top30_inf" / "selected_heads.json",
        ROOT_DIR / "result" / "prefix" / "headscan_rounds_top30_inf" / "selected_heads.json",
    ):
        if rel_path.exists():
            return str(rel_path)

    return ""


def select_flip_samples(ablation_records, baseline_map, pairs_map, position):
    selected = []
    for row in ablation_records:
        if str(row.get("position")) != position:
            continue
        key = (int(row["pair_id"]), str(row["side"]), str(row["position"]))
        baseline_row = baseline_map.get(key)
        pair_row = pairs_map.get(int(row["pair_id"]))
        if baseline_row is None or pair_row is None:
            continue
        baseline_conflict_pred = extract_pred(baseline_row, "conflict", "base")
        ablated_conflict_pred = extract_pred(row, "conflict_after", "conflict", "base_after", "masked")
        gold = str(row["gold"])
        if baseline_conflict_pred == gold:
            continue
        if ablated_conflict_pred != gold:
            continue

        side = str(row["side"])
        base_prompt = pair_row[side]["prompt"]
        gold_yes = gold == "yes"
        selected.append(
            {
                "pair_id": int(row["pair_id"]),
                "side": side,
                "position": position,
                "gold_answer": gold,
                "wrong_answer": str(row["conflict_label"]),
                "base_prompt": base_prompt,
                "baseline_conflict_pred": baseline_conflict_pred,
                "ablated_conflict_pred": ablated_conflict_pred,
                "gold_yes": gold_yes,
            }
        )
    return selected


def build_prompt_text(base_prompt, side, position):
    gold_yes = side == "correct"
    conflict_yes = not gold_yes
    conflict_user = ablate_mod.inject_evidence(base_prompt, ablate_mod.make_evidence(conflict_yes), position)
    return conflict_user


def collect_condition_outputs(
    samples,
    tokenizer,
    model,
    probe_specs,
    score_type,
    lens_ckpt,
    translators,
    batch_size,
    enable_thinking,
):
    probe_layer_indices = sorted(probe_specs.keys())
    lens_layer_indices = list(lens_ckpt["layer_indices"])
    probe_sums = [0.0 for _ in probe_layer_indices]
    gold_sums = [0.0 for _ in lens_layer_indices]
    wrong_sums = [0.0 for _ in lens_layer_indices]
    sample_count = 0

    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("Model does not expose output embeddings.")
    lm_device = output_head.weight.device
    first_device = next(model.parameters()).device
    label_token_ids = resolve_yes_no_token_ids(tokenizer)

    for batch_rows in chunked(samples, batch_size):
        prompts = [
            ablate_mod.wrap_as_chat(
                tokenizer,
                build_prompt_text(sample["base_prompt"], sample["side"], sample["position"]),
                enable_thinking,
            )
            for sample in batch_rows
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        positions = enc["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        hidden_states = out.hidden_states[1:]
        batch_index = torch.arange(len(batch_rows), device=first_device)

        for idx, layer_idx in enumerate(probe_layer_indices):
            hs_tensor = hidden_states[layer_idx]
            gathered = hs_tensor[batch_index, positions.to(hs_tensor.device)].detach().cpu()
            scores = apply_probe_batch(gathered, probe_specs[layer_idx], score_type)
            probe_sums[idx] += float(scores.sum().item())

        for idx, layer_idx in enumerate(lens_layer_indices):
            hs_tensor = hidden_states[layer_idx]
            gathered = hs_tensor[batch_index, positions.to(hs_tensor.device)]
            translator = translators[str(layer_idx)]
            translator_dtype = next(translator.parameters()).dtype
            translator_device = next(translator.parameters()).device
            tuned_hidden = translator(gathered.to(device=translator_device, dtype=translator_dtype))
            tuned_logits = output_head(tuned_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
            tuned_log_probs = torch.log_softmax(tuned_logits, dim=-1)

            for item_idx, sample in enumerate(batch_rows):
                gold_id = label_token_ids[sample["gold_answer"]]
                wrong_id = label_token_ids[sample["wrong_answer"]]
                gold_sums[idx] += float(tuned_log_probs[item_idx, gold_id].item())
                wrong_sums[idx] += float(tuned_log_probs[item_idx, wrong_id].item())

        sample_count += len(batch_rows)

    return {
        "probe_layer_indices": probe_layer_indices,
        "probe_mean_scores_by_layer": [value / max(1, sample_count) for value in probe_sums],
        "probe_count": sample_count,
        "lens_layer_indices": lens_layer_indices,
        "tuned_lens_mean_gold_logprob_by_layer": [value / max(1, sample_count) for value in gold_sums],
        "tuned_lens_mean_wrong_logprob_by_layer": [value / max(1, sample_count) for value in wrong_sums],
        "tuned_lens_count": sample_count,
    }


def summarize_ablation_effect(before_summary, after_summary):
    probe_diff = max(
        abs(float(a) - float(b))
        for a, b in zip(before_summary["probe_mean_scores_by_layer"], after_summary["probe_mean_scores_by_layer"])
    )
    gold_diff = max(
        abs(float(a) - float(b))
        for a, b in zip(
            before_summary["tuned_lens_mean_gold_logprob_by_layer"],
            after_summary["tuned_lens_mean_gold_logprob_by_layer"],
        )
    )
    wrong_diff = max(
        abs(float(a) - float(b))
        for a, b in zip(
            before_summary["tuned_lens_mean_wrong_logprob_by_layer"],
            after_summary["tuned_lens_mean_wrong_logprob_by_layer"],
        )
    )
    return {
        "probe_max_abs_diff": probe_diff,
        "gold_max_abs_diff": gold_diff,
        "wrong_max_abs_diff": wrong_diff,
    }


def plot_probe_before_after(layer_indices, before_values, after_values, ylabel, title, out_path, zero_line):
    plt.figure(figsize=(8.4, 4.8))
    plt.plot(layer_indices, before_values, marker="o", linewidth=2.2, color="#d04f3e", label="before ablation")
    plt.plot(layer_indices, after_values, marker="o", linewidth=2.2, color="#2f6db3", label="after ablation")
    plt.axhline(zero_line, linestyle="--", linewidth=1, color="gray", alpha=0.7)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
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


def main():
    ap = argparse.ArgumentParser(description="Analyze wrong->correct ablation flips with probe and tuned lens trajectories.")
    ap.add_argument("--ablation_json", required=True)
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--lens_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--head_groups", default="")
    ap.add_argument("--position", default="")
    ap.add_argument("--score_type", default="logit", choices=["logit", "prob"])
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--enable_thinking", action="store_true")
    args = ap.parse_args()

    meta, ablation_records = load_ablation_records(args.ablation_json)
    if not args.position:
        positions = meta.get("positions") or []
        if len(positions) == 1:
            args.position = str(positions[0])
        elif ablation_records:
            args.position = str(ablation_records[0]["position"])
        else:
            raise RuntimeError("Cannot infer --position from ablation json.")

    baseline_map = build_baseline_map(args.baseline_json)
    pairs_map = build_pairs_map(args.pairs)
    selected_samples = select_flip_samples(ablation_records, baseline_map, pairs_map, args.position)
    if args.max_samples > 0:
        selected_samples = selected_samples[: args.max_samples]
    if not selected_samples:
        raise RuntimeError("No wrong->correct flip samples found for the given ablation/baseline pair.")

    tokenizer = ensure_padding_token(AutoTokenizer.from_pretrained(args.model, trust_remote_code=True))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=lens_modeling.resolve_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation="eager",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    probe_specs = load_probe_specs(args.probe_dir)
    first_device = next(model.parameters()).device
    lens_ckpt, translators = load_tuned_lens(args.lens_ckpt, first_device)

    before_summary = collect_condition_outputs(
        samples=selected_samples,
        tokenizer=tokenizer,
        model=model,
        probe_specs=probe_specs,
        score_type=args.score_type,
        lens_ckpt=lens_ckpt,
        translators=translators,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking,
    )

    head_groups_path = resolve_head_groups_path(args.head_groups, meta.get("head_groups_path", ""))
    if not head_groups_path:
        raise RuntimeError("Cannot resolve head_groups path from args or ablation metadata.")
    loader = getattr(ablate_mod, "load_conflict_specific_heads", None)
    if loader is None:
        loader = getattr(ablate_mod, "load_ablation_heads", None)
    if loader is None:
        raise RuntimeError("Could not find a head loader in ablate_head_inf.py.")
    layer2heads, _ = loader(head_groups_path)
    handles = ablate_mod.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_summary = collect_condition_outputs(
            samples=selected_samples,
            tokenizer=tokenizer,
            model=model,
            probe_specs=probe_specs,
            score_type=args.score_type,
            lens_ckpt=lens_ckpt,
            translators=translators,
            batch_size=args.batch_size,
            enable_thinking=args.enable_thinking,
        )
    finally:
        ablate_mod.remove_hooks(handles)

    diff_summary = summarize_ablation_effect(before_summary, after_summary)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "baseline_json": args.baseline_json,
        "pairs": args.pairs,
        "head_groups": head_groups_path,
        "position": args.position,
        "score_type": args.score_type,
        "n_flip_correct_samples": len(selected_samples),
        "selected_samples": [
            {
                "pair_id": sample["pair_id"],
                "side": sample["side"],
                "gold_answer": sample["gold_answer"],
                "wrong_answer": sample["wrong_answer"],
                "baseline_conflict_pred": sample["baseline_conflict_pred"],
                "ablated_conflict_pred": sample["ablated_conflict_pred"],
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
