import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import read_jsonl, wrap_as_chat
from modeling import build_translators, resolve_dtype


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


def safe_slug(text):
    cleaned = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "record"


def resolve_yes_no_token_ids(tokenizer):
    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError('" Yes"/" No" must be single-token for this trajectory analyzer.')
    return {"yes": yes_ids[0], "no": no_ids[0]}


def trajectory_stats(deltas, margin_eps):
    signs = []
    for delta in deltas:
        if delta > margin_eps:
            signs.append(1)
        elif delta < -margin_eps:
            signs.append(-1)
        else:
            signs.append(0)

    onset = next((idx for idx, delta in enumerate(deltas) if abs(delta) > margin_eps), -1)

    flip = -1
    seen_gold = False
    for idx, sign in enumerate(signs):
        if sign < 0:
            seen_gold = True
        if sign > 0 and seen_gold:
            flip = idx
            break
    if flip == -1 and signs and signs[0] > 0:
        flip = 0

    non_zero = [sign for sign in signs if sign != 0]
    flip_count = 0
    for prev, cur in zip(non_zero, non_zero[1:]):
        if prev != cur:
            flip_count += 1

    stabilization = -1
    final_sign = 0
    for sign in reversed(signs):
        if sign != 0:
            final_sign = sign
            break
    if final_sign != 0:
        for idx in range(len(signs)):
            tail = [sign for sign in signs[idx:] if sign != 0]
            if tail and all(sign == final_sign for sign in tail):
                stabilization = idx
                break

    return {
        "onset_layer": onset,
        "flip_layer": flip,
        "stabilization_layer": stabilization,
        "flip_count": flip_count,
    }

def plot_delta_trajectory(layer_indices, raw_deltas, tuned_deltas, out_path, title):
    x = [layer + 1 for layer in layer_indices]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(x, raw_deltas, color="#ef553b", marker="s", linewidth=1.6, markersize=4.2, label="Raw delta")
    ax.plot(x, tuned_deltas, color="#1f77ff", marker="o", linewidth=1.6, markersize=4.0, label="Tuned delta")
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Delta")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_answer_logit_trajectory(
    layer_indices,
    raw_gold_logits,
    raw_wrong_logits,
    tuned_gold_logits,
    tuned_wrong_logits,
    out_path,
    title,
):
    x = [layer + 1 for layer in layer_indices]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(x, raw_gold_logits, color="#2ca02c", linestyle="--", marker="s", linewidth=1.5, markersize=3.8, label="Raw gold")
    ax.plot(x, raw_wrong_logits, color="#ff7f0e", linestyle="--", marker="s", linewidth=1.5, markersize=3.8, label="Raw wrong")
    ax.plot(x, tuned_gold_logits, color="#1f77b4", linestyle="-", marker="o", linewidth=1.6, markersize=3.8, label="Tuned gold")
    ax.plot(x, tuned_wrong_logits, color="#d62728", linestyle="-", marker="o", linewidth=1.6, markersize=3.8, label="Tuned wrong")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Logit")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare raw logit lens and tuned lens trajectories.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="conflict")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--margin_eps", type=float, default=0.1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--plot_max_records", type=int, default=20)
    args = ap.parse_args()

    rows = filter_rows(read_jsonl(args.manifest), args.prompt_types)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    ckpt = torch.load(args.lens_ckpt, map_location="cpu")
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
        raise RuntimeError("Model does not expose output embeddings.")

    first_device = next(model.parameters()).device
    lens_device = pick_device(args.lens_device, first_device)
    translators = build_translators(
        layer_indices=ckpt["layer_indices"],
        hidden_dim=ckpt["hidden_dim"],
        dtype=resolve_dtype(ckpt["translator_dtype"]),
        rank=ckpt["translator_rank"],
    ).to(lens_device)
    translators.load_state_dict(ckpt["state_dict"], strict=True)
    translators.eval()

    label_token_ids = resolve_yes_no_token_ids(tok)
    batch_records = []
    group_summary = {}
    raw_sums = [0.0 for _ in ckpt["layer_indices"]]
    tuned_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_raw_gold_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_raw_wrong_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_tuned_gold_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_tuned_wrong_logit_sums = [0.0 for _ in ckpt["layer_indices"]]

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
        logits = out.logits
        logits_batch_index = torch.arange(len(batch_rows), device=logits.device)
        logits_positions = positions.to(logits.device)
        final_logits = logits[logits_batch_index, logits_positions]
        final_log_probs = torch.log_softmax(final_logits, dim=-1)
        lm_device = output_head.weight.device
        gathered_states = {}
        for layer_idx in ckpt["layer_indices"]:
            hs_tensor = hidden_states[layer_idx]
            batch_index = torch.arange(len(batch_rows), device=hs_tensor.device)
            local_positions = positions.to(hs_tensor.device)
            gathered_states[layer_idx] = hs_tensor[batch_index, local_positions]

        for item_idx, row in enumerate(batch_rows):
            gold_id = label_token_ids[row["gold_answer"]]
            wrong_id = label_token_ids[row["wrong_answer"]]
            raw_deltas = []
            tuned_deltas = []
            raw_gold_logits = []
            raw_wrong_logits = []
            tuned_gold_logits = []
            tuned_wrong_logits = []

            for pos, layer_idx in enumerate(ckpt["layer_indices"]):
                hs = gathered_states[layer_idx][item_idx]
                raw_logits = output_head(hs.to(device=lm_device, dtype=output_head.weight.dtype))
                raw_log_probs = torch.log_softmax(raw_logits, dim=-1)
                raw_gold_logit = raw_logits[gold_id].item()
                raw_wrong_logit = raw_logits[wrong_id].item()
                raw_delta = (raw_log_probs[wrong_id] - raw_log_probs[gold_id]).item()
                raw_deltas.append(raw_delta)
                raw_gold_logits.append(raw_gold_logit)
                raw_wrong_logits.append(raw_wrong_logit)

                translator = translators[str(layer_idx)]
                translator_dtype = next(translator.parameters()).dtype
                tuned_hidden = translator(
                    hs.to(device=lens_device, dtype=translator_dtype).unsqueeze(0)
                ).squeeze(0)
                tuned_logits = output_head(
                    tuned_hidden.to(device=lm_device, dtype=output_head.weight.dtype)
                )
                tuned_log_probs = torch.log_softmax(tuned_logits, dim=-1)
                tuned_gold_logit = tuned_logits[gold_id].item()
                tuned_wrong_logit = tuned_logits[wrong_id].item()
                tuned_delta = (tuned_log_probs[wrong_id] - tuned_log_probs[gold_id]).item()
                tuned_deltas.append(tuned_delta)
                tuned_gold_logits.append(tuned_gold_logit)
                tuned_wrong_logits.append(tuned_wrong_logit)

                raw_sums[pos] += raw_delta
                tuned_sums[pos] += tuned_delta
                overall_raw_gold_logit_sums[pos] += raw_gold_logit
                overall_raw_wrong_logit_sums[pos] += raw_wrong_logit
                overall_tuned_gold_logit_sums[pos] += tuned_gold_logit
                overall_tuned_wrong_logit_sums[pos] += tuned_wrong_logit

            final_delta = (
                final_log_probs[item_idx, wrong_id] - final_log_probs[item_idx, gold_id]
            ).item()
            raw_stats = trajectory_stats(raw_deltas, args.margin_eps)
            tuned_stats = trajectory_stats(tuned_deltas, args.margin_eps)

            record = {
                "record_id": row["record_id"],
                "pair_id": row["pair_id"],
                "side": row["side"],
                "prompt_type": row["prompt_type"],
                "follow_label": row["follow_label"],
                "gold_answer": row["gold_answer"],
                "wrong_answer": row["wrong_answer"],
                "final_delta": final_delta,
                "raw_delta_by_layer": raw_deltas,
                "tuned_delta_by_layer": tuned_deltas,
                "raw_gold_logit_by_layer": raw_gold_logits,
                "raw_wrong_logit_by_layer": raw_wrong_logits,
                "tuned_gold_logit_by_layer": tuned_gold_logits,
                "tuned_wrong_logit_by_layer": tuned_wrong_logits,
                "raw_stats": raw_stats,
                "tuned_stats": tuned_stats,
            }
            batch_records.append(record)

            group = group_summary.setdefault(
                row["follow_label"],
                {
                    "count": 0,
                    "raw_sum_delta_by_layer": [0.0 for _ in raw_deltas],
                    "tuned_sum_delta_by_layer": [0.0 for _ in tuned_deltas],
                    "raw_sum_gold_logit_by_layer": [0.0 for _ in raw_gold_logits],
                    "raw_sum_wrong_logit_by_layer": [0.0 for _ in raw_wrong_logits],
                    "tuned_sum_gold_logit_by_layer": [0.0 for _ in tuned_gold_logits],
                    "tuned_sum_wrong_logit_by_layer": [0.0 for _ in tuned_wrong_logits],
                    "raw_stats": {"onset_layer": 0.0, "flip_layer": 0.0, "stabilization_layer": 0.0, "flip_count": 0.0},
                    "tuned_stats": {"onset_layer": 0.0, "flip_layer": 0.0, "stabilization_layer": 0.0, "flip_count": 0.0},
                },
            )
            group["count"] += 1
            for idx, value in enumerate(raw_deltas):
                group["raw_sum_delta_by_layer"][idx] += value
            for idx, value in enumerate(tuned_deltas):
                group["tuned_sum_delta_by_layer"][idx] += value
            for idx, value in enumerate(raw_gold_logits):
                group["raw_sum_gold_logit_by_layer"][idx] += value
            for idx, value in enumerate(raw_wrong_logits):
                group["raw_sum_wrong_logit_by_layer"][idx] += value
            for idx, value in enumerate(tuned_gold_logits):
                group["tuned_sum_gold_logit_by_layer"][idx] += value
            for idx, value in enumerate(tuned_wrong_logits):
                group["tuned_sum_wrong_logit_by_layer"][idx] += value
            for key, value in raw_stats.items():
                group["raw_stats"][key] += value
            for key, value in tuned_stats.items():
                group["tuned_stats"][key] += value

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectories.jsonl"
    with traj_path.open("w", encoding="utf-8") as f:
        for row in batch_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_records = max(1, len(batch_records))
    for group in group_summary.values():
        count = max(1, group["count"])
        group["raw_mean_delta_by_layer"] = [value / count for value in group["raw_sum_delta_by_layer"]]
        group["tuned_mean_delta_by_layer"] = [value / count for value in group["tuned_sum_delta_by_layer"]]
        group["raw_mean_gold_logit_by_layer"] = [value / count for value in group["raw_sum_gold_logit_by_layer"]]
        group["raw_mean_wrong_logit_by_layer"] = [value / count for value in group["raw_sum_wrong_logit_by_layer"]]
        group["tuned_mean_gold_logit_by_layer"] = [value / count for value in group["tuned_sum_gold_logit_by_layer"]]
        group["tuned_mean_wrong_logit_by_layer"] = [value / count for value in group["tuned_sum_wrong_logit_by_layer"]]
        group["raw_mean_stats"] = {key: value / count for key, value in group["raw_stats"].items()}
        group["tuned_mean_stats"] = {key: value / count for key, value in group["tuned_stats"].items()}
        del group["raw_sum_delta_by_layer"]
        del group["tuned_sum_delta_by_layer"]
        del group["raw_sum_gold_logit_by_layer"]
        del group["raw_sum_wrong_logit_by_layer"]
        del group["tuned_sum_gold_logit_by_layer"]
        del group["tuned_sum_wrong_logit_by_layer"]
        del group["raw_stats"]
        del group["tuned_stats"]

    plots_dir = out_dir / "plots"
    delta_dir = plots_dir / "delta"
    logit_dir = plots_dir / "logits"
    plot_count = min(len(batch_records), max(0, args.plot_max_records))
    for record in batch_records[:plot_count]:
        slug = safe_slug(f"{record['record_id']}_{record['prompt_type']}_{record['follow_label']}")
        title_prefix = f"{record['record_id']} | {record['prompt_type']} | {record['follow_label']}"
        plot_delta_trajectory(
            layer_indices=ckpt["layer_indices"],
            raw_deltas=record["raw_delta_by_layer"],
            tuned_deltas=record["tuned_delta_by_layer"],
            out_path=delta_dir / f"{slug}.png",
            title=f"Delta Trajectory\n{title_prefix}",
        )
        plot_answer_logit_trajectory(
            layer_indices=ckpt["layer_indices"],
            raw_gold_logits=record["raw_gold_logit_by_layer"],
            raw_wrong_logits=record["raw_wrong_logit_by_layer"],
            tuned_gold_logits=record["tuned_gold_logit_by_layer"],
            tuned_wrong_logits=record["tuned_wrong_logit_by_layer"],
            out_path=logit_dir / f"{slug}.png",
            title=f"Gold/Wrong Logit Trajectory\n{title_prefix}",
        )

    group_delta_dir = plots_dir / "group_mean_delta"
    group_logit_dir = plots_dir / "group_mean_logits"
    saved_group_delta_plots = {}
    saved_group_logit_plots = {}
    for follow_label, group in group_summary.items():
        slug = safe_slug(follow_label)
        group_delta_path = group_delta_dir / f"{slug}.png"
        plot_delta_trajectory(
            layer_indices=ckpt["layer_indices"],
            raw_deltas=group["raw_mean_delta_by_layer"],
            tuned_deltas=group["tuned_mean_delta_by_layer"],
            out_path=group_delta_path,
            title=f"Mean Delta Trajectory\n{follow_label}",
        )
        saved_group_delta_plots[follow_label] = str(group_delta_path)

        group_logit_path = group_logit_dir / f"{slug}.png"
        plot_answer_logit_trajectory(
            layer_indices=ckpt["layer_indices"],
            raw_gold_logits=group["raw_mean_gold_logit_by_layer"],
            raw_wrong_logits=group["raw_mean_wrong_logit_by_layer"],
            tuned_gold_logits=group["tuned_mean_gold_logit_by_layer"],
            tuned_wrong_logits=group["tuned_mean_wrong_logit_by_layer"],
            out_path=group_logit_path,
            title=f"Mean Gold/Wrong Logit Trajectory\n{follow_label}",
        )
        saved_group_logit_plots[follow_label] = str(group_logit_path)

    summary = {
        "manifest": args.manifest,
        "lens_ckpt": args.lens_ckpt,
        "prompt_types": args.prompt_types,
        "n_records": len(batch_records),
        "layer_indices": ckpt["layer_indices"],
        "overall_raw_mean_delta_by_layer": [value / max(1, n_records) for value in raw_sums],
        "overall_tuned_mean_delta_by_layer": [value / max(1, n_records) for value in tuned_sums],
        "overall_raw_mean_gold_logit_by_layer": [value / max(1, n_records) for value in overall_raw_gold_logit_sums],
        "overall_raw_mean_wrong_logit_by_layer": [value / max(1, n_records) for value in overall_raw_wrong_logit_sums],
        "overall_tuned_mean_gold_logit_by_layer": [value / max(1, n_records) for value in overall_tuned_gold_logit_sums],
        "overall_tuned_mean_wrong_logit_by_layer": [value / max(1, n_records) for value in overall_tuned_wrong_logit_sums],
        "by_follow_label": group_summary,
        "saved_trajectories": str(traj_path),
        "saved_delta_plot_dir": str(delta_dir),
        "saved_logit_plot_dir": str(logit_dir),
        "saved_group_mean_delta_plots": saved_group_delta_plots,
        "saved_group_mean_logit_plots": saved_group_logit_plots,
        "plot_max_records": args.plot_max_records,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"saved_trajectories={traj_path}")
    print(f"saved_summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
