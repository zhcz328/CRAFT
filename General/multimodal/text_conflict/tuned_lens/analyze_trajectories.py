import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import (
    answer_token_ids,
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


def infer_position_from_manifest(path_str: str) -> str:
    stem = Path(path_str).stem
    for candidate in ("before_question", "before_answer", "prefix"):
        if stem.endswith(candidate):
            return candidate
    return ""


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


def accumulate_trajectory_record(
    record,
    group_summary,
    raw_sums,
    tuned_sums,
    overall_raw_gold_logit_sums,
    overall_raw_wrong_logit_sums,
    overall_tuned_gold_logit_sums,
    overall_tuned_wrong_logit_sums,
):
    raw_deltas = record["raw_delta_by_layer"]
    tuned_deltas = record["tuned_delta_by_layer"]
    raw_gold_logits = record["raw_gold_logit_by_layer"]
    raw_wrong_logits = record["raw_wrong_logit_by_layer"]
    tuned_gold_logits = record["tuned_gold_logit_by_layer"]
    tuned_wrong_logits = record["tuned_wrong_logit_by_layer"]
    raw_stats = record["raw_stats"]
    tuned_stats = record["tuned_stats"]
    for idx, value in enumerate(raw_deltas):
        raw_sums[idx] += value
    for idx, value in enumerate(tuned_deltas):
        tuned_sums[idx] += value
    for idx, value in enumerate(raw_gold_logits):
        overall_raw_gold_logit_sums[idx] += value
    for idx, value in enumerate(raw_wrong_logits):
        overall_raw_wrong_logit_sums[idx] += value
    for idx, value in enumerate(tuned_gold_logits):
        overall_tuned_gold_logit_sums[idx] += value
    for idx, value in enumerate(tuned_wrong_logits):
        overall_tuned_wrong_logit_sums[idx] += value

    group = group_summary.setdefault(
        record["follow_label"],
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


def append_answer_tokens(batch_inputs, answer_token_lists, pad_token_id):
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    prompt_lens = attention_mask.sum(dim=1).tolist()

    extended_input_ids = []
    extended_attention_masks = []
    max_len = 0
    for idx in range(batch_size):
        prompt_len = int(prompt_lens[idx])
        answer_ids = list(answer_token_lists[idx])
        prompt_ids = input_ids[idx, :prompt_len]
        answer_tensor = torch.tensor(answer_ids, dtype=input_ids.dtype)
        extended_ids = torch.cat([prompt_ids, answer_tensor], dim=0)
        extended_mask = torch.ones(extended_ids.shape[0], dtype=attention_mask.dtype)
        extended_input_ids.append(extended_ids)
        extended_attention_masks.append(extended_mask)
        max_len = max(max_len, int(extended_ids.shape[0]))

    padded_input_ids = []
    padded_attention_masks = []
    for ids, mask in zip(extended_input_ids, extended_attention_masks):
        pad_len = max_len - int(ids.shape[0])
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)], dim=0)
        padded_input_ids.append(ids)
        padded_attention_masks.append(mask)

    extended_batch = {}
    for key, value in batch_inputs.items():
        if key == "input_ids":
            extended_batch[key] = torch.stack(padded_input_ids, dim=0)
        elif key == "attention_mask":
            extended_batch[key] = torch.stack(padded_attention_masks, dim=0)
        else:
            extended_batch[key] = value
    return extended_batch


def answer_score_from_logits(logits_seq, prompt_len, answer_ids):
    pred_positions = torch.arange(
        prompt_len - 1,
        prompt_len - 1 + len(answer_ids),
        device=logits_seq.device,
    )
    answer_index = torch.tensor(answer_ids, dtype=torch.long, device=logits_seq.device)
    selected_logits = logits_seq[pred_positions, answer_index]
    selected_log_probs = torch.log_softmax(logits_seq[pred_positions], dim=-1)[
        torch.arange(len(answer_ids), device=logits_seq.device),
        answer_index,
    ]
    return float(selected_log_probs.sum().item()), float(selected_logits.sum().item())


def answer_score_from_hidden(hidden_seq, prompt_len, answer_ids, output_head, lm_device):
    pred_positions = torch.arange(
        prompt_len - 1,
        prompt_len - 1 + len(answer_ids),
        device=hidden_seq.device,
    )
    pred_hidden = hidden_seq[pred_positions]
    logits = output_head(pred_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
    answer_index = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits[torch.arange(len(answer_ids), device=logits.device), answer_index]
    selected_log_probs = torch.log_softmax(logits, dim=-1)[
        torch.arange(len(answer_ids), device=logits.device),
        answer_index,
    ]
    return float(selected_log_probs.sum().item()), float(selected_logits.sum().item())


def main():
    ap = argparse.ArgumentParser(description="Compare raw and tuned multimodal first-token trajectories.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--lens_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="conflict")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--margin_eps", type=float, default=0.1)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--plot_max_records", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_dir = prefix_model_relative_path(args.out_dir, model_name=args.model_name, model=args.model)
    position = infer_position_from_manifest(args.manifest)
    resume_scope = build_resume_scope(args.manifest, args.lens_ckpt)
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=PARENT_DIR,
            task_name="tuned_lens_analyze_trajectories",
            model_name=args.model_name,
            model=args.model,
            position=position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="tuned_lens_analyze_trajectories",
        manifest=args.manifest,
        out_dir=args.out_dir,
        position=position,
    )

    ensure_video_import_compat()
    rows = filter_rows(read_jsonl(args.manifest), args.prompt_types)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    ckpt = torch.load(args.lens_ckpt, map_location="cpu")
    processor = load_processor_with_compat(args.model)
    tokenizer = processor.tokenizer
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
    translators = build_translators(
        layer_indices=ckpt["layer_indices"],
        hidden_dim=ckpt["hidden_dim"],
        dtype=resolve_dtype(ckpt["translator_dtype"]),
        rank=ckpt["translator_rank"],
    ).to(lens_device)
    translators.load_state_dict(ckpt["state_dict"], strict=True)
    translators.eval()

    eligible_rows = []
    skipped_unencodable = 0
    filter_pbar = tqdm(rows, total=len(rows), desc="filter eligible rows", unit="sample")
    for row in filter_pbar:
        gold_token_ids = answer_token_ids(tokenizer, row["gold_answer"])
        wrong_token_ids = answer_token_ids(tokenizer, row["wrong_answer"])
        if not gold_token_ids or not wrong_token_ids:
            skipped_unencodable += 1
            filter_pbar.set_postfix(kept=len(eligible_rows), skipped=skipped_unencodable)
            continue
        row = dict(row)
        row["gold_token_ids"] = gold_token_ids
        row["wrong_token_ids"] = wrong_token_ids
        eligible_rows.append(row)
        filter_pbar.set_postfix(kept=len(eligible_rows), skipped=skipped_unencodable)
    if args.resume:
        eligible_rows = [row for row in eligible_rows if not tracker.is_done(row["sample_key"])]

    batch_records = tracker.read_records("trajectory_records.jsonl")
    group_summary = {}
    raw_sums = [0.0 for _ in ckpt["layer_indices"]]
    tuned_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_raw_gold_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_raw_wrong_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_tuned_gold_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    overall_tuned_wrong_logit_sums = [0.0 for _ in ckpt["layer_indices"]]
    for record in batch_records:
        accumulate_trajectory_record(
            record,
            group_summary,
            raw_sums,
            tuned_sums,
            overall_raw_gold_logit_sums,
            overall_raw_wrong_logit_sums,
            overall_tuned_gold_logit_sums,
            overall_tuned_wrong_logit_sums,
        )
    project_root = Path(__file__).resolve().parents[1]
    lm_device = output_head.weight.device

    batches = list(chunked(eligible_rows, args.batch_size))
    batch_pbar = tqdm(batches, total=len(batches), desc="analyze trajectories", unit="batch")
    for batch_rows in batch_pbar:
        batch_inputs = build_batch_inputs(processor, batch_rows, project_root, image_size=args.image_size)
        batch_inputs.pop("token_type_ids", None)
        prompt_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
        pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        if pad_token_id is None:
            pad_token_id = 0
        gold_inputs = append_answer_tokens(
            batch_inputs,
            [row["gold_token_ids"] for row in batch_rows],
            pad_token_id=pad_token_id,
        )
        wrong_inputs = append_answer_tokens(
            batch_inputs,
            [row["wrong_token_ids"] for row in batch_rows],
            pad_token_id=pad_token_id,
        )
        gold_inputs = move_to_device(gold_inputs, first_device, model_dtype)
        wrong_inputs = move_to_device(wrong_inputs, first_device, model_dtype)

        with torch.no_grad():
            gold_out = model(**gold_inputs, output_hidden_states=True, use_cache=False)
            wrong_out = model(**wrong_inputs, output_hidden_states=True, use_cache=False)

        gold_hidden_states = gold_out.hidden_states[1:]
        wrong_hidden_states = wrong_out.hidden_states[1:]

        for item_idx, row in enumerate(batch_rows):
            prompt_len = int(prompt_lengths[item_idx])
            gold_ids = row["gold_token_ids"]
            wrong_ids = row["wrong_token_ids"]
            raw_deltas = []
            tuned_deltas = []
            raw_gold_logits = []
            raw_wrong_logits = []
            tuned_gold_logits = []
            tuned_wrong_logits = []

            for pos, layer_idx in enumerate(ckpt["layer_indices"]):
                gold_hs_seq = gold_hidden_states[layer_idx][item_idx]
                wrong_hs_seq = wrong_hidden_states[layer_idx][item_idx]
                raw_gold_logprob, raw_gold_logit = answer_score_from_hidden(
                    gold_hs_seq,
                    prompt_len,
                    gold_ids,
                    output_head,
                    lm_device,
                )
                raw_wrong_logprob, raw_wrong_logit = answer_score_from_hidden(
                    wrong_hs_seq,
                    prompt_len,
                    wrong_ids,
                    output_head,
                    lm_device,
                )
                translator = translators[str(layer_idx)]
                translator_dtype = next(translator.parameters()).dtype
                tuned_gold_hs = translator(
                    gold_hs_seq.to(device=lens_device, dtype=translator_dtype)
                )
                tuned_wrong_hs = translator(
                    wrong_hs_seq.to(device=lens_device, dtype=translator_dtype)
                )
                tuned_gold_logprob, tuned_gold_logit = answer_score_from_hidden(
                    tuned_gold_hs,
                    prompt_len,
                    gold_ids,
                    output_head,
                    lm_device,
                )
                tuned_wrong_logprob, tuned_wrong_logit = answer_score_from_hidden(
                    tuned_wrong_hs,
                    prompt_len,
                    wrong_ids,
                    output_head,
                    lm_device,
                )
                raw_delta = raw_wrong_logprob - raw_gold_logprob
                tuned_delta = tuned_wrong_logprob - tuned_gold_logprob
                raw_deltas.append(raw_delta)
                raw_gold_logits.append(raw_gold_logit)
                raw_wrong_logits.append(raw_wrong_logit)
                tuned_deltas.append(tuned_delta)
                tuned_gold_logits.append(tuned_gold_logit)
                tuned_wrong_logits.append(tuned_wrong_logit)

                raw_sums[pos] += raw_delta
                tuned_sums[pos] += tuned_delta
                overall_raw_gold_logit_sums[pos] += raw_gold_logit
                overall_raw_wrong_logit_sums[pos] += raw_wrong_logit
                overall_tuned_gold_logit_sums[pos] += tuned_gold_logit
                overall_tuned_wrong_logit_sums[pos] += tuned_wrong_logit

            final_gold_logprob, _ = answer_score_from_logits(
                gold_out.logits[item_idx],
                prompt_len,
                gold_ids,
            )
            final_wrong_logprob, _ = answer_score_from_logits(
                wrong_out.logits[item_idx],
                prompt_len,
                wrong_ids,
            )
            final_delta = final_wrong_logprob - final_gold_logprob
            raw_stats = trajectory_stats(raw_deltas, args.margin_eps)
            tuned_stats = trajectory_stats(tuned_deltas, args.margin_eps)

            record = {
                "record_id": row["record_id"],
                "sample_key": row["sample_key"],
                "img_id": row["img_id"],
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
            tracker.append_record("trajectory_records.jsonl", record)
            tracker.mark_done(record["sample_key"], {"record_id": record["record_id"]})
            accumulate_trajectory_record(
                record,
                group_summary,
                raw_sums,
                tuned_sums,
                overall_raw_gold_logit_sums,
                overall_raw_wrong_logit_sums,
                overall_tuned_gold_logit_sums,
                overall_tuned_wrong_logit_sums,
            )

        batch_pbar.set_postfix(records=len(batch_records))

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
        "image_size": args.image_size,
        "n_records": len(batch_records),
        "skipped_multi_token": 0,
        "skipped_unencodable_answer": skipped_unencodable,
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
    tracker.finish(saved_trajectories=str(traj_path), saved_summary=str(out_dir / "summary.json"), n_records=len(batch_records))

    print(f"saved_trajectories={traj_path}")
    print(f"saved_summary={out_dir / 'summary.json'}")
    print(f"skipped_multi_token=0")
    print(f"skipped_unencodable_answer={skipped_unencodable}")


if __name__ == "__main__":
    main()
