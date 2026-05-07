#!/usr/bin/env python3
"""Render a Fig.6-style text-only vs multimodal comparison for InternVL on SLAKE."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


ROOT = Path("/root/logit_lens")
FIG6_DIR = ROOT / "PIC" / "fig6"
OUT_DIR = FIG6_DIR / "result_heatmap"
SLAKE_TEXT_DIR = ROOT / "Slake_vqa" / "text_conflict"
INTERNVL_RESULTS = SLAKE_TEXT_DIR / "internvl35_4b" / "result_before_question_slake"
CONFLICT_DIR = ROOT / "conflictmedqa" / "Qwen3-4B_exp"
CONFLICT_POSITIONS = CONFLICT_DIR / "result_all_positions" / "conflict_positions_val.jsonl"
CONFLICT_PROMPT_BANK = CONFLICT_DIR / "tuned_lens" / "data" / "val_before_question.jsonl"

if str(FIG6_DIR) not in sys.path:
    sys.path.insert(0, str(FIG6_DIR))
if str(SLAKE_TEXT_DIR) not in sys.path:
    sys.path.insert(0, str(SLAKE_TEXT_DIR))

from plot_fig6_attention_heatmap import HeatmapBundle, render_comparison  # type: ignore  # noqa: E402
import head_scan_slake_mm_fastcache_accel as slake_hs  # type: ignore  # noqa: E402
from model_utils import (  # type: ignore  # noqa: E402
    infer_vision_input_dtype,
    load_mm_model,
    load_processor_with_compat,
    move_to_device,
)


EVIDENCE_BEGIN_SENTINEL = "<<EVIDENCE_BEGIN_9f3a1c>>"
EVIDENCE_END_SENTINEL = "<<EVIDENCE_END_9f3a1c>>"
IMAGE_BEGIN_SENTINEL = "<<IMAGE_BEGIN_9f3a1c>>"
IMAGE_END_SENTINEL = "<<IMAGE_END_9f3a1c>>"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_selected_pairs(path: Path) -> List[Tuple[int, int]]:
    payload = read_json(path)
    selected = payload.get("selected", [])
    if selected:
        return [(int(item["layer"]), int(item["head"])) for item in selected]
    pairs = payload.get("selected_pairs", [])
    return [(int(item["layer"]), int(item["head"])) for item in pairs]


def collect_all_head_pairs(attentions: Sequence[Any]) -> List[Tuple[int, int]]:
    n_layers = len(attentions)
    if n_layers == 0:
        return []
    n_heads = int(attentions[0].shape[1])
    return [(layer_idx, head_idx) for layer_idx in range(n_layers) for head_idx in range(n_heads)]


def build_marked_ctx_prompt(question: str, evidence: str, position: str = "before_question") -> str:
    base = slake_hs.build_nc_prompt(question)
    marked_evidence = (
        f"{EVIDENCE_BEGIN_SENTINEL}"
        f"{slake_hs.EVIDENCE_TMPL.format(ans=evidence)}"
        f"{EVIDENCE_END_SENTINEL}"
    )
    if position == "prefix":
        return marked_evidence + "\n" + base
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base:
            raise ValueError("base prompt missing Question marker")
        return base.replace(marker, "\n" + marked_evidence + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base:
            raise ValueError("base prompt missing Answer marker")
        return base.replace(marker, "\n" + marked_evidence + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


def build_plain_ctx_prompt(question: str, evidence: str, position: str = "before_question") -> str:
    return slake_hs.build_ctx_prompt(question, evidence, position=position)


def inject_marked_evidence(base_prompt: str, evidence_text: str, position: str = "before_question") -> str:
    if position == "prefix":
        return evidence_text + "\n" + base_prompt
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base prompt missing Question marker")
        return base_prompt.replace(marker, "\n" + evidence_text + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base prompt missing Answer marker")
        return base_prompt.replace(marker, "\n" + evidence_text + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


def build_conflict_prompt_bank_index(rows: Sequence[dict]) -> Dict[Tuple[int, str], str]:
    index: Dict[Tuple[int, str], str] = {}
    for row in rows:
        if str(row.get("position")) == "before_question" and str(row.get("prompt_type")) == "base":
            pair_id = int(row.get("pair_id", -1))
            side = str(row.get("side", ""))
            prompt_text = str(row.get("prompt_text", ""))
            if pair_id >= 0 and side and prompt_text:
                index[(pair_id, side)] = prompt_text
    return index


def choose_conflict_sample() -> dict:
    rows = read_jsonl(CONFLICT_POSITIONS)
    candidates = [
        row for row in rows
        if str(row.get("position")) == "before_question"
        and str(row.get("conflict", {}).get("pred", "")).strip().lower() != str(row.get("gold", "")).strip().lower()
    ]
    if not candidates:
        raise RuntimeError("No before_question conflict-following samples found in ConflictMedQA.")
    return max(candidates, key=lambda row: abs(float(row.get("shift_conflict", 0.0))))


def find_conflict_sample(pair_id: int, side: str) -> dict:
    rows = read_jsonl(CONFLICT_POSITIONS)
    for row in rows:
        if (
            str(row.get("position")) == "before_question"
            and int(row.get("pair_id", -1)) == int(pair_id)
            and str(row.get("side", "")) == str(side)
        ):
            return row
    raise KeyError(f"Cannot find ConflictMedQA sample pair_id={pair_id}, side={side}")


def build_text_only_messages(prompt_text: str) -> List[dict]:
    return [
        {"role": "system", "content": slake_hs.SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
    ]


def build_multimodal_messages(prompt_text: str, image: Image.Image) -> List[dict]:
    return [
        {"role": "system", "content": slake_hs.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_BEGIN_SENTINEL},
                {"type": "image", "image": image},
                {"type": "text", "text": IMAGE_END_SENTINEL + prompt_text},
            ],
        },
    ]


def tokenize_messages(model, processor, messages: List[dict]) -> Dict[str, Any]:
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch.pop("token_type_ids", None)
    target_dtype = infer_vision_input_dtype(model)
    return move_to_device(batch, model.device, float_dtype=target_dtype)


@torch.inference_mode()
def build_prompt_cache(model, processor, messages: List[dict]) -> Dict[str, Any]:
    base_inputs = tokenize_messages(model, processor, messages)
    out = model(**base_inputs, use_cache=True, return_dict=True)
    input_ids_prompt = base_inputs["input_ids"]
    attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": out.logits[:, -1, :].detach(),
    }


@torch.inference_mode()
def continuation_logprob(model, processor, cache_pack: Dict[str, Any], continuation: str) -> float:
    tok = processor.tokenizer
    cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    if cont_ids.numel() == 0:
        return float("-inf")

    attn_prompt = cache_pack["attn_prompt"]
    past_key_values = cache_pack["past_key_values"]
    prompt_last_logits = cache_pack["prompt_last_logits"]
    cont_len = int(cont_ids.shape[1])

    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)
    total = float(log_probs0[0, int(cont_ids[0, 0])])
    if cont_len == 1:
        return total

    inp = cont_ids[:, :-1]
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)
    out = model(
        input_ids=inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
        return_dict=True,
    )
    log_probs = torch.log_softmax(out.logits, dim=-1)
    for i in range(1, cont_len):
        total += float(log_probs[0, i - 1, int(cont_ids[0, i])])
    return total


@torch.inference_mode()
def score_two_options(model, processor, messages: List[dict], gold: str, wrong: str) -> Dict[str, float]:
    cache_pack = build_prompt_cache(model, processor, messages)
    return {
        "gold_lp": continuation_logprob(model, processor, cache_pack, " " + str(gold)),
        "wrong_lp": continuation_logprob(model, processor, cache_pack, " " + str(wrong)),
    }


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for idx in range(last):
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return idx
    return None


def decode_with_offsets(tokenizer, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def locate_substring_token_span(tokenizer, token_ids: Sequence[int], needle: str) -> Optional[Tuple[int, int]]:
    decoded, offsets = decode_with_offsets(tokenizer, token_ids)
    char_start = decoded.find(needle)
    if char_start < 0:
        return None
    char_end = char_start + len(needle)
    start_token = None
    end_token = None
    for idx, (start, end) in enumerate(offsets):
        if start_token is None and end > char_start:
            start_token = idx
        if start < char_end:
            end_token = idx + 1
        if start >= char_end:
            break
    if start_token is None or end_token is None or end_token <= start_token:
        return None
    return int(start_token), int(end_token)


def locate_marker_span(tokenizer, input_ids: Sequence[int], start_marker: str, end_marker: str) -> Optional[Tuple[int, int]]:
    begin_ids = tokenizer(start_marker, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(end_marker, add_special_tokens=False)["input_ids"]
    start_idx = find_subsequence(input_ids, begin_ids)
    end_idx = find_subsequence(input_ids, end_ids)
    if start_idx is None or end_idx is None:
        return None
    content_start = start_idx + len(begin_ids)
    if end_idx <= content_start:
        return None
    return int(content_start), int(end_idx)


def locate_evidence_span(tokenizer, input_ids: Sequence[int], wrong: str) -> Optional[Tuple[int, int]]:
    span = locate_marker_span(tokenizer, input_ids, EVIDENCE_BEGIN_SENTINEL, EVIDENCE_END_SENTINEL)
    if span is not None:
        return span
    evidence_text = slake_hs.EVIDENCE_TMPL.format(ans=wrong)
    span = locate_substring_token_span(tokenizer, input_ids, evidence_text)
    if span is not None:
        return span
    return locate_substring_token_span(tokenizer, input_ids, wrong)


def collect_selected_head_vectors(attentions: Sequence[Any], selected_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for layer_idx, head_idx in selected_pairs:
        vec = attentions[layer_idx][0, head_idx, -1].detach().float().cpu().numpy()
        vectors.append(np.asarray(vec, dtype=float))
    return np.stack(vectors, axis=0)


def average_selected_attention(attentions: Sequence[Any], selected_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    mats = []
    for layer_idx, head_idx in selected_pairs:
        mats.append(attentions[layer_idx][0, head_idx].detach().float().cpu().numpy())
    return np.mean(np.stack(mats, axis=0), axis=0)


def collect_all_head_vectors(attentions: Sequence[Any]) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    all_pairs = collect_all_head_pairs(attentions)
    vectors = collect_selected_head_vectors(attentions, all_pairs)
    return all_pairs, vectors


def predict_label(scores: Dict[str, float]) -> str:
    return "gold" if scores["gold_lp"] >= scores["wrong_lp"] else "wrong"


def resolve_data_csv(config_value: str) -> Path:
    rel = config_value.replace("./", "")
    path = SLAKE_TEXT_DIR / rel
    if not path.exists():
        raise FileNotFoundError(f"Cannot resolve data_csv from config: {config_value}")
    return path


def search_best_sample(
    model,
    processor,
    search_limit: int,
) -> Dict[str, Any]:
    payload_paths = [
        INTERNVL_RESULTS / "ablate" / "ablate_selected_heads_ctx_only_train.json",
        INTERNVL_RESULTS / "ablate" / "ablate_selected_heads_ctx_only_val.json",
    ]
    candidates: List[Dict[str, Any]] = []
    for payload_path in payload_paths:
        payload = read_json(payload_path)
        csv_path = resolve_data_csv(payload["config"]["data_csv"])
        rows = read_csv_rows(csv_path)
        for rec in payload.get("records", []):
            row_idx = int(rec["row_idx"])
            row = rows[row_idx]
            mm_scores = rec["base_ctx_scores"]
            mm_pred = str(rec["base_ctx_pred"])
            gold = str(row["gold"]).strip().lower()
            wrong = str(row["wrong"]).strip().lower()
            mm_margin = float(mm_scores["gold_lp"]) - float(mm_scores["wrong_lp"])
            candidates.append(
                {
                    "split": "train" if "train" in payload_path.name else "val",
                    "payload_path": str(payload_path),
                    "csv_path": str(csv_path),
                    "row_idx": row_idx,
                    "row": row,
                    "record": rec,
                    "gold": gold,
                    "wrong": wrong,
                    "mm_pred": mm_pred,
                    "mm_margin": mm_margin,
                }
            )

    # Prefer multimodal resist rows, especially those with smaller margins (harder cases).
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item["mm_pred"] == "gold" else 1,
            abs(float(item["mm_margin"])),
            item["row_idx"],
        ),
    )

    best: Optional[Dict[str, Any]] = None
    search_pool = ranked[: max(1, int(search_limit))]
    for item in search_pool:
        row = item["row"]
        prompt = build_plain_ctx_prompt(str(row["question"]), str(row["wrong"]), position="before_question")
        messages = build_text_only_messages(prompt)
        text_scores = score_two_options(model, processor, messages, item["gold"], item["wrong"])
        text_margin = float(text_scores["gold_lp"]) - float(text_scores["wrong_lp"])
        text_pred = predict_label(text_scores)

        combo_score = (float(text_scores["wrong_lp"]) - float(text_scores["gold_lp"])) + item["mm_margin"]
        enriched = dict(item)
        enriched.update(
            {
                "text_scores": text_scores,
                "text_margin": text_margin,
                "text_pred": text_pred,
                "combo_score": combo_score,
            }
        )
        if best is None:
            best = enriched
            continue

        best_is_ideal = (best["text_pred"] == "wrong" and best["mm_pred"] == "gold")
        cur_is_ideal = (text_pred == "wrong" and item["mm_pred"] == "gold")
        if cur_is_ideal and not best_is_ideal:
            best = enriched
            continue
        if cur_is_ideal == best_is_ideal and combo_score > float(best["combo_score"]):
            best = enriched

    if best is None:
        raise RuntimeError("No candidate sample found during InternVL SLAKE search.")
    return best


def build_bundle(
    model,
    processor,
    row: dict,
    multimodal: bool,
    split_name: str,
    row_idx: int,
) -> HeatmapBundle:
    gold = str(row["gold"]).strip().lower()
    wrong = str(row["wrong"]).strip().lower()
    question = str(row["question"])
    prompt_marked = build_marked_ctx_prompt(question, wrong, position="before_question")
    if multimodal:
        image = Image.open(str(row["image_path"])).convert("RGB")
        image.thumbnail((672, 672))
        messages = build_multimodal_messages(prompt_marked, image)
    else:
        messages = build_text_only_messages(prompt_marked)

    inputs = tokenize_messages(model, processor, messages)
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, return_dict=True, use_cache=False)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokenizer = processor.tokenizer
    conflict_span = locate_evidence_span(tokenizer, input_ids, wrong)
    visual_span = locate_marker_span(tokenizer, input_ids, IMAGE_BEGIN_SENTINEL, IMAGE_END_SENTINEL) if multimodal else None
    question_span = locate_substring_token_span(tokenizer, input_ids, "Question:")
    answer_span = locate_substring_token_span(tokenizer, input_ids, "\nAnswer:")
    query_span = None
    if question_span is not None and answer_span is not None and answer_span[0] > question_span[0]:
        query_span = (question_span[0], answer_span[0])

    all_pairs, selected_head_vectors = collect_all_head_vectors(attentions)
    avg_matrix = average_selected_attention(attentions, all_pairs)
    sample_title = f"{split_name} row {row_idx}: {question}"

    return HeatmapBundle(
        title="Text-only" if not multimodal else "Multimodal VQA",
        model_name="InternVL3_5-4B",
        sample_title=sample_title,
        matrix=avg_matrix,
        row_labels=[],
        x_len=len(input_ids),
        visual_span=visual_span,
        query_span=query_span,
        conflict_span=conflict_span,
        selected_heads=all_pairs,
        selected_head_vectors=selected_head_vectors,
        answer=gold,
        answer_token_ids=[],
        answer_token_labels=[],
        meta={
            "split": split_name,
            "row_idx": row_idx,
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "multimodal": multimodal,
        },
    )


def build_conflict_bundle(
    model,
    processor,
    selected_sample: dict,
    prompt_bank_index: Dict[Tuple[int, str], str],
) -> HeatmapBundle:
    pair_id = int(selected_sample["pair_id"])
    side = str(selected_sample["side"])
    gold = str(selected_sample["gold"]).strip().lower()
    wrong = str(selected_sample["conflict_label"]).strip().lower()
    base_prompt = prompt_bank_index.get((pair_id, side))
    if not base_prompt:
        raise KeyError(f"Missing ConflictMedQA base prompt for pair_id={pair_id}, side={side}")

    evidence = "The answer is yes." if wrong == "yes" else "The answer is no."
    marked_evidence = (
        f"{EVIDENCE_BEGIN_SENTINEL}"
        f"{evidence}"
        f"{EVIDENCE_END_SENTINEL}"
    )
    marked_prompt = inject_marked_evidence(base_prompt, marked_evidence, position="before_question")
    messages = build_text_only_messages(marked_prompt)
    inputs = tokenize_messages(model, processor, messages)
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True, return_dict=True, use_cache=False)
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Model did not return attentions for ConflictMedQA text-only bundle.")

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    tokenizer = processor.tokenizer
    conflict_span = locate_evidence_span(tokenizer, input_ids, evidence)
    question_text = ""
    if "\nQuestion:" in base_prompt and "\nAnswer:" in base_prompt:
        question_text = base_prompt.split("\nQuestion:", 1)[1].split("\nAnswer:", 1)[0].strip()
    query_span = locate_substring_token_span(tokenizer, input_ids, question_text) if question_text else None

    all_pairs, all_head_vectors = collect_all_head_vectors(attentions)
    avg_matrix = average_selected_attention(attentions, all_pairs)

    return HeatmapBundle(
        title="Text-only",
        model_name="InternVL3_5-4B",
        sample_title=f"ConflictMedQA pair {pair_id} {side}: {question_text[:90]}",
        matrix=avg_matrix,
        row_labels=[],
        x_len=len(input_ids),
        visual_span=None,
        query_span=query_span,
        conflict_span=conflict_span,
        selected_heads=all_pairs,
        selected_head_vectors=all_head_vectors,
        answer=gold,
        answer_token_ids=[],
        answer_token_labels=[],
        meta={
            "source": "conflictmedqa_qwen_question",
            "pair_id": pair_id,
            "side": side,
            "gold": gold,
            "wrong": wrong,
            "question": question_text,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/InternVL3_5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search_limit", type=int, default=80)
    parser.add_argument("--row_idx", type=int, default=-1, help="Optional explicit SLAKE row_idx within --split.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--conflict_pair_id", type=int, default=-1)
    parser.add_argument("--conflict_side", choices=["correct", "wrong"], default="wrong")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_mm_model(
        args.model,
        device_map=None,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if args.device:
        model = model.to(args.device)
    model.eval()
    processor = load_processor_with_compat(args.model)

    if args.row_idx >= 0:
        payload_path = INTERNVL_RESULTS / "ablate" / f"ablate_selected_heads_ctx_only_{args.split}.json"
        payload = read_json(payload_path)
        csv_path = resolve_data_csv(payload["config"]["data_csv"])
        rows = read_csv_rows(csv_path)
        row = rows[int(args.row_idx)]
        chosen = {
            "split": args.split,
            "row_idx": int(args.row_idx),
            "row": row,
            "record": next((rec for rec in payload["records"] if int(rec["row_idx"]) == int(args.row_idx)), None),
        }
    else:
        chosen = search_best_sample(model, processor, search_limit=args.search_limit)

    row = chosen["row"]
    split_name = str(chosen["split"])
    row_idx = int(chosen["row_idx"])
    if args.conflict_pair_id >= 0:
        conflict_sample = find_conflict_sample(args.conflict_pair_id, args.conflict_side)
    else:
        conflict_sample = choose_conflict_sample()
    prompt_bank_index = build_conflict_prompt_bank_index(read_jsonl(CONFLICT_PROMPT_BANK))
    conflict_bundle = build_conflict_bundle(
        model,
        processor,
        selected_sample=conflict_sample,
        prompt_bank_index=prompt_bank_index,
    )
    vqa_bundle = build_bundle(
        model,
        processor,
        row=row,
        multimodal=True,
        split_name=split_name,
        row_idx=row_idx,
    )

    output_stem = f"fig6_slake_internvl35_alllayers_conflictmedqa_pair{conflict_sample['pair_id']}_{split_name}_row{row_idx}"
    paths = render_comparison(conflict_bundle, vqa_bundle, output_stem, OUT_DIR)

    summary = {
        "output_stem": output_stem,
        "png": paths["png"],
        "pdf": paths["pdf"],
        "aggregation": "all_layers_all_heads_mean",
        "model": args.model,
        "left_source": "ConflictMedQA question from previous Qwen3-4B setup, run on InternVL3_5-4B text-only",
        "right_source": "Slake multimodal sample on InternVL3_5-4B",
        "conflict_pair_id": int(conflict_sample["pair_id"]),
        "conflict_side": str(conflict_sample["side"]),
        "split": split_name,
        "row_idx": row_idx,
        "question": row["question"],
        "gold": row["gold"],
        "wrong": row["wrong"],
        "image_path": row["image_path"],
        "search_limit": args.search_limit,
    }
    if "text_scores" in chosen:
        summary["text_scores"] = chosen["text_scores"]
        summary["text_pred"] = chosen["text_pred"]
        summary["text_margin"] = chosen["text_margin"]
        summary["mm_pred"] = chosen["mm_pred"]
        summary["mm_margin"] = chosen["mm_margin"]
        summary["combo_score"] = chosen["combo_score"]
    if chosen.get("record") is not None:
        summary["ablate_record"] = chosen["record"]

    summary_path = OUT_DIR / f"{output_stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
