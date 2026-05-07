#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot NC vs IC answer-time attention over image patches and evidence tokens."""

import argparse
import csv
import json
import math
import textwrap
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import PowerNorm
import torch
from PIL import Image

import ablate_head as ablate_mod
import head_scan_vqarad_mm_fastcache as hs


POSITION_DEFAULTS = {
    "prefix": {
        "ablation_json": "result_train_hulumed4b_prefix/ablate_selected_heads_ctx_only_hulumed4b_val.json",
        "selected_heads_json": "result_train_hulumed4b_prefix/headscan_vqarad_mm_hulumed4b_prefix/selected_heads_stable_hulumed4b.json",
        "out_dir": "attention_vis_prefix",
    },
    "before_question": {
        "ablation_json": "result_train_hulumed4b_before_question/ablate_selected_heads_ctx_only_hulumed4b_val.json",
        "selected_heads_json": "result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json",
        "out_dir": "attention_vis_before_question",
    },
    "before_answer": {
        "ablation_json": "result_train_hulumed4b_before_answer/ablate_selected_heads_ctx_only_hulumed4b_val.json",
        "selected_heads_json": "result_train_hulumed4b_before_answer/headscan_vqarad_mm_hulumed4b_before_answer/selected_heads_stable_hulumed4b.json",
        "out_dir": "attention_vis_before_answer",
    },
}

EVIDENCE_BEGIN_SENTINEL = "<<EVIDENCE_BEGIN_9f3a1c>>"
EVIDENCE_END_SENTINEL = "<<EVIDENCE_END_9f3a1c>>"
IMAGE_BEGIN_SENTINEL = "<<IMAGE_BEGIN_9f3a1c>>"
IMAGE_END_SENTINEL = "<<IMAGE_END_9f3a1c>>"



def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_project_path(project_root: Path, candidate: str) -> Path:
    raw = Path(candidate)
    if raw.exists():
        return raw.resolve()
    rel = (project_root / raw).resolve()
    if rel.exists():
        return rel
    normalized = candidate.replace("\\", "/")
    anchor = "Hulu-med/"
    if anchor in normalized:
        suffix = normalized.split(anchor, 1)[1]
        mapped = (project_root / suffix).resolve()
        if mapped.exists():
            return mapped
    name = raw.name
    if name:
        matches = list(project_root.rglob(name))
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(f"Cannot resolve path: {candidate}")


def resolve_image_path(row, image_root: Path, project_root: Path):
    candidates = []
    if row.get("image_path"):
        raw = Path(str(row["image_path"]).strip())
        if raw.is_absolute():
            candidates.append(raw)
        candidates.extend([project_root / raw, image_root / raw, image_root / raw.name])
    if row.get("img_id"):
        img_name = str(row["img_id"]).strip()
        candidates.extend([
            image_root / img_name,
            project_root / "data" / "VQA_RAD_Image_Folder" / img_name,
        ])
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path.resolve()
    return None


def choose_default_sample(ablation_payload, metric_name: str):
    records = ablation_payload.get("records", [])
    if not records:
        raise RuntimeError("Ablation JSON has no per-sample records")

    def effect(rec):
        return float(rec.get("metric_effect_reduction_fixed_nc", {}).get(metric_name, 0.0))

    recovered = [
        r for r in records
        if r.get("base_nc_pred") == "gold"
        and r.get("base_ctx_pred") == "wrong"
        and r.get("ab_ctx_pred") == "gold"
    ]
    if recovered:
        best = max(recovered, key=effect)
        return int(best["row_idx"]), best, "recovered_flip"

    flipped = [
        r for r in records
        if r.get("base_nc_pred") == "gold" and r.get("base_ctx_pred") == "wrong"
    ]
    if flipped:
        best = max(flipped, key=effect)
        return int(best["row_idx"]), best, "strong_conflict_flip"

    best = max(records, key=effect)
    return int(best["row_idx"]), best, "max_effect"


def find_record_by_row(records, row_idx: int):
    for rec in records:
        if int(rec.get("row_idx", -1)) == int(row_idx):
            return rec
    return None


def build_text_only_ids(processor, prompt: str):
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": hs.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids, text


def build_marked_evidence_text(evidence: str):
    return f"{EVIDENCE_BEGIN_SENTINEL}{hs.EVIDENCE_TMPL.format(ans=evidence)}{EVIDENCE_END_SENTINEL}"


def build_ctx_prompt_marked(question: str, evidence: str, position: str = "before_question"):
    base = hs.build_nc_prompt(question)
    marked_ev = build_marked_evidence_text(evidence)
    if position == "prefix":
        return marked_ev + "\n" + base
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base:
            raise ValueError("base prompt missing '\\nQuestion:' marker")
        return base.replace(marker, "\n" + marked_ev + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base:
            raise ValueError("base prompt missing '\\nAnswer:' marker")
        return base.replace(marker, "\n" + marked_ev + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


def build_mm_text_with_image_markers(processor, prompt: str, image):
    msgs = [
        {"role": "system", "content": hs.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_BEGIN_SENTINEL},
                {"type": "image", "image": image},
                {"type": "text", "text": IMAGE_END_SENTINEL + prompt},
            ],
        },
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_subsequence(haystack, needle):
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for start_idx in range(last):
        if haystack[start_idx : start_idx + len(needle)] == needle:
            return start_idx
    return None


def safe_decode(tokenizer, token_ids):
    try:
        return " ".join(tokenizer.decode(token_ids, clean_up_tokenization_spaces=False).split())
    except Exception:
        return ""


def decode_with_offsets(tokenizer, token_ids):
    pieces = []
    offsets = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def locate_substring_token_span(tokenizer, token_ids, needle: str):
    decoded, offsets = decode_with_offsets(tokenizer, token_ids)
    char_start = decoded.find(needle)
    if char_start < 0:
        return None, {
            "decoded_preview": decoded[:400],
            "needle": needle,
        }
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
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = len(token_ids)
    return (start_token, end_token), {
        "decoded_preview": decoded[:400],
        "needle": needle,
        "char_start": char_start,
        "char_end": char_end,
        "token_start": start_token,
        "token_end": end_token,
    }


def marker_debug_payload(marked_input_ids, begin_marker, end_marker, tokenizer):
    begin_span, begin_meta = locate_substring_token_span(tokenizer, marked_input_ids, begin_marker)
    end_span, end_meta = locate_substring_token_span(tokenizer, marked_input_ids, end_marker)
    return {
        "begin_marker": begin_marker,
        "end_marker": end_marker,
        "begin_span": begin_span,
        "end_span": end_span,
        "begin_meta": begin_meta,
        "end_meta": end_meta,
        "marked_prefix_decode": safe_decode(tokenizer, marked_input_ids[:160]),
        "marked_suffix_decode": safe_decode(tokenizer, marked_input_ids[-160:]),
    }


def locate_evidence_span_from_marked_inputs(actual_input_ids, marked_input_ids, tokenizer):
    payload = marker_debug_payload(marked_input_ids, EVIDENCE_BEGIN_SENTINEL, EVIDENCE_END_SENTINEL, tokenizer)
    begin_span = payload.get("begin_span")
    end_span = payload.get("end_span")
    begin_ids = tokenizer(EVIDENCE_BEGIN_SENTINEL, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(EVIDENCE_END_SENTINEL, add_special_tokens=False)["input_ids"]
    if begin_span is None or end_span is None:
        payload["method"] = "sentinel_not_found"
        return (0, 0), payload
    begin_start = begin_span[0]
    end_start = end_span[0]
    marked_evidence_span = (begin_start + len(begin_ids), end_start)
    actual_evidence_span = (
        max(0, marked_evidence_span[0] - len(begin_ids)),
        max(0, marked_evidence_span[1] - len(begin_ids)),
    )
    actual_evidence_span = (
        min(actual_evidence_span[0], len(actual_input_ids)),
        min(actual_evidence_span[1], len(actual_input_ids)),
    )
    payload.update({
        "method": "sentinel_marked_prompt",
        "begin_token_count": len(begin_ids),
        "end_token_count": len(end_ids),
        "marked_evidence_span": [int(marked_evidence_span[0]), int(marked_evidence_span[1])],
        "actual_length": len(actual_input_ids),
        "marked_length": len(marked_input_ids),
    })
    return actual_evidence_span, payload


def locate_image_span_from_marked_inputs(actual_input_ids, marked_input_ids, tokenizer):
    payload = marker_debug_payload(marked_input_ids, IMAGE_BEGIN_SENTINEL, IMAGE_END_SENTINEL, tokenizer)
    begin_span = payload.get("begin_span")
    end_span = payload.get("end_span")
    begin_ids = tokenizer(IMAGE_BEGIN_SENTINEL, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(IMAGE_END_SENTINEL, add_special_tokens=False)["input_ids"]
    if begin_span is None or end_span is None:
        payload["method"] = "sentinel_not_found"
        return (0, 0), payload
    begin_start = begin_span[0]
    end_start = end_span[0]
    marked_image_span = (begin_start + len(begin_ids), end_start)
    actual_image_span = (
        max(0, marked_image_span[0] - len(begin_ids)),
        max(0, marked_image_span[1] - len(begin_ids)),
    )
    actual_image_span = (
        min(actual_image_span[0], len(actual_input_ids)),
        min(actual_image_span[1], len(actual_input_ids)),
    )
    payload.update({
        "method": "image_sentinel_marked_prompt",
        "begin_token_count": len(begin_ids),
        "end_token_count": len(end_ids),
        "marked_image_span": [int(marked_image_span[0]), int(marked_image_span[1])],
        "actual_length": len(actual_input_ids),
        "marked_length": len(marked_input_ids),
    })
    return actual_image_span, payload


def build_token_source_labels(input_len: int, image_positions, evidence_span):
    labels = ["text"] * int(input_len)
    for pos in image_positions:
        if 0 <= int(pos) < input_len:
            labels[int(pos)] = "image"
    if evidence_span and evidence_span[1] > evidence_span[0]:
        for pos in range(int(evidence_span[0]), int(evidence_span[1])):
            if 0 <= pos < input_len and labels[pos] != "image":
                labels[pos] = "evidence"
    return labels


def collect_insert_like_spans(base_ids, target_ids):
    matcher = SequenceMatcher(a=base_ids, b=target_ids, autojunk=False)
    spans = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace") and j2 > j1:
            spans.append({
                "tag": tag,
                "base_span": (i1, i2),
                "target_span": (j1, j2),
                "length": int(j2 - j1),
            })
    return spans


def locate_inserted_target_span(base_ids, target_ids, expected_len=None):
    spans = collect_insert_like_spans(base_ids, target_ids)
    if not spans:
        return (0, 0)
    if expected_len is not None and expected_len > 0:
        best = min(
            spans,
            key=lambda item: (
                abs(item["length"] - expected_len),
                -item["length"],
                item["target_span"][0],
            ),
        )
        return best["target_span"]
    best = max(spans, key=lambda item: (item["length"], -item["target_span"][0]))
    return best["target_span"]


def get_merge_size(processor, model):
    values = []
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        for name in ("merge_size", "spatial_merge_size"):
            value = getattr(image_processor, name, None)
            if value is not None:
                values.append(int(value))
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for name in ("vision_spatial_merge_size", "spatial_merge_size"):
            value = getattr(cfg, name, None)
            if value is not None:
                values.append(int(value))
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            for name in ("spatial_merge_size", "merge_size"):
                value = getattr(vision_cfg, name, None)
                if value is not None:
                    values.append(int(value))
    for value in values:
        if value and value > 0:
            return value
    return 1


def expected_image_token_count(inputs, processor, model):
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    flat = grid[0].tolist() if torch.is_tensor(grid) else list(grid[0])
    if len(flat) != 3:
        return None
    t, h, w = [int(x) for x in flat]
    merge = get_merge_size(processor, model)
    if merge > 1 and h % merge == 0 and w % merge == 0:
        return int(t * (h // merge) * (w // merge))
    return int(t * h * w)


def refine_image_positions(input_ids, span, expected_count):
    positions = list(range(span[0], span[1]))
    if not positions:
        return positions
    if expected_count is None or expected_count <= 0 or len(positions) == expected_count:
        return positions
    span_ids = [input_ids[i] for i in positions]
    token_id, count = Counter(span_ids).most_common(1)[0]
    token_positions = [positions[i] for i, x in enumerate(span_ids) if x == token_id]
    if count == expected_count:
        return token_positions
    return positions[:expected_count] if len(positions) > expected_count else positions


def positions_to_span(positions):
    if not positions:
        return (0, 0)
    ordered = sorted(int(x) for x in positions)
    return (ordered[0], ordered[-1] + 1)


def collect_candidate_image_token_ids(processor, model):
    candidate_values = []
    tokenizer = getattr(processor, "tokenizer", None)
    for obj in (processor, tokenizer, getattr(model, "config", None), getattr(getattr(model, "config", None), "vision_config", None)):
        if obj is None:
            continue
        for name in (
            "image_token_id",
            "vision_token_id",
            "image_pad_token_id",
            "img_token_id",
            "boi_token_id",
            "eoi_token_id",
        ):
            value = getattr(obj, name, None)
            if value is not None:
                candidate_values.append(value)
        for name in (
            "image_token",
            "image_pad_token",
            "boi_token",
            "eoi_token",
            "img_token",
        ):
            value = getattr(obj, name, None)
            if value:
                candidate_values.append(value)
    ids = set()
    if tokenizer is not None:
        for value in candidate_values:
            if isinstance(value, int):
                ids.add(int(value))
            elif isinstance(value, str):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(value)
                    if token_id is not None and int(token_id) >= 0:
                        ids.add(int(token_id))
                except Exception:
                    pass
    return sorted(ids)


def locate_image_positions_by_token_ids(input_ids, candidate_ids, expected_count):
    best = []
    best_score = None
    for token_id in candidate_ids:
        positions = [idx for idx, value in enumerate(input_ids) if int(value) == int(token_id)]
        if not positions:
            continue
        span_width = positions[-1] - positions[0] + 1
        density = len(positions) / max(span_width, 1)
        if expected_count is not None and expected_count > 0:
            score = (abs(len(positions) - expected_count), -density, -len(positions), positions[0])
        else:
            score = (-len(positions), -density, positions[0])
        if best_score is None or score < best_score:
            best = positions
            best_score = score
    return best


def locate_image_positions_dense_repeat(input_ids, expected_count):
    counts = Counter(int(x) for x in input_ids)
    best = []
    best_score = None
    for token_id, count in counts.items():
        if count < 8:
            continue
        positions = [idx for idx, value in enumerate(input_ids) if int(value) == token_id]
        span_width = positions[-1] - positions[0] + 1
        density = len(positions) / max(span_width, 1)
        if density < 0.5:
            continue
        if expected_count is not None and expected_count > 0:
            score = (abs(count - expected_count), -density, -count, positions[0])
        else:
            score = (-count, -density, positions[0])
        if best_score is None or score < best_score:
            best = positions
            best_score = score
    return best


def locate_image_positions(input_ids, text_only_ids, expected_count, processor, model):
    candidate_ids = collect_candidate_image_token_ids(processor, model)
    positions = locate_image_positions_by_token_ids(input_ids, candidate_ids, expected_count)
    if positions:
        return positions, "special_token_id", {"candidate_ids": candidate_ids}

    positions = locate_image_positions_dense_repeat(input_ids, expected_count)
    if positions:
        token_id = int(input_ids[positions[0]])
        return positions, "dense_repeat_token", {"token_id": token_id, "count": len(positions)}

    span = locate_inserted_target_span(text_only_ids, input_ids, expected_len=expected_count)
    positions = refine_image_positions(input_ids, span, expected_count)
    return positions, "sequence_alignment_fallback", {"raw_span": [int(span[0]), int(span[1])]}

def build_inputs_mm_with_spans(processor, prompt: str, image, device, model, marked_prompt: str = ""):
    inputs = hs.build_inputs_mm(processor, prompt, image, device)
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    expected_image_count = expected_image_token_count(inputs, processor, model)

    image_span = (0, 0)
    image_method = "unresolved"
    image_debug = {"method": "none"}

    try:
        marked_image_text = build_mm_text_with_image_markers(processor, prompt, image)
        marked_image_inputs = hs.build_processor_inputs_mm(processor, marked_image_text, image)
        marked_image_input_ids = marked_image_inputs["input_ids"][0].detach().cpu().tolist()
        image_span, image_debug = locate_image_span_from_marked_inputs(
            input_ids,
            marked_image_input_ids,
            processor.tokenizer,
        )
        image_method = image_debug.get("method", "image_sentinel_marked_prompt")
        image_debug["marked_text_preview"] = marked_image_text[:400]
    except Exception as exc:
        image_debug = {"method": "image_sentinel_exception", "error": repr(exc)}

    if image_span[1] <= image_span[0]:
        text_only_ids, _ = build_text_only_ids(processor, prompt)
        image_positions, image_method, fallback_debug = locate_image_positions(
            input_ids,
            text_only_ids,
            expected_image_count,
            processor,
            model,
        )
        image_span = positions_to_span(image_positions)
        image_debug = fallback_debug
    else:
        image_positions = list(range(int(image_span[0]), int(image_span[1])))

    evidence_span = (0, 0)
    evidence_debug = {"method": "none"}
    if marked_prompt:
        marked_inputs = hs.build_inputs_mm(processor, marked_prompt, image, device)
        marked_input_ids = marked_inputs["input_ids"][0].detach().cpu().tolist()
        evidence_span, evidence_debug = locate_evidence_span_from_marked_inputs(
            input_ids,
            marked_input_ids,
            processor.tokenizer,
        )
        evidence_debug["marked_text_preview"] = marked_prompt[:400]
    token_source_labels = build_token_source_labels(len(input_ids), image_positions, evidence_span)
    return {
        "inputs": inputs,
        "input_ids": input_ids,
        "image_positions": image_positions,
        "image_span": image_span,
        "expected_image_count": expected_image_count,
        "image_locate_method": image_method,
        "image_locate_debug": image_debug,
        "evidence_span": evidence_span,
        "evidence_locate_debug": evidence_debug,
        "token_source_labels": token_source_labels,
    }

@torch.no_grad()
def build_prompt_cache_with_attentions(model, base_inputs):
    target_dtype = hs.infer_vision_input_dtype(model)
    fixed_inputs = hs.move_to_device(base_inputs, next(model.parameters()).device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=True, output_attentions=False, return_dict=True)
    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    cache = {
        "attn_prompt": attn_prompt.detach(),
        "past_key_values": out.past_key_values,
        "prompt_last_logits": out.logits[:, -1, :].detach(),
    }
    del out
    return cache


def choose_answer_tokens(tokenizer, scores, gold: str, wrong: str):
    answer = gold if scores["gold_lp"] >= scores["wrong_lp"] else wrong
    token_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise RuntimeError(f"Answer tokenization failed for: {answer!r}")
    token_labels = []
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        token_labels.append(piece if piece else str(token_id))
    return answer, [int(x) for x in token_ids], token_labels


@torch.no_grad()
def rollout_answer_attentions(model, prompt_cache, answer_token_ids):
    device = next(model.parameters()).device
    prompt_len = int(prompt_cache["attn_prompt"].shape[1])
    past_key_values = prompt_cache["past_key_values"]
    base_mask = prompt_cache["attn_prompt"].to(device)
    columns = []

    for step_idx, answer_token_id in enumerate(answer_token_ids):
        token = torch.tensor([[int(answer_token_id)]], dtype=torch.long, device=device)
        attn_mask = torch.cat(
            [
                base_mask,
                torch.ones((1, step_idx + 1), dtype=base_mask.dtype, device=device),
            ],
            dim=1,
        )
        out = model(
            input_ids=token,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        if not getattr(out, "attentions", None):
            raise RuntimeError("Model did not return attentions. Use eager attention.")
        columns.append([layer[0, :, 0, :prompt_len].detach().float().cpu() for layer in out.attentions])
        past_key_values = out.past_key_values

    return columns, prompt_len


def region_mass(matrix, positions):
    if not positions or matrix.numel() == 0:
        return 0.0
    valid_positions = [int(pos) for pos in positions if 0 <= int(pos) < matrix.shape[1]]
    if not valid_positions:
        return 0.0
    return float(matrix[:, valid_positions].sum(dim=1).mean())


def compact_tick_positions(n_items: int, n_ticks: int = 4):
    if n_items <= 0:
        return []
    if n_items <= n_ticks:
        return list(range(n_items))
    step = max(1, n_items // n_ticks)
    ticks = list(range(0, n_items, step))
    ticks = [t for t in ticks if t < n_items]
    if ticks and ticks[-1] != n_items - 1:
        ticks.append(n_items - 1)
    if len(ticks) > n_ticks + 1:
        ticks = ticks[: n_ticks] + [ticks[-1]]
    return sorted(set(ticks))


def build_attention_matrix(answer_columns, layer_idx: int, head_idx: int):
    rows = []
    for step_columns in answer_columns:
        rows.append(step_columns[layer_idx][head_idx].detach().float().cpu())
    if not rows:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.stack(rows, dim=0)



def draw_spans(ax, image_span, evidence_span):
    vision_color = "#2563eb"
    evidence_color = "#d62728"

    def draw_bracket(x0, x1, y, color, label, side="bottom", lw=0.85, text_dx=0.0, text_dy=None):
        span = max(1.0, x1 - x0)
        mid = (x0 + x1) / 2.0
        cap = 0.030
        notch_half = max(span * 0.010, 7.0)
        notch_depth = 0.072
        shoulder = max(span * 0.0035, 2.2)
        default_text_offset = -0.070 if side == "bottom" else 0.052
        text_offset = default_text_offset if text_dy is None else text_dy
        cap_sign = 1.0 if side == "bottom" else -1.0
        notch_sign = -1.0 if side == "bottom" else 1.0

        xs = [
            x0,
            x0,
            mid - notch_half,
            mid - shoulder,
            mid,
            mid + shoulder,
            mid + notch_half,
            x1,
            x1,
        ]
        ys = [
            y + cap_sign * cap,
            y,
            y,
            y + notch_sign * (notch_depth * 0.42),
            y + notch_sign * notch_depth,
            y + notch_sign * (notch_depth * 0.42),
            y,
            y,
            y + cap_sign * cap,
        ]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=5,
        )
        ax.text(
            mid + text_dx,
            y + text_offset,
            label,
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top" if side == "bottom" else "bottom",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=6,
        )

    if image_span and image_span[1] > image_span[0]:
        x0 = image_span[0] - 0.5
        x1 = image_span[1] - 0.5
        ax.axvspan(x0, x1, color=vision_color, alpha=0.060, zorder=0)
        ax.vlines([x0, x1], ymin=-0.5, ymax=ax.get_ylim()[0], colors=vision_color, alpha=0.20, linewidth=0.75)
        draw_bracket(x0, x1, -0.165, vision_color, "VISION TOKENS", side="bottom", lw=0.85, text_dy=-0.092)

    if evidence_span and evidence_span[1] > evidence_span[0]:
        x0 = evidence_span[0] - 0.5
        x1 = evidence_span[1] - 0.5
        ax.axvspan(x0, x1, color=evidence_color, alpha=0.070, zorder=0)
        ax.vlines([x0, x1], ymin=-0.5, ymax=ax.get_ylim()[0], colors=evidence_color, alpha=0.22, linewidth=0.7)
        draw_bracket(x0, x1, 1.018, evidence_color, "EVIDENCE TOKENS", side="top", lw=0.85, text_dx=-62.0, text_dy=0.072)



def build_display_matrix(panel):
    matrix = panel["attention_matrix"].detach().float().clone()
    image_span = panel.get("image_span", (0, 0))
    evidence_span = panel.get("evidence_span", (0, 0))

    if image_span and image_span[1] > image_span[0]:
        x0 = max(0, int(image_span[0]))
        x1 = min(matrix.shape[1], int(image_span[1]))
        if x1 > x0:
            matrix[:, x0:x1] *= 2.35

    if evidence_span and evidence_span[1] > evidence_span[0]:
        x0 = max(0, int(evidence_span[0]))
        x1 = min(matrix.shape[1], int(evidence_span[1]))
        if x1 > x0:
            matrix[:, x0:x1] *= 1.45

    return matrix

def render_head_figure(out_path: Path, meta, nc_panel, ic_panel):
    fig, axes = plt.subplots(2, 1, figsize=(17.2, 7.0), gridspec_kw={"hspace": 0.95})
    fig.subplots_adjust(left=0.08, right=0.92, top=0.85, bottom=0.08)

    question = " ".join(str(meta["question"]).split())
    if len(question) > 96:
        question = question[:93] + "..."
    fig.suptitle(
        f"L{meta['layer']}  H{meta['head']}  |  row={meta['row_idx']}\nquestion: {question}",
        fontsize=14,
        fontweight="semibold",
        y=0.965,
    )

    panels = [
        (axes[0], nc_panel, f"NC   pred={meta['nc_answer']}   |   A_vision={nc_panel['A_vision']:.4f}   |   A_conflict={nc_panel['A_conflict']:.4f}"),
        (axes[1], ic_panel, f"IC   pred={meta['ic_answer']}   |   A_vision={ic_panel['A_vision']:.4f}   |   A_conflict={ic_panel['A_conflict']:.4f}"),
    ]

    nc_display = build_display_matrix(nc_panel)
    ic_display = build_display_matrix(ic_panel)
    all_values = torch.cat([nc_display.reshape(-1), ic_display.reshape(-1)]).detach().float().cpu()
    positive = all_values[all_values > 0]
    vmax = float(torch.quantile(positive, 0.989)) if positive.numel() > 0 else 1e-6
    vmax = max(vmax, 1e-6)
    norm = PowerNorm(gamma=0.26, vmin=0.0, vmax=vmax)

    img = None
    for ax, panel, title in panels:
        display_matrix = build_display_matrix(panel).numpy()
        img = ax.imshow(display_matrix, cmap="viridis", norm=norm, interpolation="nearest", aspect="auto", origin="upper")
        draw_spans(ax, panel["image_span"], panel["evidence_span"])

        ax.set_xlim(-0.5, panel["attention_matrix"].shape[1] - 0.5)
        x_ticks = compact_tick_positions(panel["attention_matrix"].shape[1], n_ticks=5)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="x", labelsize=11, pad=2, length=3)

        y_ticks = list(range(len(panel["answer_token_labels"])))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(panel["answer_token_labels"], fontsize=12)
        ax.tick_params(axis="y", labelsize=12, length=0)

        ax.set_xlabel("")
        ax.set_ylabel("Answer tokens", fontsize=12, labelpad=8)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_facecolor("#fbfbfb")
        ax.margins(x=0)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#666666")

    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.94, fraction=0.03, pad=0.02)
    cbar.set_label("Attention score", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--image_root", default="data/VQA_RAD_Image_Folder")
    ap.add_argument("--ablation_json", default="")
    ap.add_argument("--selected_heads_json", default="")
    ap.add_argument("--data_csv", default="")
    ap.add_argument("--row_idx", type=int, default=-1)
    ap.add_argument("--top_k_heads", type=int, default=6)
    ap.add_argument("--head_pairs", default="")
    ap.add_argument("--metric_name", default="follow_conflict")
    ap.add_argument("--max_image_side", type=int, default=672)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    defaults = POSITION_DEFAULTS[args.position]
    ablation_json = resolve_project_path(project_root, args.ablation_json or defaults["ablation_json"])
    selected_heads_json = resolve_project_path(project_root, args.selected_heads_json or defaults["selected_heads_json"])
    out_dir = args.out_dir or defaults["out_dir"]

    ablation_payload = read_json(ablation_json)
    data_csv = resolve_project_path(project_root, args.data_csv) if args.data_csv else resolve_project_path(project_root, ablation_payload["config"]["data_csv"])
    records = ablation_payload.get("records", [])

    if args.row_idx >= 0:
        row_idx = int(args.row_idx)
        chosen_record = find_record_by_row(records, row_idx)
        selection_reason = "manual_row_idx"
    else:
        row_idx, chosen_record, selection_reason = choose_default_sample(ablation_payload, args.metric_name)

    model_name = args.model or ablation_payload["config"].get("model", "")
    if not model_name:
        raise SystemExit("Please provide --model or ensure the ablation JSON stores it.")

    rows = read_csv_rows(data_csv)
    row = rows[row_idx]
    question = str(row["question"]).strip()
    gold = str(row["gold"]).strip().lower()
    wrong = str(row["wrong"]).strip().lower()

    image_root = resolve_project_path(project_root, args.image_root)
    image_path = resolve_image_path(row, image_root, project_root)
    if image_path is None:
        raise FileNotFoundError(f"Cannot resolve image for row_idx={row_idx}")

    model, processor = hs.load_model_and_processor(model_name, args.device, args.dtype)
    device = next(model.parameters()).device
    image = hs.resize_image_max_side(Image.open(image_path), max_side=args.max_image_side)

    nc_prompt = hs.build_nc_prompt(question)
    ic_prompt = hs.build_ctx_prompt(question, wrong, position=args.position)
    ic_prompt_marked = build_ctx_prompt_marked(question, wrong, position=args.position)

    nc_pack = build_inputs_mm_with_spans(processor, nc_prompt, image, device, model)
    ic_pack = build_inputs_mm_with_spans(processor, ic_prompt, image, device, model, marked_prompt=ic_prompt_marked)

    nc_inputs = nc_pack["inputs"]
    ic_inputs = ic_pack["inputs"]
    nc_input_ids = nc_pack["input_ids"]
    ic_input_ids = ic_pack["input_ids"]
    nc_image_positions = nc_pack["image_positions"]
    ic_image_positions = ic_pack["image_positions"]
    nc_image_span = nc_pack["image_span"]
    ic_image_span = ic_pack["image_span"]
    evidence_span_ic = ic_pack["evidence_span"]
    nc_expected_image_count = nc_pack["expected_image_count"]
    ic_expected_image_count = ic_pack["expected_image_count"]
    nc_image_method = nc_pack["image_locate_method"]
    ic_image_method = ic_pack["image_locate_method"]
    nc_image_debug = nc_pack["image_locate_debug"]
    ic_image_debug = ic_pack["image_locate_debug"]
    evidence_locate_debug = ic_pack["evidence_locate_debug"]

    tokenizer = processor.tokenizer
    nc_cache = build_prompt_cache_with_attentions(model, nc_inputs)
    ic_cache = build_prompt_cache_with_attentions(model, ic_inputs)
    nc_scores = hs.score_two_options_from_cache(model, processor, nc_cache, gold, wrong)
    ic_scores = hs.score_two_options_from_cache(model, processor, ic_cache, gold, wrong)
    nc_answer, nc_answer_token_ids, nc_answer_token_labels = choose_answer_tokens(tokenizer, nc_scores, gold, wrong)
    ic_answer, ic_answer_token_ids, ic_answer_token_labels = choose_answer_tokens(tokenizer, ic_scores, gold, wrong)

    nc_answer_columns, nc_prompt_len = rollout_answer_attentions(model, nc_cache, nc_answer_token_ids)
    ic_answer_columns, ic_prompt_len = rollout_answer_attentions(model, ic_cache, ic_answer_token_ids)

    if args.head_pairs:
        selected_pairs = []
        for piece in args.head_pairs.split(","):
            layer_str, head_str = piece.strip().split(":")
            selected_pairs.append((int(layer_str), int(head_str)))
    else:
        selected_pairs = ablate_mod.parse_selected_heads(str(selected_heads_json))[: max(1, args.top_k_heads)]

    out_root = (project_root / out_dir / f"row_{row_idx:04d}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_heads = []
    evidence_preview = " ".join(tokenizer.decode(ic_input_ids[evidence_span_ic[0]: evidence_span_ic[1]]).split())

    for layer_idx, head_idx in selected_pairs:
        nc_attention_matrix = build_attention_matrix(nc_answer_columns, layer_idx, head_idx)
        ic_attention_matrix = build_attention_matrix(ic_answer_columns, layer_idx, head_idx)

        nc_panel = {
            "attention_matrix": nc_attention_matrix,
            "A_vision": region_mass(nc_attention_matrix, nc_image_positions),
            "A_conflict": 0.0,
            "image_span": nc_image_span,
            "evidence_span": (0, 0),
            "answer_token_labels": nc_answer_token_labels,
        }
        ic_panel = {
            "attention_matrix": ic_attention_matrix,
            "A_vision": region_mass(ic_attention_matrix, ic_image_positions),
            "A_conflict": region_mass(ic_attention_matrix, list(range(evidence_span_ic[0], evidence_span_ic[1]))),
            "image_span": ic_image_span,
            "evidence_span": evidence_span_ic,
            "answer_token_labels": ic_answer_token_labels,
        }

        fig_path = out_root / f"L{layer_idx:02d}_H{head_idx:02d}_attention_matrix.png"
        render_head_figure(
            out_path=fig_path,
            meta={
                "layer": int(layer_idx),
                "head": int(head_idx),
                "row_idx": row_idx,
                "question": question,
                "nc_answer": nc_answer,
                "ic_answer": ic_answer,
                "evidence_preview": evidence_preview,
            },
            nc_panel=nc_panel,
            ic_panel=ic_panel,
        )

        summary_heads.append({
            "layer": int(layer_idx),
            "head": int(head_idx),
            "A_vision_NC": nc_panel["A_vision"],
            "A_vision_IC": ic_panel["A_vision"],
            "A_vision_delta": ic_panel["A_vision"] - nc_panel["A_vision"],
            "A_conflict_NC": 0.0,
            "A_conflict_IC": ic_panel["A_conflict"],
            "A_conflict_delta": ic_panel["A_conflict"],
            "figure_png": str(fig_path),
        })

    summary = {
        "selection_reason": selection_reason,
        "position": args.position,
        "ablation_json": str(ablation_json),
        "selected_heads_json": str(selected_heads_json),
        "data_csv": str(data_csv),
        "model": model_name,
        "sample": {
            "row_idx": row_idx,
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "image_path": str(image_path),
            "ablation_record": chosen_record,
            "nc_scores": nc_scores,
            "ic_scores": ic_scores,
            "nc_answer": nc_answer,
            "ic_answer": ic_answer,
            "nc_answer_token_ids": [int(x) for x in nc_answer_token_ids],
            "ic_answer_token_ids": [int(x) for x in ic_answer_token_ids],
            "nc_answer_token_labels": nc_answer_token_labels,
            "ic_answer_token_labels": ic_answer_token_labels,
        },
        "spans": {
            "nc_image_span": [int(nc_image_span[0]), int(nc_image_span[1])],
            "ic_image_span": [int(ic_image_span[0]), int(ic_image_span[1])],
            "evidence_span_ic": [int(evidence_span_ic[0]), int(evidence_span_ic[1])],
            "nc_image_token_count": len(nc_image_positions),
            "ic_image_token_count": len(ic_image_positions),
            "nc_expected_image_token_count": nc_expected_image_count,
            "ic_expected_image_token_count": ic_expected_image_count,
            "nc_image_locate_method": nc_image_method,
            "ic_image_locate_method": ic_image_method,
            "nc_image_locate_debug": nc_image_debug,
            "ic_image_locate_debug": ic_image_debug,
            "evidence_locate_debug": evidence_locate_debug,
            "evidence_preview": evidence_preview,
            "nc_token_source_labels_preview": nc_pack["token_source_labels"][:64],
            "ic_token_source_labels_preview": ic_pack["token_source_labels"][:64],
        },
        "heads": summary_heads,
    }
    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] summary: {summary_path}")
    for item in summary_heads:
        print(f"[OK] figure: {item['figure_png']}")


if __name__ == "__main__":
    main()




