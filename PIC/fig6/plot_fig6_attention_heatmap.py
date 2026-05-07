#!/usr/bin/env python3
"""Build Fig. 6 attention heatmaps for VQA-RAD and ConflictMedQA.

The script renders the full self-attention matrix for the selected conflict
heads in each model, then places both heatmaps in one shared figure with a
common color scale.
"""

from __future__ import annotations

import argparse
import importlib.util
import gc
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image
import torch


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "fig6"
RESULTS_DIR = OUT_DIR / "result_heatmap"

VQA_DIR = ROOT / "VQA_RAD" / "Hulu-med"
CONFLICT_DIR = ROOT / "conflictmedqa" / "Qwen3-4B_exp"

# Load the VQA helpers by path so we can reuse the existing token-span and
# attention-rollout utilities without duplicating the long helper file.
if str(VQA_DIR) not in sys.path:
    sys.path.insert(0, str(VQA_DIR))

import plot_cross_attention_nc_ic as vqa_vis  # type: ignore  # noqa: E402
import head_scan_vqarad_mm_fastcache as vqa_hs  # type: ignore  # noqa: E402


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


conflict_ablate = load_module_from_path("fig6_conflict_ablate", CONFLICT_DIR / "ablate_head.py")


PALETTE = {
    "visual": "#2f6df6",
    "conflict": "#d62728",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "fig6_purple_gold",
    ["#090012", "#31104b", "#6c3fa4", "#b65c72", "#fde725"],
)


@dataclass
class HeatmapBundle:
    title: str
    model_name: str
    sample_title: str
    matrix: np.ndarray
    row_labels: List[str]
    x_len: int
    visual_span: Optional[Tuple[int, int]]
    query_span: Optional[Tuple[int, int]]
    conflict_span: Optional[Tuple[int, int]]
    selected_heads: List[Tuple[int, int]]
    selected_head_vectors: np.ndarray
    answer: str
    answer_token_ids: List[int]
    answer_token_labels: List[str]
    meta: Dict[str, Any]


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


def safe_text(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def contiguous_ranges(values: Sequence[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(v) for v in values))
    if not ordered:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev + 1))
            start = prev = value
    ranges.append((start, prev + 1))
    return ranges


def pick_best_span(spans: List[Tuple[int, int]], preference: str = "largest") -> Optional[Tuple[int, int]]:
    if not spans:
        return None
    if preference == "largest":
        return max(spans, key=lambda x: (x[1] - x[0], -x[0]))
    return spans[0]


def normalize_prompt_span(span: Optional[Tuple[int, int]], length: int) -> Optional[Tuple[int, int]]:
    if span is None:
        return None
    start, end = int(span[0]), int(span[1])
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    if end <= start:
        return None
    return start, end


def compact_tick_positions(n_items: int, n_ticks: int = 6) -> List[int]:
    if n_items <= 0:
        return []
    if n_items <= n_ticks:
        return list(range(n_items))
    ticks = np.linspace(0, n_items - 1, num=n_ticks, dtype=int).tolist()
    return sorted(set(int(t) for t in ticks))


def parse_head_pairs(path: Path) -> List[Tuple[int, int]]:
    return [tuple(pair) for pair in vqa_vis.ablate_mod.parse_selected_heads(str(path))]


def span_block_stats(matrix: np.ndarray, span: Optional[Tuple[int, int]]) -> Optional[Tuple[float, float]]:
    if span is None:
        return None
    start, end = int(span[0]), int(span[1])
    if end <= start:
        return None
    block = np.asarray(matrix[start:end, start:end], dtype=float)
    if block.size == 0:
        return None
    off_diag = block.copy()
    np.fill_diagonal(off_diag, np.nan)
    valid = off_diag[np.isfinite(off_diag)]
    if valid.size == 0:
        valid = block[np.isfinite(block)]
    if valid.size == 0:
        return None
    return float(np.mean(valid)), float(np.max(valid))


def suppress_diagonal_band(matrix: np.ndarray, width: int = 1) -> np.ndarray:
    """Hide trivial self-attention so non-diagonal structure is easier to compare."""
    arr = np.asarray(matrix, dtype=float)
    out = arr.copy()
    n = min(out.shape[0], out.shape[1])
    for offset in range(width):
        idx = np.arange(0, n - offset)
        out[idx + offset, idx] = 0.0
    return out


def choose_vqa_resist_record(ablation_payload: Dict[str, Any]) -> Tuple[int, dict, str]:
    records = ablation_payload.get("records", [])
    if not records:
        raise RuntimeError("Ablation JSON has no per-sample records")

    resist_records = [
        r for r in records
        if str(r.get("base_nc_pred")) == str(r.get("base_ctx_pred"))
    ]
    if not resist_records:
        raise RuntimeError("No resist records found in the VQA ablation JSON")

    gold_resist = [
        r for r in resist_records
        if str(r.get("base_nc_pred")) == str(r.get("gold"))
    ]
    key_fn = lambda r: abs(float(r.get("metric_abs_ctx_change", {}).get("follow_conflict", 0.0)))
    if gold_resist:
        best = max(gold_resist, key=key_fn)
        return int(best["row_idx"]), best, "resist_gold"

    best = max(resist_records, key=key_fn)
    return int(best["row_idx"]), best, "resist_same_pred"


def choose_vqa_resist_records(ablation_payload: Dict[str, Any], limit: int) -> List[dict]:
    records = ablation_payload.get("records", [])
    if not records:
        raise RuntimeError("Ablation JSON has no per-sample records")

    resist_records = [
        r for r in records
        if str(r.get("base_nc_pred")) == str(r.get("base_ctx_pred"))
    ]
    if not resist_records:
        raise RuntimeError("No resist records found in the VQA ablation JSON")

    def score(rec: dict) -> Tuple[int, float, int]:
        gold_match = 0 if str(rec.get("base_nc_pred")) == str(rec.get("gold")) else 1
        delta = abs(float(rec.get("metric_abs_ctx_change", {}).get("follow_conflict", 0.0)))
        row_idx = int(rec.get("row_idx", 0))
        return (gold_match, -delta, row_idx)

    ranked = sorted(resist_records, key=score)
    return ranked[: max(0, int(limit))]


def choose_conflict_resist_samples(summary: Dict[str, Any], limit: int) -> List[dict]:
    selected_samples = summary.get("selected_samples", [])
    if not selected_samples:
        raise RuntimeError("No selected_samples found in the conflict summary")

    resist_samples = [
        s for s in selected_samples
        if str(s.get("side")) == "correct"
    ]
    if not resist_samples:
        raise RuntimeError("No resist samples found in the conflict summary")

    def score(sample: dict) -> Tuple[int, int]:
        pair_id = int(sample.get("pair_id", 0))
        return (pair_id, 0)

    ranked = sorted(resist_samples, key=score)
    return ranked[: max(0, int(limit))]


def choose_vqa_correct_prediction_records(pred_rows: Sequence[dict], csv_rows: Sequence[dict], limit: int) -> List[dict]:
    csv_index = {
        (
            str(row.get("img_id", "")).strip().lower(),
            str(row.get("question", "")).strip().lower(),
            str(row.get("gold", "")).strip().lower(),
        ): idx
        for idx, row in enumerate(csv_rows)
    }
    candidates = [
        row for row in pred_rows
        if str(row.get("position")) == "before_question"
        and str(row.get("conflict_pred")).strip().lower() == str(row.get("gold")).strip().lower()
    ]
    mapped = []
    for row in candidates:
        key = (
            str(row.get("img_id", "")).strip().lower(),
            str(row.get("question", "")).strip().lower(),
            str(row.get("gold", "")).strip().lower(),
        )
        if key not in csv_index:
            continue
        enriched = dict(row)
        enriched["csv_row_idx"] = int(csv_index[key])
        mapped.append(enriched)
    if not mapped:
        raise RuntimeError("No before_question correct-prediction samples found in VQA preds.jsonl")

    def score(row: dict) -> Tuple[float, int]:
        margin = abs(float(row.get("conflict_margin", 0.0)))
        sample_id = int(row.get("id", 0))
        return (-margin, sample_id)

    ranked = sorted(mapped, key=score)
    return ranked[: max(0, int(limit))]


def choose_conflict_error_records(rows: Sequence[dict], limit: int) -> List[dict]:
    candidates = [
        row for row in rows
        if str(row.get("position")) == "before_question"
        and str(row.get("conflict", {}).get("pred", "")).strip().lower() != str(row.get("gold", "")).strip().lower()
    ]
    if not candidates:
        raise RuntimeError("No before_question error samples found in conflict_positions_val.jsonl")

    def score(row: dict) -> Tuple[float, int, str]:
        shift = abs(float(row.get("shift_conflict", 0.0)))
        pair_id = int(row.get("pair_id", 0))
        side = str(row.get("side", ""))
        return (-shift, pair_id, side)

    ranked = sorted(candidates, key=score)
    return ranked[: max(0, int(limit))]


def build_conflict_prompt_bank_index(rows: Sequence[dict]) -> Dict[Tuple[int, str], str]:
    index: Dict[Tuple[int, str], str] = {}
    for row in rows:
        if (
            str(row.get("position")) == "before_question"
            and str(row.get("prompt_type")) == "base"
        ):
            pair_id = int(row.get("pair_id", -1))
            side = str(row.get("side", ""))
            prompt_text = str(row.get("prompt_text", ""))
            if pair_id >= 0 and side and prompt_text:
                index[(pair_id, side)] = prompt_text
    return index


def filter_conflict_samples_with_prompt_bank(samples: Sequence[dict], prompt_bank_index: Dict[Tuple[int, str], str]) -> List[dict]:
    return [
        sample for sample in samples
        if (int(sample.get("pair_id", -1)), str(sample.get("side", ""))) in prompt_bank_index
    ]


def average_selected_attention(attentions: Sequence[Any], selected_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    matrices = []
    for layer_idx, head_idx in selected_pairs:
        matrices.append(attentions[layer_idx][0, head_idx].detach().float().cpu().numpy())
    if not matrices:
        raise ValueError("No selected attention matrices found.")
    return np.mean(np.stack(matrices, axis=0), axis=0)


def collect_selected_head_vectors(attentions: Sequence[Any], selected_pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for layer_idx, head_idx in selected_pairs:
        vec = attentions[layer_idx][0, head_idx, -1].detach().float().cpu().numpy()
        vectors.append(np.asarray(vec, dtype=float))
    if not vectors:
        raise ValueError("No selected head vectors found.")
    return np.stack(vectors, axis=0)


def compact_row_ticks(row_labels: Sequence[str], max_ticks: int = 10) -> Tuple[np.ndarray, List[str]]:
    n = len(row_labels)
    if n <= 0:
        return np.asarray([], dtype=int), []
    positions = np.asarray(compact_tick_positions(n, max_ticks), dtype=int)
    labels = [row_labels[int(pos)] for pos in positions]
    return positions, labels


def split_span(span: Optional[Tuple[int, int]], n_parts: int) -> List[Tuple[int, int]]:
    if span is None:
        return []
    start, end = int(span[0]), int(span[1])
    if end <= start or n_parts <= 0:
        return []
    edges = np.linspace(start, end, num=n_parts + 1)
    parts: List[Tuple[int, int]] = []
    prev = start
    for edge in edges[1:]:
        nxt = max(prev + 1, int(round(edge)))
        nxt = min(nxt, end)
        parts.append((prev, nxt))
        prev = nxt
    if parts:
        parts[-1] = (parts[-1][0], end)
    return [(a, b) for a, b in parts if b > a]


def aggregate_group_values(values: np.ndarray, label: str, topk_frac: float = 0.2) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    if label.startswith("V"):
        k = max(1, int(np.ceil(arr.size * topk_frac)))
        return float(np.mean(np.sort(arr)[-k:]))
    return float(np.mean(arr))


def summarize_vectors_by_groups(vectors: np.ndarray, groups: Sequence[Tuple[str, Tuple[int, int]]]) -> np.ndarray:
    rows: List[np.ndarray] = []
    for vec in np.asarray(vectors, dtype=float):
        vals = []
        for label, (start, end) in groups:
            vals.append(aggregate_group_values(vec[int(start) : int(end)], label) if end > start else 0.0)
        rows.append(np.asarray(vals, dtype=float))
    return np.stack(rows, axis=0)


def summarize_layers_by_groups(
    vectors: np.ndarray,
    selected_pairs: Sequence[Tuple[int, int]],
    groups: Sequence[Tuple[str, Tuple[int, int]]],
) -> Tuple[np.ndarray, List[str]]:
    if len(selected_pairs) != len(vectors):
        raise ValueError("selected_pairs and vectors must have the same length for layer aggregation.")

    layer_to_rows: Dict[int, List[np.ndarray]] = {}
    for (layer_idx, _head_idx), vec in zip(selected_pairs, np.asarray(vectors, dtype=float)):
        vals = []
        for label, (start, end) in groups:
            vals.append(aggregate_group_values(vec[int(start) : int(end)], label) if end > start else 0.0)
        layer_to_rows.setdefault(int(layer_idx), []).append(np.asarray(vals, dtype=float))

    ordered_layers = sorted(layer_to_rows.keys())
    summary_rows = [np.mean(np.stack(layer_to_rows[layer], axis=0), axis=0) for layer in ordered_layers]
    row_labels = [f"Layer {layer + 1}" for layer in ordered_layers]
    return np.stack(summary_rows, axis=0), row_labels


def head_labels_for_count(n_heads: int) -> List[str]:
    return [f"Head {idx + 1}" for idx in range(n_heads)]


def draw_bottom_group_bar(ax, groups: Sequence[Tuple[str, Tuple[int, int], str]], y: float = -0.11) -> None:
    x0 = 0
    for _label, (_start, _end), color in groups:
        width = 1
        rect = Rectangle((x0, y), width, 0.025, transform=ax.transAxes, color=color, clip_on=False)
        ax.add_patch(rect)
        x0 += width / max(len(groups), 1)


def draw_group_outline(ax, start_col: int, end_col: int, color: str, n_rows: int, lw: float = 3.0) -> None:
    rect = Rectangle((start_col - 0.5, -0.5), end_col - start_col, n_rows, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device, dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor = value.to(device=device)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        else:
            moved[key] = value
    return moved


def release_cuda_objects(*objs: Any) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_vqa_text_only_bundle(selected_record: dict) -> HeatmapBundle:
    selected_heads_json = (
        VQA_DIR
        / "result_train_hulumed4b_before_question"
        / "headscan_vqarad_mm_hulumed4b_before_question"
        / "selected_heads_stable_hulumed4b.json"
    )
    data_csv = VQA_DIR / "data" / "nc_cc_both_correct_rerun_tmp_val.csv"
    model_name = "/root/autodl-tmp/Hulu-Med-4B"

    rows = vqa_vis.read_csv_rows(data_csv)
    row_idx = int(selected_record["csv_row_idx"])
    chosen_record = selected_record
    row = rows[row_idx]

    question = str(row["question"]).strip()
    gold = str(row["gold"]).strip().lower()
    wrong = str(row["wrong"]).strip().lower()

    model = None
    processor = None
    enc = None
    try:
        model, processor = vqa_hs.load_model_and_processor(model_name, device="auto", dtype="bf16")
        prompt = vqa_hs.build_ctx_prompt(question, wrong, position="before_question")
        marked_prompt = vqa_vis.build_ctx_prompt_marked(question, wrong, position="before_question")

        input_ids, chat_text = vqa_vis.build_text_only_ids(processor, prompt)
        marked_ids, _marked_chat_text = vqa_vis.build_text_only_ids(processor, marked_prompt)

        conflict_span, _ = vqa_vis.locate_substring_token_span(
            processor.tokenizer,
            marked_ids,
            vqa_vis.EVIDENCE_BEGIN_SENTINEL,
        )
        conflict_end_span, _ = vqa_vis.locate_substring_token_span(
            processor.tokenizer,
            marked_ids,
            vqa_vis.EVIDENCE_END_SENTINEL,
        )
        if conflict_span is None or conflict_end_span is None:
            raise RuntimeError("Failed to locate conflict evidence span in the Hulu-med text-only prompt.")
        begin_len = len(processor.tokenizer(vqa_vis.EVIDENCE_BEGIN_SENTINEL, add_special_tokens=False)["input_ids"])
        conflict_start = conflict_span[0] + begin_len
        conflict_end = conflict_end_span[0]
        conflict_span = normalize_prompt_span((conflict_start, conflict_end), len(marked_ids))

        query_span, _ = vqa_vis.locate_substring_token_span(processor.tokenizer, input_ids, question)
        query_span = normalize_prompt_span(query_span, len(input_ids))

        enc = processor.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        enc = move_batch_to_device(enc, next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            out = model(**enc, output_attentions=True, use_cache=False, return_dict=True)
        selected_pairs = parse_head_pairs(selected_heads_json)
        matrix = average_selected_attention(out.attentions, selected_pairs)
        matrix = np.asarray(matrix, dtype=float)
        selected_head_vectors = collect_selected_head_vectors(out.attentions, selected_pairs)

        sample_title = f"row={row_idx} | {safe_text(question, 72)} | gold={gold} | wrong={wrong}"

        return HeatmapBundle(
            title="VQA-RAD Hulu-med-4B text-only before_question",
            model_name=model_name,
            sample_title=sample_title,
            matrix=matrix,
            row_labels=head_labels_for_count(len(selected_pairs)),
            x_len=int(matrix.shape[1]),
            visual_span=None,
            query_span=query_span,
            conflict_span=conflict_span,
            selected_heads=selected_pairs,
            selected_head_vectors=selected_head_vectors,
            answer="",
            answer_token_ids=[],
            answer_token_labels=[],
            meta={
                "selection_reason": "same_model_text_only",
                "sample_kind": "pred_correct",
                "row_idx": row_idx,
                "selected_heads_json": str(selected_heads_json),
                "data_csv": str(data_csv),
                "question": question,
                "gold": gold,
                "wrong": wrong,
                "chosen_record": chosen_record,
                "query_span": query_span,
                "conflict_span": conflict_span,
                "same_model_text_only": True,
            },
        )
    finally:
        release_cuda_objects(enc, processor, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def build_conflict_hulumed_text_only_bundle(
    selected_sample: dict,
    prompt_bank_index: Dict[Tuple[int, str], str],
) -> HeatmapBundle:
    selected_heads_json = (
        VQA_DIR
        / "result_train_hulumed4b_before_question"
        / "headscan_vqarad_mm_hulumed4b_before_question"
        / "selected_heads_stable_hulumed4b.json"
    )
    model_name = "/root/autodl-tmp/Hulu-Med-4B"

    side = str(selected_sample["side"])
    gold = str(selected_sample["gold"]).strip().lower()
    wrong = str(selected_sample["conflict_label"]).strip().lower()
    position = str(selected_sample.get("position", "before_question"))
    base_prompt = prompt_bank_index.get((int(selected_sample["pair_id"]), side))
    if not base_prompt:
        raise KeyError(
            f"Cannot find base prompt for pair_id={selected_sample['pair_id']} side={side} in prompt bank."
        )

    model = None
    processor = None
    enc = None
    try:
        model, processor = vqa_hs.load_model_and_processor(model_name, device="auto", dtype="bf16")

        evidence = conflict_ablate.make_evidence(label_yes=(wrong == "yes"))
        marked_evidence = f"{vqa_vis.EVIDENCE_BEGIN_SENTINEL}{evidence}{vqa_vis.EVIDENCE_END_SENTINEL}"
        prompt = conflict_ablate.inject_evidence(base_prompt, evidence, position)
        marked_prompt = conflict_ablate.inject_evidence(base_prompt, marked_evidence, position)

        input_ids, chat_text = vqa_vis.build_text_only_ids(processor, prompt)
        marked_ids, _ = vqa_vis.build_text_only_ids(processor, marked_prompt)

        conflict_span, _ = vqa_vis.locate_substring_token_span(
            processor.tokenizer,
            marked_ids,
            vqa_vis.EVIDENCE_BEGIN_SENTINEL,
        )
        conflict_end_span, _ = vqa_vis.locate_substring_token_span(
            processor.tokenizer,
            marked_ids,
            vqa_vis.EVIDENCE_END_SENTINEL,
        )
        if conflict_span is None or conflict_end_span is None:
            raise RuntimeError("Failed to locate conflict evidence span in the Hulu-med text-only prompt.")
        begin_len = len(processor.tokenizer(vqa_vis.EVIDENCE_BEGIN_SENTINEL, add_special_tokens=False)["input_ids"])
        conflict_start = conflict_span[0] + begin_len
        conflict_end = conflict_end_span[0]
        conflict_span = normalize_prompt_span((conflict_start, conflict_end), len(marked_ids))

        question_match = re.search(r"(?is)question\s*:\s*(.*?)\n\s*answer\s*:", base_prompt)
        question_text = question_match.group(1).strip() if question_match else ""
        query_span = None
        if question_text:
            query_span, _ = vqa_vis.locate_substring_token_span(processor.tokenizer, input_ids, question_text)
            query_span = normalize_prompt_span(query_span, len(input_ids))

        enc = processor.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        enc = move_batch_to_device(enc, next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            out = model(**enc, output_attentions=True, use_cache=False, return_dict=True)
        selected_pairs = parse_head_pairs(selected_heads_json)
        matrix = average_selected_attention(out.attentions, selected_pairs)
        matrix = np.asarray(matrix, dtype=float)
        selected_head_vectors = collect_selected_head_vectors(out.attentions, selected_pairs)

        base_prompt_summary = safe_text(base_prompt.splitlines()[0], 70)
        sample_title = (
            f"pair={selected_sample['pair_id']} | side={side} | gold={gold} | "
            f"wrong={wrong} | {base_prompt_summary}"
        )

        return HeatmapBundle(
            title="ConflictMedQA text-only on Hulu-med-4B",
            model_name=model_name,
            sample_title=sample_title,
            matrix=matrix,
            row_labels=head_labels_for_count(len(selected_pairs)),
            x_len=int(matrix.shape[1]),
            visual_span=None,
            query_span=query_span,
            conflict_span=conflict_span,
            selected_heads=selected_pairs,
            selected_head_vectors=selected_head_vectors,
            answer="",
            answer_token_ids=[],
            answer_token_labels=[],
            meta={
                "same_model_text_only": True,
                "same_model_source": "conflictmedqa_text",
                "selected_sample": selected_sample,
                "base_prompt": base_prompt,
                "gold": gold,
                "wrong": wrong,
                "position": position,
                "query_span": query_span,
                "conflict_span": conflict_span,
                "selected_heads_json": str(selected_heads_json),
            },
        )
    finally:
        release_cuda_objects(enc, processor, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def build_vqa_bundle(selected_record: Optional[dict] = None) -> HeatmapBundle:
    ablation_json = VQA_DIR / "result_train_hulumed4b_before_question" / "ablate_selected_heads_ctx_only_hulumed4b_val.json"
    preds_jsonl = (
        VQA_DIR
        / "eval-results_vqarad_hulumed4b_all"
        / "_archive_zengjiaqi_Medical_LLM_Hulu-Med-4B"
        / "before_question"
        / "preds.jsonl"
    )
    selected_heads_json = (
        VQA_DIR
        / "result_train_hulumed4b_before_question"
        / "headscan_vqarad_mm_hulumed4b_before_question"
        / "selected_heads_stable_hulumed4b.json"
    )
    data_csv = VQA_DIR / "data" / "nc_cc_both_correct_rerun_tmp_val.csv"
    image_root = VQA_DIR / "data" / "VQA_RAD_Image_Folder"
    model_name = "/root/autodl-tmp/Hulu-Med-4B"

    rows = vqa_vis.read_csv_rows(data_csv)
    if selected_record is None:
        pred_rows = read_jsonl(preds_jsonl)
        chosen_pred = choose_vqa_correct_prediction_records(pred_rows, rows, 1)[0]
        row_idx = int(chosen_pred["csv_row_idx"])
        chosen_record = chosen_pred
        selection_reason = "pred_correct_default"
    else:
        row_idx = int(selected_record["csv_row_idx"])
        chosen_record = selected_record
        selection_reason = "pred_correct_batch"
    row = rows[row_idx]

    question = str(row["question"]).strip()
    gold = str(row["gold"]).strip().lower()
    wrong = str(row["wrong"]).strip().lower()
    image_path = vqa_vis.resolve_image_path(row, image_root, VQA_DIR)
    if image_path is None:
        raise FileNotFoundError(f"Cannot resolve image for row_idx={row_idx}")

    model = None
    processor = None
    image = None
    ic_inputs = None
    try:
        model, processor = vqa_hs.load_model_and_processor(model_name, device="auto", dtype="bf16")
        device = next(model.parameters()).device
        image = vqa_hs.resize_image_max_side(Image.open(image_path), max_side=672)

        ic_prompt = vqa_hs.build_ctx_prompt(question, wrong, position="before_question")
        ic_prompt_marked = vqa_vis.build_ctx_prompt_marked(question, wrong, position="before_question")
        ic_pack = vqa_vis.build_inputs_mm_with_spans(
            processor=processor,
            prompt=ic_prompt,
            image=image,
            device=device,
            model=model,
            marked_prompt=ic_prompt_marked,
        )
        ic_inputs = ic_pack["inputs"]
        evidence_span = normalize_prompt_span(tuple(ic_pack["evidence_span"]), len(ic_pack["input_ids"]))
        visual_span = normalize_prompt_span(tuple(ic_pack["image_span"]), len(ic_pack["input_ids"]))
        question_span, _ = vqa_vis.locate_substring_token_span(processor.tokenizer, ic_pack["input_ids"], question)
        question_span = normalize_prompt_span(question_span, len(ic_pack["input_ids"]))

        ic_inputs = move_batch_to_device(ic_inputs, device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            out = model(**ic_inputs, output_attentions=True, use_cache=False, return_dict=True)
        selected_pairs = parse_head_pairs(selected_heads_json)
        matrix = average_selected_attention(out.attentions, selected_pairs)
        matrix = np.asarray(matrix, dtype=float)
        selected_head_vectors = collect_selected_head_vectors(out.attentions, selected_pairs)

        sample_title = f"row={row_idx} | {safe_text(question, 72)} | gold={gold} | wrong={wrong}"

        return HeatmapBundle(
            title="VQA-RAD Hulu-med-4B before_question",
            model_name=model_name,
            sample_title=sample_title,
            matrix=matrix,
            row_labels=head_labels_for_count(len(selected_pairs)),
            x_len=int(matrix.shape[1]),
            visual_span=visual_span,
            query_span=question_span,
            conflict_span=evidence_span,
            selected_heads=selected_pairs,
            selected_head_vectors=selected_head_vectors,
            answer="",
            answer_token_ids=[],
            answer_token_labels=[],
            meta={
                "selection_reason": selection_reason,
                "sample_kind": "pred_correct",
                "row_idx": row_idx,
                "ablation_json": str(ablation_json),
                "preds_jsonl": str(preds_jsonl),
                "selected_heads_json": str(selected_heads_json),
                "data_csv": str(data_csv),
                "image_path": str(image_path),
                "row_idx": row_idx,
                "question": question,
                "gold": gold,
                "wrong": wrong,
                "chosen_record": chosen_record,
                "question_span": question_span,
                "visual_span": visual_span,
                "ic_evidence_span": evidence_span,
            },
        )
    finally:
        release_cuda_objects(ic_inputs, image, processor, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def build_conflict_bundle(
    selected_sample: Optional[dict] = None,
    prompt_bank_index: Optional[Dict[Tuple[int, str], str]] = None,
) -> HeatmapBundle:
    positions_path = CONFLICT_DIR / "result_all_positions" / "conflict_positions_val.jsonl"
    prompt_bank_path = CONFLICT_DIR / "tuned_lens" / "data" / "val_before_question.jsonl"
    selected_heads_json = (
        CONFLICT_DIR
        / "result_train"
        / "before_question"
        / "headscan_rounds_top50_inf"
        / "head_groups.json"
    )
    if selected_sample is None:
        sample = choose_conflict_error_records(read_jsonl(positions_path), 1)[0]
    else:
        sample = selected_sample

    side = str(sample["side"])
    gold = str(sample["gold"]).strip().lower()
    wrong = str(sample["conflict_label"]).strip().lower()
    position = str(sample.get("position", "before_question"))

    if prompt_bank_index is None:
        prompt_bank_index = build_conflict_prompt_bank_index(read_jsonl(prompt_bank_path))
    base_prompt = prompt_bank_index.get((int(sample["pair_id"]), side))
    if not base_prompt:
        raise KeyError(
            f"Cannot find base prompt for pair_id={sample['pair_id']} side={side} in {prompt_bank_path}"
        )

    model_name = "/root/autodl-tmp/qwen3-4B"
    tokenizer = None
    model = None
    enc = None
    try:
        tokenizer = conflict_ablate.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = conflict_ablate.AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=conflict_ablate.torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        evidence = conflict_ablate.make_evidence(label_yes=(wrong == "yes"))
        marked_evidence = f"<<FIG6_CONFLICT_BEGIN>>{evidence}<<FIG6_CONFLICT_END>>"
        marked_prompt = conflict_ablate.inject_evidence(base_prompt, marked_evidence, position)
        marked_chat = conflict_ablate.wrap_as_chat(tokenizer, marked_prompt, enable_thinking=False)
        chat_prompt = conflict_ablate.wrap_as_chat(
            tokenizer,
            conflict_ablate.inject_evidence(base_prompt, evidence, position),
            enable_thinking=False,
        )

        marked_ids = tokenizer(marked_chat, add_special_tokens=False)["input_ids"]
        conflict_span, _ = vqa_vis.locate_substring_token_span(
            tokenizer,
            marked_ids,
            "<<FIG6_CONFLICT_BEGIN>>",
        )
        conflict_end_span, _ = vqa_vis.locate_substring_token_span(
            tokenizer,
            marked_ids,
            "<<FIG6_CONFLICT_END>>",
        )
        if conflict_span is None or conflict_end_span is None:
            raise RuntimeError("Failed to locate conflict evidence span in the marked prompt.")
        conflict_start = conflict_span[0] + len(tokenizer("<<FIG6_CONFLICT_BEGIN>>", add_special_tokens=False)["input_ids"])
        conflict_end = conflict_end_span[0]
        conflict_span = normalize_prompt_span((conflict_start, conflict_end), len(marked_ids))

        enc = tokenizer(chat_prompt, return_tensors="pt")
        input_ids = enc["input_ids"][0].detach().cpu().tolist()
        question_match = re.search(r"(?is)question\s*:\s*(.*?)\n\s*answer\s*:", base_prompt)
        question_text = question_match.group(1).strip() if question_match else ""
        query_span = None
        if question_text:
            query_span, _ = vqa_vis.locate_substring_token_span(tokenizer, input_ids, question_text)
            query_span = normalize_prompt_span(query_span, len(input_ids))
        enc = move_batch_to_device(enc, next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            out = model(**enc, output_attentions=True, use_cache=False, return_dict=True)
        selected_pairs = parse_head_pairs(selected_heads_json)
        matrix = average_selected_attention(out.attentions, selected_pairs)
        matrix = np.asarray(matrix, dtype=float)
        selected_head_vectors = collect_selected_head_vectors(out.attentions, selected_pairs)

        base_prompt_summary = safe_text(base_prompt.splitlines()[0], 70)
        sample_title = f"pair={sample['pair_id']} | side={side} | gold={gold} | wrong={wrong} | {base_prompt_summary}"

        return HeatmapBundle(
            title="ConflictMedQA Qwen3-4B before_question",
            model_name=model_name,
            sample_title=sample_title,
            matrix=matrix,
            row_labels=head_labels_for_count(len(selected_pairs)),
            x_len=int(matrix.shape[1]),
            visual_span=None,
            query_span=query_span,
            conflict_span=conflict_span,
            selected_heads=selected_pairs,
            selected_head_vectors=selected_head_vectors,
            answer="",
            answer_token_ids=[],
            answer_token_labels=[],
            meta={
                "positions_path": str(positions_path),
                "prompt_bank_path": str(prompt_bank_path),
                "selected_sample": sample,
                "pair_prompt_source": side,
                "sample_kind": "pred_error",
                "base_prompt": base_prompt,
                "query_span": query_span,
                "conflict_span": conflict_span,
                "gold": gold,
                "wrong": wrong,
                "position": position,
                "selected_heads_json": str(selected_heads_json),
            },
        )
    finally:
        release_cuda_objects(enc, tokenizer, model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
def draw_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    color: str,
    label: str,
    side: str = "bottom",
    lw: float = 0.9,
    text_dx: float = 0.0,
    text_dy: Optional[float] = None,
) -> None:
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


def draw_span_box(
    ax: plt.Axes,
    span: Optional[Tuple[int, int]],
    color: str,
    label: str,
    total_rows: int,
    label_side: str = "bottom",
) -> None:
    if span is None:
        return
    start, end = int(span[0]), int(span[1])
    if end <= start:
        return
    x0 = start - 0.5
    x1 = end - 0.5
    ax.axvspan(x0, x1, color=color, alpha=0.065, zorder=0)
    ax.add_patch(
        Rectangle(
            (x0, -0.5),
            end - start,
            max(total_rows, 1),
            fill=False,
            edgecolor=color,
            linewidth=1.05,
            zorder=4,
        )
    )
    if label:
        draw_bracket(
            ax,
            x0,
            x1,
            -0.165 if label_side == "bottom" else 1.02,
            color=color,
            label=label,
            side=label_side,
            text_dy=-0.092 if label_side == "bottom" else 0.072,
        )


def add_zoom_inset(
    ax: plt.Axes,
    matrix: np.ndarray,
    span: Optional[Tuple[int, int]],
    cmap: LinearSegmentedColormap,
    norm: PowerNorm,
    box_color: str,
) -> None:
    if span is None:
        return
    start, end = int(span[0]), int(span[1])
    if end <= start:
        return
    n = int(matrix.shape[0])
    span_len = max(1, end - start)
    pad = max(2, int(round(span_len * 0.18)))
    x0 = max(0, start - pad)
    x1 = min(n, end + pad)
    y0 = max(0, start - pad)
    y1 = min(n, end + pad)

    axins = inset_axes(ax, width="38%", height="38%", loc="upper right", borderpad=1.0)
    axins.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    axins.set_xlim(x0 - 0.5, x1 - 0.5)
    axins.set_ylim(y1 - 0.5, y0 - 0.5)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor("#160018")
    for spine in axins.spines.values():
        spine.set_linewidth(1.1)
        spine.set_edgecolor(box_color)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=box_color, lw=1.0)


def render_heatmap(bundle: HeatmapBundle, output_stem: str, highlight_visual: bool = False) -> Dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(bundle.matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D heatmap matrix, got shape={matrix.shape}")

    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    vmax = float(np.quantile(positive, 0.995)) if positive.size else 1e-6
    vmax = max(vmax, 1e-6)
    norm = PowerNorm(gamma=0.28, vmin=0.0, vmax=vmax)
    cmap = HEATMAP_CMAP

    fig_h = 3.8 if matrix.shape[0] <= 2 else 4.3
    fig, ax = plt.subplots(figsize=(22.0, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    x_ticks = compact_tick_positions(bundle.x_len, 6)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=11)

    if len(bundle.row_labels) == matrix.shape[0]:
        row_labels = bundle.row_labels
    elif matrix.shape[0] == 1:
        row_labels = [bundle.answer]
    else:
        row_labels = [f"tok{i}" for i in range(matrix.shape[0])]
    ax.set_yticks(list(range(matrix.shape[0])))
    ax.set_yticklabels(row_labels, fontsize=12)

    title = bundle.title
    subtitle = bundle.sample_title
    head_text = ", ".join([f"L{l}H{h}" for l, h in bundle.selected_heads])
    fig.suptitle(
        f"{title}\n{subtitle}\naveraged heads: {head_text}",
        fontsize=17,
        fontweight="semibold",
        y=0.98,
    )

    ax.set_xlabel("Prompt token index", fontsize=14)
    ax.set_ylabel("Answer token(s)", fontsize=14)

    if highlight_visual and bundle.visual_span is not None:
        draw_span_box(ax, bundle.visual_span, PALETTE["visual"], "VISUAL TOKENS", matrix.shape[0], "bottom")

    draw_span_box(ax, bundle.conflict_span, PALETTE["conflict"], "CONFLICT TOKENS", matrix.shape[0], "bottom")

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")

    cbar = fig.colorbar(im, ax=ax, pad=0.012, shrink=0.92)
    cbar.set_label("Attention score", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.90])

    png_path = OUT_DIR / f"{output_stem}.png"
    pdf_path = OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"png": str(png_path), "pdf": str(pdf_path)}


def render_comparison(
    conflict_bundle: HeatmapBundle,
    vqa_bundle: HeatmapBundle,
    output_stem: str,
    output_dir: Path,
    gamma: float = 1.35,
    contrast_mode: str = "evidence_relative",
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_groups(bundle: HeatmapBundle, multimodal: bool) -> List[Tuple[str, Tuple[int, int], str]]:
        groups: List[Tuple[str, Tuple[int, int], str]] = []
        if multimodal and bundle.visual_span is not None:
            for idx, span in enumerate(split_span(bundle.visual_span, 3), start=1):
                groups.append((f"V{idx}", span, "#0aa40f"))
        if bundle.query_span is not None:
            for idx, span in enumerate(split_span(bundle.query_span, 3), start=1):
                groups.append((f"Q{idx}", span, "#8f8f8f"))
        if bundle.conflict_span is not None:
            for idx, span in enumerate(split_span(bundle.conflict_span, 3), start=1):
                groups.append((f"CE{idx}", span, "#e80000"))
        return groups

    conflict_groups = make_groups(conflict_bundle, multimodal=False)
    vqa_groups = make_groups(vqa_bundle, multimodal=True)
    conflict_summary, conflict_row_labels = summarize_layers_by_groups(
        conflict_bundle.selected_head_vectors,
        conflict_bundle.selected_heads,
        [(a, b) for a, b, _c in conflict_groups],
    )
    vqa_summary, vqa_row_labels = summarize_layers_by_groups(
        vqa_bundle.selected_head_vectors,
        vqa_bundle.selected_heads,
        [(a, b) for a, b, _c in vqa_groups],
    )

    all_vals = np.concatenate([conflict_summary.ravel(), vqa_summary.ravel()])
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size:
        vmin = float(np.quantile(finite_vals, 0.05))
        vcenter = float(np.quantile(finite_vals, 0.45))
        vmax = float(np.quantile(finite_vals, 0.95))
        if not (vmin < vcenter < vmax):
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
            vcenter = float(np.median(finite_vals))
        if not (vmin < vcenter < vmax):
            eps = max(abs(vcenter) * 1e-3, 1e-6)
            vmin = vcenter - eps
            vmax = vcenter + eps
    else:
        vmin, vcenter, vmax = 0.0, 0.5, 1.0
    cmap = LinearSegmentedColormap.from_list(
        "fig6_strong_diverging",
        [
            (0.00, "#1f4fbf"),
            (0.34, "#6ea3ff"),
            (0.50, "#f7f7f7"),
            (0.56, "#ff8a68"),
            (0.72, "#ef5338"),
            (1.00, "#b10014"),
        ],
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.6, 12.0),
        gridspec_kw={"wspace": 0.01, "width_ratios": [1.42, 1.46]},
    )
    fig.patch.set_facecolor("white")

    panel_specs = [
        (axes[0], conflict_bundle, conflict_summary, conflict_row_labels, conflict_groups, "Text-only"),
        (axes[1], vqa_bundle, vqa_summary, vqa_row_labels, vqa_groups, "Multimodal VQA"),
    ]

    ims = []
    for panel_idx, (ax, bundle, summary_matrix, row_labels, groups, panel_title) in enumerate(panel_specs):
        im = ax.imshow(
            summary_matrix,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(panel_title, fontsize=24, pad=20)
        ax.set_facecolor("white")
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([label for label, _span, _color in groups], rotation=90, fontsize=14)
        ax.tick_params(axis="x", pad=2)
        y_positions, y_labels = compact_row_ticks(row_labels, max_ticks=10)
        ax.set_yticks(y_positions)
        display_y_labels = [label.replace("Layer ", "") for label in y_labels]
        ax.set_yticklabels(display_y_labels, fontsize=13 if len(row_labels) > 24 else 15)
        ax.set_ylabel("Layers" if panel_idx == 0 else "", fontsize=18, labelpad=12 if panel_idx == 0 else 0)
        if panel_idx == 0:
            ax.yaxis.set_label_coords(-0.075, 0.5)
        ax.tick_params(axis="both", length=5, width=1.2, colors="black", labelsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")

        # Group outlines and bottom segment bars.
        start_col = 0
        group_block_start = 0
        while group_block_start < len(groups):
            prefix = re.sub(r"\d+$", "", groups[group_block_start][0])
            group_block_end = group_block_start
            while group_block_end < len(groups) and re.sub(r"\d+$", "", groups[group_block_end][0]) == prefix:
                group_block_end += 1
            color = groups[group_block_start][2]
            draw_group_outline(ax, group_block_start, group_block_end, color, summary_matrix.shape[0], lw=4.0)
            x0 = group_block_start / len(groups)
            width = (group_block_end - group_block_start) / len(groups)
            rect = Rectangle((x0, -0.105), width, 0.018, transform=ax.transAxes, color=color, clip_on=False)
            ax.add_patch(rect)
            block_label = {
                "V": "Vision\ntokens",
                "Q": "Query\ntokens",
                "CE": "Conflict\nevidence tokens",
            }.get(prefix, prefix)
            label_x = x0 + width / 2.0
            if len(groups) > 6:
                if prefix == "V":
                    label_x -= 0.035
                elif prefix == "CE":
                    label_x += 0.035
            ax.text(
                label_x,
                -0.155,
                block_label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=17.6,
                linespacing=0.95,
            )
            group_block_start = group_block_end

        ax.text(
            0.5,
            -0.27,
            "Token Positions",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=24,
        )

    # Central divider.
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.14, 0.86], transform=fig.transFigure, color="#8a8a8a", linewidth=1.8))

    cax = fig.add_axes([0.010, 0.36, 0.016, 0.30])
    cbar = fig.colorbar(ims[0], cax=cax, orientation="vertical")
    cbar.set_ticks([])
    cbar.outline.set_linewidth(1.0)

    fig.subplots_adjust(left=0.04, right=0.992, bottom=0.14, top=0.90)
    axes[0].set_position([0.07, 0.24, 0.38, 0.60])
    axes[1].set_position([0.55, 0.24, 0.38, 0.60])

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "png": str(png_path),
        "pdf": str(pdf_path),
        "vmin": float(vmin),
        "vcenter": float(vcenter),
        "vmax": float(vmax),
    }


def write_joint_summary(
    conflict_bundle: HeatmapBundle,
    vqa_bundle: HeatmapBundle,
    outputs: Dict[str, str],
    stem: str,
    output_dir: Path,
) -> Path:
    summary = {
        "conflict": {
            "title": conflict_bundle.title,
            "model_name": conflict_bundle.model_name,
            "sample_title": conflict_bundle.sample_title,
            "selected_heads": [{"layer": int(l), "head": int(h)} for l, h in conflict_bundle.selected_heads],
            "conflict_span": None
            if conflict_bundle.conflict_span is None
            else [int(conflict_bundle.conflict_span[0]), int(conflict_bundle.conflict_span[1])],
            "matrix_shape": [int(conflict_bundle.matrix.shape[0]), int(conflict_bundle.matrix.shape[1])],
            "meta": conflict_bundle.meta,
        },
        "vqa": {
            "title": vqa_bundle.title,
            "model_name": vqa_bundle.model_name,
            "sample_title": vqa_bundle.sample_title,
            "selected_heads": [{"layer": int(l), "head": int(h)} for l, h in vqa_bundle.selected_heads],
            "conflict_span": None
            if vqa_bundle.conflict_span is None
            else [int(vqa_bundle.conflict_span[0]), int(vqa_bundle.conflict_span[1])],
            "matrix_shape": [int(vqa_bundle.matrix.shape[0]), int(vqa_bundle.matrix.shape[1])],
            "meta": vqa_bundle.meta,
        },
        "outputs": outputs,
        "out_dir": str(OUT_DIR),
    }
    path = output_dir / f"{stem}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary(bundle: HeatmapBundle, outputs: Dict[str, str], stem: str) -> Path:
    summary = {
        "title": bundle.title,
        "model_name": bundle.model_name,
        "sample_title": bundle.sample_title,
        "answer": bundle.answer,
        "answer_token_ids": bundle.answer_token_ids,
        "answer_token_labels": bundle.answer_token_labels,
        "selected_heads": [{"layer": int(l), "head": int(h)} for l, h in bundle.selected_heads],
        "visual_span": None if bundle.visual_span is None else [int(bundle.visual_span[0]), int(bundle.visual_span[1])],
        "conflict_span": None if bundle.conflict_span is None else [int(bundle.conflict_span[0]), int(bundle.conflict_span[1])],
        "matrix_shape": [int(bundle.matrix.shape[0]), int(bundle.matrix.shape[1])],
        "outputs": outputs,
        "meta": bundle.meta,
    }
    path = OUT_DIR / f"{stem}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Fig. 6 attention heatmaps.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of resist samples to render (paired comparison figures).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.35,
        help="PowerNorm gamma for contrast. Larger values make high attention differences easier to see.",
    )
    parser.add_argument(
        "--contrast_mode",
        choices=["raw", "evidence_relative"],
        default="evidence_relative",
        help="Display raw attention or a contrast-enhanced evidence-relative view.",
    )
    parser.add_argument(
        "--use_same_model",
        action="store_true",
        help="Use Hulu-med for both panels: text-only on the left and multimodal on the right.",
    )
    return parser.parse_args()


def make_output_stem(index: int, conflict_bundle: HeatmapBundle, vqa_bundle: HeatmapBundle) -> str:
    if conflict_bundle.meta.get("same_model_text_only"):
        pair_id = f"same_model_row{conflict_bundle.meta.get('row_idx', 'na')}"
    else:
        pair_id = conflict_bundle.meta.get("selected_sample", {}).get("pair_id", "na")
    row_idx = vqa_bundle.meta.get("row_idx", "na")
    return f"fig6_sample_{index:02d}_pair{pair_id}_row{row_idx}"


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num_samples must be >= 1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vqa_pred_rows = read_jsonl(
        VQA_DIR
        / "eval-results_vqarad_hulumed4b_all"
        / "_archive_zengjiaqi_Medical_LLM_Hulu-Med-4B"
        / "before_question"
        / "preds.jsonl"
    )
    vqa_csv_rows = vqa_vis.read_csv_rows(VQA_DIR / "data" / "nc_cc_both_correct_rerun_tmp_val.csv")
    vqa_records = choose_vqa_correct_prediction_records(vqa_pred_rows, vqa_csv_rows, args.num_samples)
    conflict_position_rows = read_jsonl(
        CONFLICT_DIR / "result_all_positions" / "conflict_positions_val.jsonl"
    )
    conflict_prompt_bank_index = build_conflict_prompt_bank_index(
        read_jsonl(CONFLICT_DIR / "tuned_lens" / "data" / "val_before_question.jsonl")
    )
    conflict_samples = filter_conflict_samples_with_prompt_bank(
        choose_conflict_error_records(conflict_position_rows, max(args.num_samples * 4, args.num_samples)),
        conflict_prompt_bank_index,
    )[: args.num_samples]
    batch_count = min(args.num_samples, len(vqa_records), len(conflict_samples))
    if batch_count < args.num_samples:
        print(
            json.dumps(
                {
                    "requested": args.num_samples,
                    "available_vqa": len(vqa_records),
                    "available_conflict": len(conflict_samples),
                    "using": batch_count,
                    "mode": "same_model" if args.use_same_model else "cross_model",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    all_items: List[Dict[str, Any]] = []
    for index in range(batch_count):
        vqa_bundle = build_vqa_bundle(selected_record=vqa_records[index])
        if args.use_same_model:
            conflict_bundle = build_conflict_hulumed_text_only_bundle(
                selected_sample=conflict_samples[index],
                prompt_bank_index=conflict_prompt_bank_index,
            )
        else:
            conflict_bundle = build_conflict_bundle(
                selected_sample=conflict_samples[index],
                prompt_bank_index=conflict_prompt_bank_index,
            )
        stem = make_output_stem(index + 1, conflict_bundle, vqa_bundle)

        outputs = render_comparison(
            conflict_bundle,
            vqa_bundle,
            stem,
            RESULTS_DIR,
            gamma=args.gamma,
            contrast_mode=args.contrast_mode,
        )
        summary = write_joint_summary(
            conflict_bundle,
            vqa_bundle,
            outputs,
            stem,
            RESULTS_DIR,
        )
        item = {
            "index": index + 1,
            "stem": stem,
            "summary": str(summary),
            "outputs": outputs,
            "conflict_bundle": {
                "title": conflict_bundle.title,
                "model_name": conflict_bundle.model_name,
                "sample_title": conflict_bundle.sample_title,
                "selected_sample": conflict_bundle.meta.get("selected_sample"),
                "same_model_text_only": bool(conflict_bundle.meta.get("same_model_text_only")),
            },
            "vqa_bundle": {
                "title": vqa_bundle.title,
                "model_name": vqa_bundle.model_name,
                "sample_title": vqa_bundle.sample_title,
                "row_idx": vqa_bundle.meta.get("row_idx"),
                "chosen_record": vqa_bundle.meta.get("chosen_record"),
            },
        }
        all_items.append(item)

        conflict_bundle = None
        vqa_bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    manifest = {
        "requested_samples": args.num_samples,
        "generated_samples": batch_count,
        "out_dir": str(RESULTS_DIR),
        "items": all_items,
    }
    (RESULTS_DIR / "fig6_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
