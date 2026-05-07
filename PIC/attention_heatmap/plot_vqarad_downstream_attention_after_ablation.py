#!/usr/bin/env python3
"""Plot downstream prompt self-attention heatmaps before/after conflict-head ablation.

This script:
1. Loads the stable conflict heads selected for VQA-RAD Hulu-Med-4B.
2. Picks the strongest recovered validation example from the saved ablation JSON.
3. Captures prompt self-attention on the conflict-context prompt before and after
   ablating the selected conflict heads.
4. Scores downstream, non-ablated heads by how much their local attention matrix
   changes, then plots the top 3 heads in a 3x3 grid:
      row 1: Before Ablation
      row 2: After Ablation
      row 3: Difference
"""

from __future__ import annotations

import csv
import gc
import importlib.machinery
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from PIL import Image


ROOT = Path("/root/logit_lens")
PROJECT_ROOT = ROOT / "VQA_RAD" / "Hulu-med"
OUT_DIR = ROOT / "PIC" / "attention_heatmap"

SELECTED_HEADS_JSON = PROJECT_ROOT / "result_train_hulumed4b_before_question" / "headscan_vqarad_mm_hulumed4b_before_question" / "selected_heads_stable_hulumed4b.json"
HEADSCAN_JSON = PROJECT_ROOT / "result_train_hulumed4b_before_question" / "headscan_vqarad_mm_hulumed4b_before_question" / "head_scan_merged_unique_layers.json"
MODEL_NAME = "/root/autodl-tmp/Hulu-Med-4B"

POSITION = "before_question"
DTYPE = "bf16"
DEVICE = "cuda"
MAX_IMAGE_SIDE = 672
MAX_WINDOW_TOKENS = 28
N_HEADS_TO_PLOT = 3
TOP_RECORDS_PER_SPLIT = 8

DATASET_CONFIGS = [
    {
        "name": "train",
        "ablation_json": PROJECT_ROOT / "result_train_hulumed4b_before_question" / "ablate_selected_heads_ctx_only_hulumed4b_train.json",
        "data_csv": PROJECT_ROOT / "data" / "nc_cc_both_correct_rerun_tmp_train.csv",
    },
    {
        "name": "val",
        "ablation_json": PROJECT_ROOT / "result_train_hulumed4b_before_question" / "ablate_selected_heads_ctx_only_hulumed4b_val.json",
        "data_csv": PROJECT_ROOT / "data" / "nc_cc_both_correct_rerun_tmp_val.csv",
    },
    {
        "name": "all",
        "ablation_json": PROJECT_ROOT / "result_train_hulumed4b_before_question" / "ablate_selected_heads_ctx_only_hulumed4b_all.json",
        "data_csv": PROJECT_ROOT / "data" / "nc_cc_both_correct_rerun_tmp.csv",
    },
]


def install_pandas_stub() -> None:
    """Provide the tiny subset needed by imported helper modules."""
    if "pandas" in sys.modules:
        return
    stub = types.ModuleType("pandas")
    stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)

    def isna(value):
        try:
            return value != value
        except Exception:
            return value is None

    stub.isna = isna
    stub.notna = lambda value: not isna(value)
    stub.DataFrame = object
    stub.Series = object
    sys.modules["pandas"] = stub


install_pandas_stub()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ablate_head as ablate_mod  # type: ignore  # noqa: E402
import head_scan_vqarad_mm_fastcache as hs  # type: ignore  # noqa: E402
import plot_cross_attention_nc_ic as vis  # type: ignore  # noqa: E402


@dataclass
class HeadDelta:
    layer: int
    head: int
    score: float
    before_mass: float
    after_mass: float
    delta_mass: float


@dataclass
class SampleSelection:
    dataset: str
    row_idx: int
    record: dict
    row: dict
    window: Tuple[int, int]
    input_ids: List[int]
    scored_heads: List[HeadDelta]
    selected_heads: List[HeadDelta]
    sample_score: float
    max_head_score: float
    image_path: str


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_row(path: Path, row_idx: int) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[int(row_idx)]


def shortlist_records(configs: Sequence[dict], top_per_split: int) -> List[Tuple[str, Path, dict]]:
    candidates: List[Tuple[str, Path, dict]] = []
    for cfg in configs:
        payload = read_json(cfg["ablation_json"])
        records = payload.get("records", [])
        usable = [
            r for r in records
            if str(r.get("base_nc_pred")) == "gold"
            and str(r.get("base_ctx_pred")) == "wrong"
        ]
        usable.sort(
            key=lambda r: float(r.get("metric_effect_reduction_fixed_nc", {}).get("follow_conflict", 0.0)),
            reverse=True,
        )
        for record in usable[:top_per_split]:
            candidates.append((cfg["name"], cfg["data_csv"], record))
    return candidates


def collect_candidate_layers(selected_pairs: Sequence[Tuple[int, int]], scan_layers: Sequence[int]) -> List[int]:
    max_conflict_layer = max(int(layer) for layer, _ in selected_pairs)
    downstream = sorted({int(layer) for layer in scan_layers if int(layer) > max_conflict_layer})
    if not downstream:
        raise RuntimeError("No downstream layers found after the selected conflict heads.")
    return downstream


@torch.no_grad()
def capture_prompt_attentions(model, inputs) -> Tuple[List[torch.Tensor], List[int]]:
    target_dtype = hs.infer_vision_input_dtype(model)
    fixed_inputs = hs.move_to_device(inputs, next(model.parameters()).device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=False, output_attentions=True, return_dict=True)
    if not getattr(out, "attentions", None):
        raise RuntimeError("Model did not return prompt attentions.")
    attentions = [layer[0].detach().float().cpu() for layer in out.attentions]
    input_ids = fixed_inputs["input_ids"][0].detach().cpu().tolist()
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attentions, input_ids


def choose_plot_window(pack: dict, max_tokens: int) -> Tuple[int, int]:
    prompt_len = len(pack["input_ids"])
    ev0, ev1 = [int(x) for x in pack.get("evidence_span", (0, 0))]
    start = ev0 if ev1 > ev0 else max(0, prompt_len - max_tokens)
    end = prompt_len
    if end - start > max_tokens:
        start = end - max_tokens
    start = max(0, min(start, prompt_len))
    end = max(start, min(end, prompt_len))
    return start, end


def extract_window_matrix(attentions: Sequence[torch.Tensor], layer_idx: int, head_idx: int, window: Tuple[int, int]) -> np.ndarray:
    start, end = window
    mat = attentions[layer_idx][head_idx, start:end, start:end].numpy()
    return np.asarray(mat, dtype=np.float32)


def lower_triangle_values(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray:
    k = 0 if include_diagonal else -1
    mask = np.tril(np.ones(matrix.shape, dtype=bool), k=k)
    return matrix[mask]


def score_head_changes(
    before_attns: Sequence[torch.Tensor],
    after_attns: Sequence[torch.Tensor],
    candidate_layers: Sequence[int],
    excluded_pairs: Sequence[Tuple[int, int]],
    window: Tuple[int, int],
) -> List[HeadDelta]:
    excluded = {tuple(map(int, pair)) for pair in excluded_pairs}
    scored: List[HeadDelta] = []
    for layer_idx in candidate_layers:
        n_heads = int(before_attns[layer_idx].shape[0])
        for head_idx in range(n_heads):
            if (int(layer_idx), int(head_idx)) in excluded:
                continue
            before = extract_window_matrix(before_attns, layer_idx, head_idx, window)
            after = extract_window_matrix(after_attns, layer_idx, head_idx, window)
            delta = after - before
            score = float(np.mean(np.abs(lower_triangle_values(delta))))
            before_mass = float(np.mean(lower_triangle_values(before)))
            after_mass = float(np.mean(lower_triangle_values(after)))
            scored.append(
                HeadDelta(
                    layer=int(layer_idx),
                    head=int(head_idx),
                    score=score,
                    before_mass=before_mass,
                    after_mass=after_mass,
                    delta_mass=after_mass - before_mass,
                )
            )
    scored.sort(key=lambda item: (-item.score, item.layer, item.head))
    return scored


def select_plot_heads(scored: Sequence[HeadDelta], n: int) -> List[HeadDelta]:
    chosen: List[HeadDelta] = []
    used_layers = set()
    for item in scored:
        if item.layer in used_layers:
            continue
        chosen.append(item)
        used_layers.add(item.layer)
        if len(chosen) == n:
            return chosen
    for item in scored:
        if any((item.layer == x.layer and item.head == x.head) for x in chosen):
            continue
        chosen.append(item)
        if len(chosen) == n:
            break
    return chosen


def evaluate_candidate(
    model,
    processor,
    row: dict,
    row_idx: int,
    dataset_name: str,
    record: dict,
    selected_pairs: Sequence[Tuple[int, int]],
    candidate_layers: Sequence[int],
) -> SampleSelection:
    layer2heads: Dict[int, List[int]] = {}
    for layer_idx, head_idx in selected_pairs:
        layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))

    image_path = hs.resolve_image_path(row, str(PROJECT_ROOT))
    if image_path is None:
        raise FileNotFoundError(f"Could not resolve image for row_idx={row_idx} in {dataset_name}")

    question = str(row["question"])
    wrong = str(row["wrong"]).strip()
    image = hs.resize_image_max_side(Image.open(image_path), max_side=MAX_IMAGE_SIDE)
    ctx_prompt = hs.build_ctx_prompt(question, wrong, position=POSITION)
    ctx_prompt_marked = vis.build_ctx_prompt_marked(question, wrong, position=POSITION)
    pack = vis.build_inputs_mm_with_spans(
        processor,
        ctx_prompt,
        image,
        next(model.parameters()).device,
        model,
        marked_prompt=ctx_prompt_marked,
    )
    ctx_inputs = pack["inputs"]
    input_ids = pack["input_ids"]
    plot_window = choose_plot_window(pack, MAX_WINDOW_TOKENS)

    before_attns, _ = capture_prompt_attentions(model, ctx_inputs)
    handles = hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_attns, _ = capture_prompt_attentions(model, ctx_inputs)
    finally:
        hs.remove_hooks(handles)

    scored = score_head_changes(before_attns, after_attns, candidate_layers, selected_pairs, plot_window)
    selected = select_plot_heads(scored, N_HEADS_TO_PLOT)
    sample_score = float(np.mean([item.score for item in selected])) if selected else 0.0
    max_head_score = float(max((item.score for item in scored), default=0.0))
    return SampleSelection(
        dataset=dataset_name,
        row_idx=int(row_idx),
        record=record,
        row=row,
        window=plot_window,
        input_ids=[int(x) for x in input_ids],
        scored_heads=scored,
        selected_heads=selected,
        sample_score=sample_score,
        max_head_score=max_head_score,
        image_path=str(image_path),
    )


def decode_token_window(tokenizer, input_ids: Sequence[int], window: Tuple[int, int]) -> List[str]:
    start, end = window
    tokens = []
    for token_id in input_ids[start:end]:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        piece = piece.replace("\n", "\\n")
        tokens.append(piece if piece else str(token_id))
    return tokens


def masked_for_plot(matrix: np.ndarray) -> np.ma.MaskedArray:
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    return np.ma.masked_array(matrix, mask=mask)


def plot_grid(
    selected_heads: Sequence[HeadDelta],
    before_attns: Sequence[torch.Tensor],
    after_attns: Sequence[torch.Tensor],
    window: Tuple[int, int],
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    before_mats = [extract_window_matrix(before_attns, item.layer, item.head, window) for item in selected_heads]
    after_mats = [extract_window_matrix(after_attns, item.layer, item.head, window) for item in selected_heads]
    diff_mats = [after - before for before, after in zip(before_mats, after_mats)]

    offdiag_vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in before_mats + after_mats])
    if offdiag_vals.size == 0:
        offdiag_vals = np.concatenate([lower_triangle_values(x) for x in before_mats + after_mats])
    diff_vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in diff_mats])
    if diff_vals.size == 0:
        diff_vals = np.concatenate([lower_triangle_values(x) for x in diff_mats])

    seq_vmin = float(np.percentile(offdiag_vals, 1))
    seq_vmax = float(np.percentile(offdiag_vals, 99.5))
    if seq_vmax <= seq_vmin:
        seq_vmin = float(np.min(offdiag_vals))
        seq_vmax = float(np.max(offdiag_vals))
    seq_norm = PowerNorm(gamma=0.65, vmin=seq_vmin, vmax=seq_vmax)

    diff_abs = float(np.percentile(np.abs(diff_vals), 99.5))
    if diff_abs <= 0:
        diff_abs = float(np.max(np.abs(diff_vals))) if diff_vals.size else 1e-6
    diff_abs = max(diff_abs, 1e-6)
    diff_norm = TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs)

    seq_cmap = plt.get_cmap("coolwarm").copy()
    seq_cmap.set_bad(color="#8d8d8d")
    diff_cmap = plt.get_cmap("coolwarm").copy()
    diff_cmap.set_bad(color="#8d8d8d")

    fig, axes = plt.subplots(3, len(selected_heads), figsize=(4.4 * len(selected_heads), 12.0), constrained_layout=True)
    if len(selected_heads) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    row_names = ["Before Ablation", "After Ablation", "Difference"]
    tick_positions = list(range(0, window[1] - window[0], max(1, math.ceil((window[1] - window[0]) / 6))))
    if (window[1] - window[0] - 1) not in tick_positions:
        tick_positions.append(window[1] - window[0] - 1)
    tick_positions = sorted(set(int(x) for x in tick_positions if x >= 0))

    for col_idx, item in enumerate(selected_heads):
        mats = [before_mats[col_idx], after_mats[col_idx], diff_mats[col_idx]]
        norms = [seq_norm, seq_norm, diff_norm]
        cmaps = [seq_cmap, seq_cmap, diff_cmap]
        for row_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(masked_for_plot(mats[row_idx]), cmap=cmaps[row_idx], norm=norms[row_idx], interpolation="nearest")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=8, length=2)
            if row_idx == 0:
                ax.set_title(f"Layer {item.layer} Head {item.head}", fontsize=13, pad=8)
            if col_idx == 0:
                ax.set_ylabel(row_names[row_idx], fontsize=12)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_pairs = ablate_mod.parse_selected_heads(str(SELECTED_HEADS_JSON))
    scan_layers = read_json(HEADSCAN_JSON)["scan_layers"]
    candidate_layers = collect_candidate_layers(selected_pairs, scan_layers)
    shortlist = shortlist_records(DATASET_CONFIGS, TOP_RECORDS_PER_SPLIT)
    print(f"Shortlisted {len(shortlist)} candidates across {len(DATASET_CONFIGS)} dataset configs.")

    model, processor = hs.load_model_and_processor(MODEL_NAME, DEVICE, DTYPE)
    best: SampleSelection | None = None
    best_before_attns: List[torch.Tensor] | None = None
    best_after_attns: List[torch.Tensor] | None = None

    for idx, (dataset_name, data_csv, record) in enumerate(shortlist, start=1):
        row_idx = int(record["row_idx"])
        eff = float(record.get("metric_effect_reduction_fixed_nc", {}).get("follow_conflict", 0.0))
        print(f"[{idx}/{len(shortlist)}] Evaluate {dataset_name} row {row_idx} effect={eff:.4f}")
        row = read_csv_row(data_csv, row_idx)
        question = str(row["question"])
        wrong = str(row["wrong"]).strip()
        image_path = hs.resolve_image_path(row, str(PROJECT_ROOT))
        if image_path is None:
            continue
        image = hs.resize_image_max_side(Image.open(image_path), max_side=MAX_IMAGE_SIDE)
        ctx_prompt = hs.build_ctx_prompt(question, wrong, position=POSITION)
        ctx_prompt_marked = vis.build_ctx_prompt_marked(question, wrong, position=POSITION)
        pack = vis.build_inputs_mm_with_spans(
            processor,
            ctx_prompt,
            image,
            next(model.parameters()).device,
            model,
            marked_prompt=ctx_prompt_marked,
        )
        ctx_inputs = pack["inputs"]
        before_attns, _ = capture_prompt_attentions(model, ctx_inputs)
        layer2heads: Dict[int, List[int]] = {}
        for layer_idx, head_idx in selected_pairs:
            layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
        handles = hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
        try:
            after_attns, _ = capture_prompt_attentions(model, ctx_inputs)
        finally:
            hs.remove_hooks(handles)
        plot_window = choose_plot_window(pack, MAX_WINDOW_TOKENS)
        scored = score_head_changes(before_attns, after_attns, candidate_layers, selected_pairs, plot_window)
        selected = select_plot_heads(scored, N_HEADS_TO_PLOT)
        if len(selected) < N_HEADS_TO_PLOT:
            del before_attns, after_attns, scored, selected
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        sample_score = float(np.mean([item.score for item in selected]))
        max_head_score = float(max(item.score for item in scored))
        current = SampleSelection(
            dataset=dataset_name,
            row_idx=row_idx,
            record=record,
            row=row,
            window=plot_window,
            input_ids=[int(x) for x in pack["input_ids"]],
            scored_heads=scored,
            selected_heads=selected,
            sample_score=sample_score,
            max_head_score=max_head_score,
            image_path=str(image_path),
        )
        if best is None or (current.sample_score, current.max_head_score) > (best.sample_score, best.max_head_score):
            best = current
            best_before_attns = before_attns
            best_after_attns = after_attns
            print(
                "  -> new best:",
                {
                    "dataset": current.dataset,
                    "row_idx": current.row_idx,
                    "sample_score": round(current.sample_score, 6),
                    "max_head_score": round(current.max_head_score, 6),
                },
            )
        else:
            del before_attns, after_attns
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best is None or best_before_attns is None or best_after_attns is None:
        raise RuntimeError("Could not find a valid best sample across dataset splits.")

    row = best.row
    row_idx = best.row_idx
    gold = str(row["gold"]).strip()
    wrong = str(row["wrong"]).strip()
    question = str(row["question"])
    selected_plot_heads = best.selected_heads
    plot_window = best.window
    input_ids = best.input_ids

    sample_stub = f"{best.dataset}_row{row_idx:04d}"
    out_png = OUT_DIR / f"{sample_stub}_downstream_attention_before_after_diff.png"
    out_pdf = OUT_DIR / f"{sample_stub}_downstream_attention_before_after_diff.pdf"
    meta_path = OUT_DIR / f"{sample_stub}_downstream_attention_before_after_diff_meta.json"

    title = (
        "VQA-RAD Hulu-Med-4B Downstream Attention Routing "
        f"({best.dataset}, row {row_idx}, conflict heads ablated)"
    )
    plot_grid(selected_plot_heads, best_before_attns, best_after_attns, plot_window, title, out_png, out_pdf)

    tokenizer = processor.tokenizer
    meta = {
        "model": MODEL_NAME,
        "position": POSITION,
        "sample": {
            "dataset": best.dataset,
            "row_idx": row_idx,
            "img_id": row.get("img_id"),
            "image_path": best.image_path,
            "question": question,
            "gold": gold,
            "wrong": wrong,
        },
        "selection": {
            "selected_heads_json": str(SELECTED_HEADS_JSON),
            "searched_datasets": [
                {
                    "name": cfg["name"],
                    "ablation_json": str(cfg["ablation_json"]),
                    "data_csv": str(cfg["data_csv"]),
                }
                for cfg in DATASET_CONFIGS
            ],
            "shortlist_top_per_split": TOP_RECORDS_PER_SPLIT,
            "chosen_record": best.record,
            "ablated_conflict_heads": [{"layer": int(l), "head": int(h)} for l, h in selected_pairs],
            "downstream_candidate_layers": [int(x) for x in candidate_layers],
            "best_sample_score_mean_top3": best.sample_score,
            "best_sample_score_max_head": best.max_head_score,
            "selected_downstream_heads": [
                {
                    "layer": item.layer,
                    "head": item.head,
                    "mean_abs_delta": item.score,
                    "before_mean_lower_triangle": item.before_mass,
                    "after_mean_lower_triangle": item.after_mass,
                    "delta_mean_lower_triangle": item.delta_mass,
                }
                for item in selected_plot_heads
            ],
            "top_ranked_candidates": [
                {
                    "layer": item.layer,
                    "head": item.head,
                    "mean_abs_delta": item.score,
                }
                for item in best.scored_heads[:20]
            ],
        },
        "plot_window": {
            "start": int(plot_window[0]),
            "end": int(plot_window[1]),
            "length": int(plot_window[1] - plot_window[0]),
            "token_ids": [int(x) for x in input_ids[plot_window[0] : plot_window[1]]],
            "token_pieces": decode_token_window(tokenizer, input_ids, plot_window),
            "evidence_span": [int(x) for x in pack.get("evidence_span", (0, 0))],
            "image_span": [int(x) for x in pack.get("image_span", (0, 0))],
        },
        "outputs": {
            "png": str(out_png),
            "pdf": str(out_pdf),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["selection"]["selected_downstream_heads"], ensure_ascii=False, indent=2))
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
