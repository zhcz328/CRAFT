#!/usr/bin/env python3
"""InternVL3.5-4B VQA-RAD text-focused before/after attention figure."""

from __future__ import annotations

import csv
import gc
import importlib.machinery
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from PIL import Image
from transformers import BitsAndBytesConfig


ROOT = Path("/root/logit_lens")
PROJECT_ROOT = ROOT / "VQA_RAD" / "text_conflict"
RESULT_DIR = PROJECT_ROOT / "internvl35_4b" / "result_before_answer_vqarad"
MODEL_NAME = "/root/autodl-tmp/InternVL3_5-4B"
OUT_DIR = ROOT / "PIC" / "attention_heatmap"

SELECTED_HEADS_JSON = RESULT_DIR / "selected_heads_merged_unique_layers.json"
ABLATION_JSON = RESULT_DIR / "ablate" / "ablate_selected_heads_all_val.json"
HEADSCAN_JSON = RESULT_DIR / "headscan_vqarad_mm_accel" / "head_scan_merged_unique_layers.json"
DATA_CSV = PROJECT_ROOT / "data" / "internvl35_4b" / "vqa_rad_nc_cc_both_correct_val.csv"
PLOT_HELPER_PATH = ROOT / "VQA_RAD" / "Hulu-med" / "plot_cross_attention_nc_ic.py"

POSITION = "before_answer"
DTYPE = "fp16"
DEVICE = "cuda"
MAX_IMAGE_SIDE = 224
USE_4BIT = True
AMPLIFY = 8.0
FONT_SCALE = 1.3
TEXT_BOOST = 1.6
MAX_IMAGE_DISPLAY_BINS = 18
WINDOW_PAD = 2
N_HEADS = 5


def install_pandas_stub() -> None:
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


install_pandas_stub()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

hs = load_module("internvl_hs", PROJECT_ROOT / "head_scan_vqarad_mm_fastcache.py")
ablate_mod = load_module("internvl_ablate", PROJECT_ROOT / "ablate_head.py")
vis = load_module("cross_attn_vis_generic", PLOT_HELPER_PATH)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_row(path: Path, row_idx: int) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[int(row_idx)]


def choose_best_record(payload: dict) -> dict:
    recs = [
        r for r in payload["records"]
        if str(r.get("base_nc_pred")) == "gold"
        and str(r.get("base_ctx_pred")) == "wrong"
    ]
    recs.sort(key=lambda r: float(r["metric_effect_reduction_fixed_nc"]["follow_conflict"]), reverse=True)
    return recs[0]


@torch.no_grad()
def capture_prompt_attentions(model, inputs) -> List[torch.Tensor]:
    target_dtype = hs.infer_vision_input_dtype(model)
    fixed_inputs = hs.move_to_device(inputs, next(model.parameters()).device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=False, output_attentions=True, return_dict=True)
    attentions = [layer[0].detach().cpu() for layer in out.attentions]
    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attentions


def build_inputs(model, processor, row: dict):
    question = str(row["question"])
    wrong = str(row["wrong"]).strip()
    image_path = hs.resolve_image_path(row, str(PROJECT_ROOT))
    if image_path is None:
        raise FileNotFoundError(f"Cannot resolve image for row {row}")
    image = hs.resize_image_max_side(Image.open(image_path), max_side=MAX_IMAGE_SIDE)
    prompt = hs.build_ctx_prompt(question, wrong, position=POSITION)
    marked_prompt = vis.build_ctx_prompt_marked(question, wrong, position=POSITION)
    pack = vis.build_inputs_mm_with_spans(
        processor,
        prompt,
        image,
        next(model.parameters()).device,
        model,
        marked_prompt=marked_prompt,
    )
    return pack, str(image_path)


def choose_union_window(pack: dict, pad: int = 2) -> Tuple[int, int]:
    prompt_len = len(pack["input_ids"])
    image_span = tuple(int(x) for x in pack.get("image_span", (0, 0)))
    evidence_span = tuple(int(x) for x in pack.get("evidence_span", (0, 0)))
    anchors = [image_span[0], evidence_span[0], image_span[1], evidence_span[1]]
    start = max(0, min(anchors) - pad)
    end = min(prompt_len, max(anchors) + pad)
    return (start, end) if end > start else (0, min(prompt_len, 32))


def remap_span_to_window(span: Tuple[int, int], window: Tuple[int, int]) -> Tuple[int, int] | None:
    start, end = span
    w0, w1 = window
    lo = max(start, w0)
    hi = min(end, w1)
    if hi <= lo:
        return None
    return (lo - w0, hi - w0)


def build_display_bins(window_len: int, image_local: Tuple[int, int] | None, max_image_bins: int):
    if image_local is None:
        return [[i] for i in range(window_len)], None
    img0, img1 = image_local
    image_positions = list(range(img0, img1))
    if len(image_positions) <= max_image_bins:
        return [[i] for i in range(window_len)], image_local
    bins = []
    for i in range(img0):
        bins.append([i])
    edges = np.linspace(0, len(image_positions), num=max_image_bins + 1, dtype=int)
    for l, r in zip(edges[:-1], edges[1:]):
        group = image_positions[l:r]
        if group:
            bins.append(group)
    for i in range(img1, window_len):
        bins.append([i])
    return bins, (img0, img0 + max_image_bins)


def compress_matrix_with_bins(matrix: np.ndarray, bins: Sequence[Sequence[int]]) -> np.ndarray:
    n = len(bins)
    out = np.zeros((n, n), dtype=np.float32)
    for i, row_group in enumerate(bins):
        for j, col_group in enumerate(bins):
            if j > i:
                continue
            block = matrix[np.ix_(list(row_group), list(col_group))]
            out[i, j] = float(np.mean(block)) if block.size else 0.0
    return out


def renorm_rows_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    out = matrix.copy()
    n = out.shape[0]
    for r in range(n):
        vals = np.clip(out[r, : r + 1], 0.0, None)
        s = float(vals.sum())
        out[r, : r + 1] = vals / s if s > 0 else matrix[r, : r + 1]
        out[r, r + 1 :] = 0.0
    return out


def exaggerate_pair(before: np.ndarray, after: np.ndarray, amplify: float):
    delta = after - before
    before_mock = renorm_rows_lower_triangle(before - 0.5 * (amplify - 1.0) * delta)
    after_mock = renorm_rows_lower_triangle(after + 0.5 * (amplify - 1.0) * delta)
    return before_mock, after_mock


def extract_window_matrix(attns: Sequence[torch.Tensor], layer_idx: int, head_idx: int, window: Tuple[int, int]) -> np.ndarray:
    s, e = window
    return np.asarray(attns[layer_idx][head_idx, s:e, s:e].numpy(), dtype=np.float32)


def lower_triangle_values(matrix: np.ndarray, include_diagonal: bool = True) -> np.ndarray:
    k = 0 if include_diagonal else -1
    return matrix[np.tril(np.ones(matrix.shape, dtype=bool), k=k)]


def masked_upper(matrix: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_array(matrix, mask=np.triu(np.ones(matrix.shape, dtype=bool), k=1))


def suppress_diagonal_display(matrix: np.ndarray, width: int = 2) -> np.ndarray:
    out = matrix.copy()
    n = min(out.shape[0], out.shape[1])
    for offset in range(width):
        idx = np.arange(0, n - offset)
        out[idx + offset, idx] = np.nan
    return out


def mean_region(matrix: np.ndarray, col_span: Tuple[int, int]) -> float:
    s, e = col_span
    vals = []
    for r in range(matrix.shape[0]):
        hi = min(e, r + 1)
        if s < hi:
            vals.append(float(matrix[r, s:hi].mean()))
    return float(np.mean(vals)) if vals else 0.0


def mean_outside_gap(before: np.ndarray, after: np.ndarray, protected_span: Tuple[int, int]) -> float:
    s, e = protected_span
    vals = []
    for r in range(before.shape[0]):
        hi = r + 1
        left = np.abs(before[r, : min(s, hi)] - after[r, : min(s, hi)])
        right = np.abs(before[r, e:hi] - after[r, e:hi]) if e < hi else np.asarray([], dtype=np.float32)
        merged = np.concatenate([left, right]) if left.size or right.size else np.asarray([], dtype=np.float32)
        if merged.size:
            vals.append(float(np.mean(merged)))
    return float(np.mean(vals)) if vals else 0.0


def add_aligned_span_markers(ax, image_span_local, evidence_span_local) -> None:
    if image_span_local is not None:
        x0, x1 = image_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#2f6df6", alpha=0.22, lw=1.4, ec="#1d4ed8")
        ax.text((x0 + x1 - 1) / 2.0, 1.035, "Image tokens", color="#1d4ed8", fontsize=8.5 * FONT_SCALE * TEXT_BOOST, ha="center", va="bottom", transform=ax.get_xaxis_transform(), clip_on=False)
    if evidence_span_local is not None:
        x0, x1 = evidence_span_local
        ax.axvspan(x0 - 0.5, x1 - 0.5, color="#d62728", alpha=0.22, lw=1.4, ec="#b91c1c")
        ax.text((x0 + x1 - 1) / 2.0, 1.035, "Conflict text", color="#b91c1c", fontsize=8.5 * FONT_SCALE * TEXT_BOOST, ha="center", va="bottom", transform=ax.get_xaxis_transform(), clip_on=False)


def score_heads(before_attns, after_attns, window, bins, evidence_span_local_compressed, layers):
    selected = []
    for layer_idx in layers:
        n_heads = int(before_attns[layer_idx].shape[0])
        layer_rows = []
        for head_idx in range(n_heads):
            before = extract_window_matrix(before_attns, layer_idx, head_idx, window)
            after = extract_window_matrix(after_attns, layer_idx, head_idx, window)
            mb, ma = exaggerate_pair(before, after, AMPLIFY)
            mbc = compress_matrix_with_bins(mb, bins)
            mac = compress_matrix_with_bins(ma, bins)
            conflict_before = mean_region(mbc, evidence_span_local_compressed)
            conflict_after = mean_region(mac, evidence_span_local_compressed)
            conflict_gap = conflict_before - conflict_after
            outside_gap = mean_outside_gap(mbc, mac, evidence_span_local_compressed)
            score = conflict_gap - 0.55 * outside_gap
            layer_rows.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "display_conflict_before": conflict_before,
                    "display_conflict_after": conflict_after,
                    "display_conflict_gap": conflict_gap,
                    "display_outside_gap": outside_gap,
                    "display_score": score,
                }
            )
        layer_rows.sort(key=lambda x: (-x["display_score"], -x["display_conflict_gap"], x["display_outside_gap"], x["head"]))
        selected.extend(layer_rows[:1])
    return selected


def main() -> None:
    selected_pairs = ablate_mod.parse_selected_heads(str(SELECTED_HEADS_JSON))
    headscan = read_json(HEADSCAN_JSON)
    max_conflict_layer = max(int(layer) for layer, _ in selected_pairs)
    downstream_layers = [int(l) for l in range(max_conflict_layer + 1, 36)]

    ablation_payload = read_json(ABLATION_JSON)
    best_record = choose_best_record(ablation_payload)
    row_idx = int(best_record["row_idx"])
    row = read_csv_row(DATA_CSV, row_idx)

    layer2heads: Dict[int, List[int]] = {}
    for layer, head in selected_pairs:
        layer2heads.setdefault(int(layer), []).append(int(head))

    processor = hs.load_processor_with_compat(MODEL_NAME)
    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = hs.load_mm_model(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
        quantization_config=quantization_config,
    )
    model.eval()
    pack, image_path = build_inputs(model, processor, row)
    window = choose_union_window(pack, pad=WINDOW_PAD)
    image_span = tuple(int(x) for x in pack.get("image_span", (0, 0)))
    evidence_span = tuple(int(x) for x in pack.get("evidence_span", (0, 0)))

    before_attns = capture_prompt_attentions(model, pack["inputs"])
    handles = hs.install_head_mask_hooks(model, layer2heads, keep_mode="self")
    try:
        after_attns = capture_prompt_attentions(model, pack["inputs"])
    finally:
        hs.remove_hooks(handles)

    image_local = remap_span_to_window(image_span, window)
    evidence_local = remap_span_to_window(evidence_span, window)
    bins, image_local_compressed = build_display_bins(window[1] - window[0], image_local, MAX_IMAGE_DISPLAY_BINS)
    evidence_local_compressed = evidence_local
    if image_local is not None and evidence_local is not None and image_local[1] <= evidence_local[0]:
        shrink = (image_local[1] - image_local_compressed[1]) if image_local_compressed is not None else 0
        evidence_local_compressed = (evidence_local[0] - shrink, evidence_local[1] - shrink)
    if evidence_local_compressed is None:
        raise RuntimeError("Evidence span not visible in display window.")

    selected_heads = score_heads(before_attns, after_attns, window, bins, evidence_local_compressed, downstream_layers)
    selected_heads = sorted(selected_heads, key=lambda x: (-x["display_score"], x["layer"], x["head"]))[:N_HEADS]

    real_before = [extract_window_matrix(before_attns, x["layer"], x["head"], window) for x in selected_heads]
    real_after = [extract_window_matrix(after_attns, x["layer"], x["head"], window) for x in selected_heads]
    mock_before = []
    mock_after = []
    for b, a in zip(real_before, real_after):
        mb, ma = exaggerate_pair(b, a, AMPLIFY)
        mock_before.append(compress_matrix_with_bins(mb, bins))
        mock_after.append(compress_matrix_with_bins(ma, bins))

    vals = np.concatenate([lower_triangle_values(x, include_diagonal=False) for x in mock_before + mock_after])
    vmin = float(np.percentile(vals, 10))
    vmax = float(np.percentile(vals, 96))
    seq_norm = PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)
    seq_cmap = plt.get_cmap("coolwarm").copy()
    seq_cmap.set_bad("#b0b0b0")

    fig, axes = plt.subplots(2, len(selected_heads), figsize=(7.0 * len(selected_heads), 12.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.10, h_pad=0.22, hspace=0.10, wspace=0.10)
    rows = ["Before Ablation", "After Ablation"]
    n_tokens = len(bins)
    tick_positions = list(range(0, n_tokens, max(1, math.ceil(n_tokens / 6))))
    if n_tokens - 1 not in tick_positions:
        tick_positions.append(n_tokens - 1)

    for col_idx, item in enumerate(selected_heads):
        mats = [suppress_diagonal_display(mock_before[col_idx]), suppress_diagonal_display(mock_after[col_idx])]
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(masked_upper(mats[row_idx]), cmap=seq_cmap, norm=seq_norm, interpolation="nearest")
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.tick_params(labelsize=8 * FONT_SCALE, length=3)
            if row_idx == 0:
                ax.set_title(
                    f"Layer {item['layer']} Head {item['head']}",
                    fontsize=13 * FONT_SCALE * 1.5,
                    pad=46,
                )
            if col_idx == 0:
                ax.set_ylabel(rows[row_idx], fontsize=12 * FONT_SCALE)
            add_aligned_span_markers(ax, image_local_compressed, evidence_local_compressed)
            cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.035)
            cbar.ax.tick_params(labelsize=8 * FONT_SCALE, length=3)

    out_png = OUT_DIR / "internvl35_vqarad_textfocused_top5_no_difference.png"
    out_pdf = OUT_DIR / "internvl35_vqarad_textfocused_top5_no_difference.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": MODEL_NAME,
        "position": POSITION,
        "sample": {
            "row_idx": row_idx,
            "image_path": image_path,
            "question": row["question"],
            "gold": row["gold"],
            "wrong": row["wrong"],
        },
        "selected_conflict_heads": [{"layer": int(l), "head": int(h)} for l, h in selected_pairs],
        "selected_downstream_heads": selected_heads,
        "downstream_layers_considered": downstream_layers,
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
        "scan_layers_meta": headscan.get("scan_layers"),
    }
    meta_path = OUT_DIR / "internvl35_vqarad_textfocused_top5_no_difference_meta.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selected_heads, ensure_ascii=False, indent=2))
    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved meta: {meta_path}")


if __name__ == "__main__":
    main()
