import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import build_messages_multimodal
from analysis_image_conflict import (
    build_nc_prompt,
    get_answer_candidates_from_row,
    parse_json_list,
    resolve_row_path,
    score_map_to_label,
)
from filter_fine_grained import score_candidate_options_cached
from mask_utils import build_masked_image, extract_inline_mask_data
from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)


def ensure_video_import_compat():
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def load_mm_model(model_name, device_map=None, torch_dtype=None):
    return shared_load_mm_model(model_name, device_map=device_map, torch_dtype=torch_dtype)


def resize_image_max_side(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def stable_seed(*parts: Any) -> int:
    text = "||".join(str(part or "") for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def apply_random_image(image: Image.Image, seed: int) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    rng = np.random.default_rng(seed)
    out = rng.integers(0, 256, size=arr.shape, dtype=np.uint8)
    return Image.fromarray(out, mode="RGB")


def apply_shuffle_patches(image: Image.Image, patch_size: int, seed: int) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    if patch_size <= 1:
        return Image.fromarray(arr, mode="RGB")

    h, w = arr.shape[:2]
    main_h = (h // patch_size) * patch_size
    main_w = (w // patch_size) * patch_size
    if main_h == 0 or main_w == 0:
        return Image.fromarray(arr, mode="RGB")

    main = arr[:main_h, :main_w]
    gh = main_h // patch_size
    gw = main_w // patch_size
    patches = (
        main.reshape(gh, patch_size, gw, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(gh * gw, patch_size, patch_size, 3)
    )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(gh * gw)
    shuffled = patches[perm]
    rebuilt = (
        shuffled.reshape(gh, gw, patch_size, patch_size, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(main_h, main_w, 3)
    )
    out = arr.copy()
    out[:main_h, :main_w] = rebuilt
    return Image.fromarray(out, mode="RGB")


def build_clean_image(row: Any, image_root: str, max_side: int) -> Image.Image:
    img_path = resolve_row_path(row, image_root, "image_path")
    if img_path is None:
        raise FileNotFoundError("Missing image_path")
    return resize_image_max_side(Image.open(img_path).convert("RGB"), max_side=max_side)


def build_ablated_image(
    row: Any,
    image_root: str,
    max_side: int,
    ablation: str,
    mask_scale: float,
    patch_size: int,
) -> Image.Image:
    img_path = resolve_row_path(row, image_root, "image_path")
    det_path = resolve_row_path(row, image_root, "detection_path")
    mask_path = resolve_row_path(row, image_root, "mask_path")
    inline_mask = extract_inline_mask_data(row)
    if img_path is None:
        raise FileNotFoundError("Missing image_path")

    source = Image.open(img_path).convert("RGB")
    row_seed = stable_seed(row.get("id", ""), row.get("qid", ""), row.get("img_id", ""), row.get("question", ""))

    if ablation == "object_mask":
        target_labels = parse_json_list(row.get("ic_target_labels"))
        if not target_labels or (inline_mask is None and (det_path is None or mask_path is None)):
            raise ValueError("object_mask requires ic_target_labels and either inline mask data or detection_path/mask_path")
        image, _meta = build_masked_image(
            source_path=img_path,
            detection_path=det_path,
            target_labels=target_labels,
            mask_path=mask_path,
            mask_rle=inline_mask["mask_rle"] if inline_mask is not None else None,
            mask_height=inline_mask["mask_height"] if inline_mask is not None else None,
            mask_width=inline_mask["mask_width"] if inline_mask is not None else None,
            mask_scale=mask_scale,
        )
        return resize_image_max_side(image, max_side=max_side)

    if ablation == "mask_all":
        image, _meta = build_masked_image(
            source_path=img_path,
            detection_path=det_path,
            target_labels=[],
            mask_path=mask_path,
            mask_rle=inline_mask["mask_rle"] if inline_mask is not None else None,
            mask_height=inline_mask["mask_height"] if inline_mask is not None else None,
            mask_width=inline_mask["mask_width"] if inline_mask is not None else None,
            mask_all=True,
            mask_scale=mask_scale,
        )
        return resize_image_max_side(image, max_side=max_side)

    if ablation == "random":
        return resize_image_max_side(apply_random_image(source, row_seed), max_side=max_side)

    if ablation == "shuffle_patches":
        return resize_image_max_side(apply_shuffle_patches(source, patch_size=patch_size, seed=row_seed), max_side=max_side)

    raise ValueError(f"Unsupported ablation: {ablation}")


def score_closed_set(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    gold: str,
    wrong: str,
    unknown: str,
) -> Dict[str, Any]:
    messages = build_messages_multimodal(image, prompt)
    score_map = score_candidate_options_cached(model, processor, messages, [gold, wrong, unknown])
    gold_lp = float(score_map.get(gold, float("-inf")))
    wrong_lp = float(score_map.get(wrong, float("-inf")))
    unknown_lp = float(score_map.get(unknown, float("-inf")))
    pred_answer, pred_label = score_map_to_label(score_map, gold, wrong, unknown)
    return {
        "candidate_scores": score_map,
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "unknown_lp": unknown_lp,
        "pred_answer": pred_answer,
        "pred_label": pred_label,
        "follow_conflict": float(unknown_lp - gold_lp),
        "follow_context": float(unknown_lp - max(gold_lp, wrong_lp)),
    }


def keep_sample(sample_filter: str, clean_scores: Dict[str, Any], ablated_scores: Dict[str, Any]) -> bool:
    if sample_filter == "none":
        return True
    if sample_filter == "base_gold":
        return clean_scores["pred_label"] == "gold"
    if sample_filter == "base_gold_ablated_nonunknown":
        return clean_scores["pred_label"] == "gold" and ablated_scores["pred_label"] != "unknown"
    raise ValueError(f"Unsupported sample_filter: {sample_filter}")


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    pred_clean = Counter(r["clean_pred_label"] for r in records)
    pred_abl = Counter(r["ablated_pred_label"] for r in records)
    by_category = defaultdict(lambda: {"n": 0, "sum_delta_follow_context": 0.0, "sum_delta_follow_conflict": 0.0, "unknown_flip": 0})

    sum_delta_fc = 0.0
    sum_delta_ff = 0.0
    clean_unknown = 0
    ablated_unknown = 0
    gold_to_unknown = 0

    for r in records:
        sum_delta_fc += float(r["delta_follow_context"])
        sum_delta_ff += float(r["delta_follow_conflict"])
        clean_unknown += int(r["clean_pred_label"] == "unknown")
        ablated_unknown += int(r["ablated_pred_label"] == "unknown")
        gold_to_unknown += int(r["clean_pred_label"] == "gold" and r["ablated_pred_label"] == "unknown")

        group = by_category[str(r.get("category", ""))]
        group["n"] += 1
        group["sum_delta_follow_context"] += float(r["delta_follow_context"])
        group["sum_delta_follow_conflict"] += float(r["delta_follow_conflict"])
        group["unknown_flip"] += int(r["clean_pred_label"] != "unknown" and r["ablated_pred_label"] == "unknown")

    by_category_out = {}
    for key, group in by_category.items():
        cnt = max(1, int(group["n"]))
        by_category_out[key] = {
            "n": int(group["n"]),
            "mean_delta_follow_context": float(group["sum_delta_follow_context"] / cnt),
            "mean_delta_follow_conflict": float(group["sum_delta_follow_conflict"] / cnt),
            "unknown_flip_rate": float(group["unknown_flip"] / cnt),
        }

    return {
        "n_records": n,
        "clean_pred_counts": dict(pred_clean),
        "ablated_pred_counts": dict(pred_abl),
        "mean_delta_follow_context": float(sum_delta_fc / n) if n else 0.0,
        "mean_delta_follow_conflict": float(sum_delta_ff / n) if n else 0.0,
        "clean_unknown_rate": float(clean_unknown / n) if n else 0.0,
        "ablated_unknown_rate": float(ablated_unknown / n) if n else 0.0,
        "unknown_rate_delta": float((ablated_unknown - clean_unknown) / n) if n else 0.0,
        "gold_to_unknown_rate": float(gold_to_unknown / n) if n else 0.0,
        "by_category": by_category_out,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate how image perturbations shift image-conflict samples toward unknown.")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", default=".")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--ablation", default="object_mask", choices=["object_mask", "mask_all", "random", "shuffle_patches"])
    ap.add_argument("--sample_filter", default="base_gold", choices=["none", "base_gold", "base_gold_ablated_nonunknown"])
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_image_side", type=int, default=672)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_summary", required=True)
    args = ap.parse_args()

    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_jsonl = prefix_model_relative_path(args.out_jsonl, model_name=args.model_name, model=args.model)
    args.out_summary = prefix_model_relative_path(args.out_summary, model_name=args.model_name, model=args.model)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    ensure_video_import_compat()
    df = pd.read_csv(args.data_csv)
    if args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    model = load_mm_model(args.model, device_map=args.device_map, torch_dtype=dtype_map[args.dtype])
    model.eval()
    processor = load_processor_with_compat(args.model)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Image eval"):
        question = str(row.get("question", "")).strip()
        gold, wrong, unknown, _cands = get_answer_candidates_from_row(row)
        if not question or not gold or not unknown:
            skipped += 1
            continue

        try:
            clean_image = build_clean_image(row, args.image_root, args.max_image_side)
            ablated_image = build_ablated_image(
                row,
                image_root=args.image_root,
                max_side=args.max_image_side,
                ablation=args.ablation,
                mask_scale=args.mask_scale,
                patch_size=args.patch_size,
            )
        except Exception:
            skipped += 1
            continue

        prompt = build_nc_prompt(question)
        try:
            clean_scores = score_closed_set(model, processor, clean_image, prompt, gold, wrong, unknown)
            ablated_scores = score_closed_set(model, processor, ablated_image, prompt, gold, wrong, unknown)
        except Exception:
            skipped += 1
            continue

        if not keep_sample(args.sample_filter, clean_scores, ablated_scores):
            continue

        rec = {
            "row_idx": int(idx),
            "id": row.get("id", ""),
            "qid": row.get("qid", ""),
            "img_id": row.get("img_id", ""),
            "question": question,
            "category": row.get("category", row.get("content_type", "")),
            "content_type": row.get("content_type", ""),
            "ablation": args.ablation,
            "sample_filter": args.sample_filter,
            "gold": gold,
            "wrong": wrong,
            "unknown": unknown,
            "clean_pred_answer": clean_scores["pred_answer"],
            "clean_pred_label": clean_scores["pred_label"],
            "ablated_pred_answer": ablated_scores["pred_answer"],
            "ablated_pred_label": ablated_scores["pred_label"],
            "clean_follow_context": clean_scores["follow_context"],
            "ablated_follow_context": ablated_scores["follow_context"],
            "delta_follow_context": float(ablated_scores["follow_context"] - clean_scores["follow_context"]),
            "clean_follow_conflict": clean_scores["follow_conflict"],
            "ablated_follow_conflict": ablated_scores["follow_conflict"],
            "delta_follow_conflict": float(ablated_scores["follow_conflict"] - clean_scores["follow_conflict"]),
            "clean_candidate_scores": clean_scores["candidate_scores"],
            "ablated_candidate_scores": ablated_scores["candidate_scores"],
        }
        records.append(rec)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_records(records)
    summary.update(
        {
            "model": args.model,
            "model_name": args.model_name,
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "ablation": args.ablation,
            "sample_filter": args.sample_filter,
            "mask_scale": float(args.mask_scale),
            "patch_size": int(args.patch_size),
            "max_samples": int(args.max_samples),
            "max_image_side": int(args.max_image_side),
            "skipped": int(skipped),
            "out_jsonl": str(out_jsonl),
        }
    )
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_jsonl={out_jsonl}")
    print(f"saved_summary={out_summary}")
    print(json.dumps(
        {
            "n_records": summary["n_records"],
            "mean_delta_follow_context": summary["mean_delta_follow_context"],
            "mean_delta_follow_conflict": summary["mean_delta_follow_conflict"],
            "unknown_rate_delta": summary["unknown_rate_delta"],
            "gold_to_unknown_rate": summary["gold_to_unknown_rate"],
            "skipped": summary["skipped"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
