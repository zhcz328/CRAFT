#!/usr/bin/env python3
import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path("/root/logit_lens/Slake_vqa/image_conflict")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from analysis_image_conflict import get_answer_candidates_from_row, score_map_to_label
from eval_slake_mm_image_conflict import (
    FULL_IMAGE_NOISE_STD,
    build_messages_multimodal,
    ensure_video_import_compat,
    load_mm_model,
    make_prompt,
    parse_json_list,
    read_input_table,
    resolve_row_path,
    resize_image_if_needed,
    stable_seed_from_text,
    summarize_records,
)
from filter_fine_grained import score_candidate_options_cached
from mask_utils import build_masked_image, extract_inline_mask_data, normalize_answer
from model_utils import load_processor_with_compat, resolve_model_selection


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sweep mask_scale for SLAKE image-conflict evaluation.")
    ap.add_argument(
        "--data_csv",
        default="/root/logit_lens/Slake_vqa/image_conflict/data/hulumed4b/slake_nc_correct_ic_ready.csv",
    )
    ap.add_argument("--image_root", default="/root/autodl-tmp/data/SLAKE/imgs")
    ap.add_argument("--model", default="/root/autodl-tmp/Hulu-Med-4B")
    ap.add_argument("--model_name", default="hulumed-4b")
    ap.add_argument("--out_dir", default="/root/logit_lens/exp/ablation_on_mask_scale")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--resize_max_side", type=int, default=672)
    ap.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plot_name", default="mask_scale_vs_ic_unknown_rate.png")
    ap.add_argument(
        "--empty_cache_every",
        type=int,
        default=10,
        help="Run gc.collect()/torch.cuda.empty_cache() every N samples. Set 0 to disable periodic cache clearing.",
    )
    return ap.parse_args()


def build_scale_values(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values: List[float] = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 10))
        cur += step
    return values


def scale_slug(value: float) -> str:
    text = f"{value:.1f}"
    return text.replace(".", "_")


def runtime_scale(value: float) -> float:
    return max(float(value), 1e-6)


def read_existing_nc_prediction(row: pd.Series) -> tuple[str, str]:
    raw = str(row.get("nc_pred_raw", "")).strip()
    norm = str(row.get("nc_pred", "")).strip()
    if raw:
        return raw, normalize_answer(raw)
    if norm:
        return norm, normalize_answer(norm)
    raise ValueError("Missing nc prediction columns in input CSV.")


def init_summary_accumulator() -> Dict[str, Any]:
    return {
        "n_records": 0,
        "nc_correct": 0,
        "ic_unknown": 0,
        "ic_gold": 0,
        "ic_wrong": 0,
        "ic_other": 0,
        "changed_vs_nc": 0,
        "ic_norm_counter": Counter(),
        "by_category": defaultdict(lambda: {"n": 0, "nc_correct": 0, "ic_unknown": 0, "ic_gold": 0, "ic_wrong": 0, "ic_other": 0}),
    }


def update_summary_accumulator(acc: Dict[str, Any], row: Dict[str, Any]) -> None:
    acc["n_records"] += 1
    category = str(row.get("category", "other") or "other")
    group = acc["by_category"][category]
    group["n"] += 1
    if row.get("nc_correct"):
        acc["nc_correct"] += 1
        group["nc_correct"] += 1
    if row.get("ic_is_unknown"):
        acc["ic_unknown"] += 1
        group["ic_unknown"] += 1
    elif row.get("ic_matches_gold"):
        acc["ic_gold"] += 1
        group["ic_gold"] += 1
    elif row.get("ic_matches_wrong"):
        acc["ic_wrong"] += 1
        group["ic_wrong"] += 1
    else:
        acc["ic_other"] += 1
        group["ic_other"] += 1
    if row.get("changed_vs_nc"):
        acc["changed_vs_nc"] += 1
    acc["ic_norm_counter"][row.get("ic_pred_norm", "")] += 1


def finalize_summary_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "n_records": acc["n_records"],
        "nc_correct": acc["nc_correct"],
        "ic_unknown": acc["ic_unknown"],
        "ic_gold": acc["ic_gold"],
        "ic_wrong": acc["ic_wrong"],
        "ic_other": acc["ic_other"],
        "changed_vs_nc": acc["changed_vs_nc"],
        "ic_top_predictions": acc["ic_norm_counter"].most_common(20),
        "by_category": dict(acc["by_category"]),
    }
    n = max(1, summary["n_records"])
    summary["nc_correct_rate"] = summary["nc_correct"] / n
    summary["ic_unknown_rate"] = summary["ic_unknown"] / n
    summary["ic_gold_rate"] = summary["ic_gold"] / n
    summary["ic_wrong_rate"] = summary["ic_wrong"] / n
    summary["ic_other_rate"] = summary["ic_other"] / n
    summary["changed_vs_nc_rate"] = summary["changed_vs_nc"] / n
    return summary


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = build_scale_values(args.start, args.stop, args.step)
    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    df = read_input_table(args.data_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()

    args.model, _ = resolve_model_selection(model_name=args.model_name, model=args.model)
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    model.eval()
    ensure_video_import_compat()
    processor = load_processor_with_compat(args.model)
    print(f"runtime_dtype={args.dtype}")
    print(f"resize_max_side={args.resize_max_side}")

    scale_dirs = {scale: out_dir / f"mask_scale_{scale_slug(scale)}" for scale in scales}
    for scale_dir in scale_dirs.values():
        scale_dir.mkdir(parents=True, exist_ok=True)
    accumulators = {scale: init_summary_accumulator() for scale in scales}
    pred_paths = {scale: scale_dirs[scale] / "preds.jsonl" for scale in scales}
    pred_handles = {scale: pred_paths[scale].open("w", encoding="utf-8") for scale in scales}
    prompt_cache: Dict[str, str] = {}
    skipped = 0
    try:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mask-scale sweep"):
            target_labels = parse_json_list(row.get("ic_target_labels"))
            img_path = resolve_row_path(row, args.image_root, key="image_path")
            det_path = resolve_row_path(row, args.image_root, key="detection_path")
            mask_path = resolve_row_path(row, args.image_root, key="mask_path")
            inline_mask = extract_inline_mask_data(row)
            if img_path is None or (not target_labels or (inline_mask is None and (det_path is None or mask_path is None))):
                skipped += 1
                continue

            question = str(row.get("question", "")).strip()
            gold_raw = str(row.get("gold", row.get("answer", ""))).strip()
            gold, wrong, unknown, answer_candidates = get_answer_candidates_from_row(row)
            prompt = prompt_cache.setdefault(question, make_prompt(question))
            noise_seed = stable_seed_from_text(f"{idx}\t{row.get('img_id', '')}\t{question}")
            nc_pred_raw, nc_pred_norm = read_existing_nc_prediction(row)

            for scale in scales:
                ic_image = None
                ic_messages = None
                ic_scores = None
                ic_meta = None
                try:
                    ic_image, ic_meta = build_masked_image(
                        source_path=img_path,
                        detection_path=det_path,
                        target_labels=target_labels,
                        mask_path=mask_path,
                        mask_rle=inline_mask["mask_rle"] if inline_mask is not None else None,
                        mask_height=inline_mask["mask_height"] if inline_mask is not None else None,
                        mask_width=inline_mask["mask_width"] if inline_mask is not None else None,
                        mask_all=False,
                        add_noise=False,
                        add_noise_all=False,
                        noise_std=FULL_IMAGE_NOISE_STD,
                        noise_seed=noise_seed,
                        mask_scale=runtime_scale(scale),
                        noise_scale=1.0,
                    )
                    ic_image = resize_image_if_needed(ic_image, args.resize_max_side)
                    ic_messages = build_messages_multimodal(ic_image, prompt)
                    ic_scores = score_candidate_options_cached(model, processor, ic_messages, answer_candidates)
                    ic_pred_raw = max(ic_scores, key=lambda key: ic_scores[key])
                    ic_pred_norm = normalize_answer(ic_pred_raw)
                    _, ic_pred_label = score_map_to_label(ic_scores, gold, wrong, unknown)

                    record = {
                        "id": row.get("id", ""),
                        "qid": row.get("qid", ""),
                        "img_id": row.get("img_id", ""),
                        "img_name": row.get("img_name", ""),
                        "question": question,
                        "gold": gold_raw,
                        "wrong": wrong,
                        "unknown_target": unknown,
                        "category": row.get("category", row.get("content_type", "other")),
                        "content_type": row.get("content_type", ""),
                        "split": row.get("split", ""),
                        "image_path": str(img_path),
                        "detection_path": str(det_path) if det_path is not None else "",
                        "mask_path": str(mask_path) if mask_path is not None else "",
                        "ic_target_labels": target_labels,
                        "answer_candidates": answer_candidates,
                        "mask_scale": float(scale),
                        "runtime_mask_scale": float(runtime_scale(scale)),
                        "nc_pred_raw": nc_pred_raw,
                        "nc_pred_norm": nc_pred_norm,
                        "ic_pred_raw": ic_pred_raw,
                        "ic_pred_norm": ic_pred_norm,
                        "ic_pred_label": ic_pred_label,
                        "ic_scores_json": json.dumps(ic_scores, ensure_ascii=False),
                        "nc_correct": True,
                        "ic_is_unknown": ic_pred_label == "unknown",
                        "ic_matches_gold": ic_pred_label == "gold",
                        "ic_matches_wrong": ic_pred_label == "wrong",
                        "changed_vs_nc": nc_pred_norm != ic_pred_norm,
                        "mask_bbox": ic_meta.get("mask_bbox"),
                        "used_mask_pixels": bool(ic_meta.get("used_mask_pixels", False)),
                        "fallback_rectangles": int(ic_meta.get("fallback_rectangles", 0)),
                        "noise_std": 0.0,
                        "noise_seed": None,
                    }
                    pred_handles[scale].write(json.dumps(record, ensure_ascii=False) + "\n")
                    update_summary_accumulator(accumulators[scale], record)
                finally:
                    del ic_scores, ic_messages, ic_meta
                    if ic_image is not None and hasattr(ic_image, "close"):
                        try:
                            ic_image.close()
                        except Exception:
                            pass
                    del ic_image

            if args.empty_cache_every > 0 and (idx + 1) % args.empty_cache_every == 0:
                release_cuda_memory()
    finally:
        for handle in pred_handles.values():
            handle.close()
        release_cuda_memory()

    aggregate_rows: List[Dict[str, Any]] = []
    for scale in scales:
        scale_dir = scale_dirs[scale]
        preds_path = pred_paths[scale]
        summary_path = scale_dir / "summary.json"

        summary = finalize_summary_accumulator(accumulators[scale])
        summary.update(
            {
                "model": args.model,
                "model_name": args.model_name,
                "data_csv": args.data_csv,
                "image_root": args.image_root,
                "out_dir": str(scale_dir),
                "preds_jsonl": str(preds_path),
                "summary_json": str(summary_path),
                "mask_scale": float(scale),
                "runtime_mask_scale": float(runtime_scale(scale)),
                "skipped": skipped,
            }
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        aggregate_rows.append(
            {
                "mask_scale": float(scale),
                "runtime_mask_scale": float(runtime_scale(scale)),
                "n_records": int(summary["n_records"]),
                "skipped": int(skipped),
                "ic_unknown": int(summary["ic_unknown"]),
                "ic_unknown_rate": float(summary["ic_unknown_rate"]),
                "ic_gold_rate": float(summary["ic_gold_rate"]),
                "ic_wrong_rate": float(summary["ic_wrong_rate"]),
                "changed_vs_nc_rate": float(summary["changed_vs_nc_rate"]),
                "summary_json": str(summary_path),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values("mask_scale").reset_index(drop=True)
    aggregate_csv = out_dir / "mask_scale_sweep_summary.csv"
    aggregate_json = out_dir / "mask_scale_sweep_summary.json"
    aggregate_df.to_csv(aggregate_csv, index=False)
    aggregate_json.write_text(aggregate_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(aggregate_df["mask_scale"], aggregate_df["ic_unknown_rate"], marker="o", linewidth=2)
    plt.xlabel("mask scale")
    plt.ylabel("ic_unknown_rate")
    plt.title("Mask Scale vs IC Unknown Rate")
    plt.grid(True, alpha=0.3)
    plt.xticks(scales, rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / args.plot_name, dpi=200)
    plt.close()

    print(f"saved_csv={aggregate_csv}")
    print(f"saved_json={aggregate_json}")
    print(f"saved_plot={out_dir / args.plot_name}")
    print(aggregate_df.to_string(index=False))


if __name__ == "__main__":
    main()
