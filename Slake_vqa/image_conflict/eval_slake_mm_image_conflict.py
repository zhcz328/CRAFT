# -*- coding: utf-8 -*-
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from mask_utils import UNKNOWN_ANSWER, build_masked_image, extract_inline_mask_data, normalize_answer
from analysis_image_conflict import get_answer_candidates_from_row, score_map_to_label
from filter_fine_grained import score_candidate_options_cached
from model_utils import (
    build_chat_template_inputs,
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir


SYSTEM_PROMPT = "You are a helpful medical QA assistant."
FULL_IMAGE_NOISE_STD = 64.0
BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "If the image does not contain enough information to answer the question, output ONLY unknown.\n"
    "Output ONLY the final answer.\n\n"
)


def ensure_video_import_compat() -> None:
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def load_mm_model(model_name, device_map=None, torch_dtype=None, trust_remote_code=True, attn_implementation=None, quantization_config=None):
    return shared_load_mm_model(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )


def make_prompt(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def build_messages_multimodal(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]


def move_to_device(obj, device, float_dtype=None):
    if torch.is_tensor(obj):
        if float_dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=float_dtype)
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device, float_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device, float_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device, float_dtype) for v in obj)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def infer_model_dtype(model):
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return None


def resolve_row_path(row, image_root: str, key: str) -> Path | None:
    root = Path(image_root)
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None

    path = Path(raw)
    candidates = [path, root / path]
    parts = list(path.parts)
    for idx in range(1, len(parts)):
        candidates.append(root / Path(*parts[idx:]))
    win_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if win_match:
        drive = win_match.group(1).lower()
        tail = win_match.group(2).replace("\\", "/")
        candidates.append(Path(f"/mnt/{drive}/{tail}"))

    wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", raw)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        tail = wsl_match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{tail}"))

    seen = set()
    for candidate in candidates:
        key_name = str(candidate)
        if key_name in seen:
            continue
        seen.add(key_name)
        if candidate.exists():
            return candidate
    return None


def resize_image_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


def parse_json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return []


def stable_seed_from_text(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def read_input_table(path: str) -> pd.DataFrame:
    in_path = Path(path)
    if in_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(in_path, engine="openpyxl")
    return pd.read_csv(in_path)


def resume_key_for_row(row, idx: int) -> str:
    return f"{idx}\t{row.get('img_id', '')}\t{row.get('question', '')}\t{row.get('gold', row.get('answer', ''))}"


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "n_records": len(records),
        "nc_correct": 0,
        "ic_unknown": 0,
        "ic_gold": 0,
        "ic_wrong": 0,
        "ic_other": 0,
        "changed_vs_nc": 0,
        "ic_norm_counter": Counter(),
        "by_category": defaultdict(lambda: {"n": 0, "nc_correct": 0, "ic_unknown": 0, "ic_gold": 0, "ic_wrong": 0, "ic_other": 0}),
    }
    for row in records:
        category = str(row.get("category", "other") or "other")
        group = summary["by_category"][category]
        group["n"] += 1
        if row.get("nc_correct"):
            summary["nc_correct"] += 1
            group["nc_correct"] += 1
        if row.get("ic_is_unknown"):
            summary["ic_unknown"] += 1
            group["ic_unknown"] += 1
        elif row.get("ic_matches_gold"):
            summary["ic_gold"] += 1
            group["ic_gold"] += 1
        elif row.get("ic_matches_wrong"):
            summary["ic_wrong"] += 1
            group["ic_wrong"] += 1
        else:
            summary["ic_other"] += 1
            group["ic_other"] += 1
        if row.get("changed_vs_nc"):
            summary["changed_vs_nc"] += 1
        summary["ic_norm_counter"][row.get("ic_pred_norm", "")] += 1

    n = max(1, summary["n_records"])
    summary["nc_correct_rate"] = summary["nc_correct"] / n
    summary["ic_unknown_rate"] = summary["ic_unknown"] / n
    summary["ic_gold_rate"] = summary["ic_gold"] / n
    summary["ic_wrong_rate"] = summary["ic_wrong"] / n
    summary["ic_other_rate"] = summary["ic_other"] / n
    summary["changed_vs_nc_rate"] = summary["changed_vs_nc"] / n
    summary["ic_top_predictions"] = summary["ic_norm_counter"].most_common(20)
    summary["by_category"] = dict(summary["by_category"])
    del summary["ic_norm_counter"]
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate SLAKE image conflict with NC original images and IC masked images.")
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--dataset", default="slake_vqa")
    ap.add_argument("--out_dir", default="eval-results_slake_image_conflict")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resize_max_side", type=int, default=672)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--debug_mask_dir", default="")
    ap.add_argument("--save_processed_images", action="store_true", help="Save every processed IC image under out_dir/pic.")
    ap.add_argument("--mask_all", action="store_true", help="Use a full black image for IC instead of a target-only masked image.")
    ap.add_argument("--add_noise", action="store_true", help="Use target-region Gaussian noise for IC instead of a target-only masked image.")
    ap.add_argument("--noise_scale", type=float, default=1.0, help="Scale factor for the target region when applying local noise.")
    ap.add_argument("--mask_scale", type=float, default=1.0, help="Scale factor for the target region when applying local masking.")
    ap.add_argument("--add_noise_all", action="store_true", help="Use a full-image Gaussian-noise IC image instead of a target-only masked image.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if sum(bool(flag) for flag in (args.mask_all, args.add_noise, args.add_noise_all)) > 1:
        ap.error("--mask_all, --add_noise, and --add_noise_all are mutually exclusive.")
    if args.mask_scale <= 0 or args.noise_scale <= 0:
        ap.error("--mask_scale and --noise_scale must be positive.")
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_dir = prefix_model_relative_path(args.out_dir, model_name=args.model_name, model=args.model)

    df = read_input_table(args.data_csv)
    dtype_map = {"auto": None, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    model.eval()
    ensure_video_import_compat()
    processor = load_processor_with_compat(args.model)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    debug_mask_dir = Path(args.debug_mask_dir) if args.debug_mask_dir else None
    if debug_mask_dir is not None:
        debug_mask_dir.mkdir(parents=True, exist_ok=True)
    processed_pic_dir = out_root / "pic" if args.save_processed_images else None
    if processed_pic_dir is not None:
        processed_pic_dir.mkdir(parents=True, exist_ok=True)

    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=Path(__file__).resolve().parent,
            task_name="eval_image_conflict",
            model_name=args.model_name,
            model=args.model,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="eval_image_conflict",
        data_csv=args.data_csv,
        model=args.model,
        model_name=args.model_name,
        out_dir=str(out_root),
        save_processed_images=bool(args.save_processed_images),
        mask_all=bool(args.mask_all),
        add_noise=bool(args.add_noise),
        add_noise_all=bool(args.add_noise_all),
        mask_scale=float(args.mask_scale),
        noise_scale=float(args.noise_scale),
    )

    preds = tracker.read_records("preds.jsonl")
    seen = int(tracker.state.get("seen", 0))
    skipped = int(tracker.state.get("skipped", 0))

    prompt_cache: Dict[str, str] = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluate NC/IC"):
        if args.limit > 0 and seen >= args.limit:
            break
        row_key = resume_key_for_row(row, idx)
        if tracker.is_done(row_key):
            continue
        seen += 1

        target_labels = parse_json_list(row.get("ic_target_labels"))
        img_path = resolve_row_path(row, args.image_root, key="image_path")
        det_path = resolve_row_path(row, args.image_root, key="detection_path")
        mask_path = resolve_row_path(row, args.image_root, key="mask_path")
        inline_mask = extract_inline_mask_data(row)
        if img_path is None or (not args.mask_all and not args.add_noise_all and (not target_labels or (inline_mask is None and (det_path is None or mask_path is None)))):
            skipped += 1
            tracker.mark_done(row_key, {"status": "skip_missing_assets_or_target"})
            tracker.update(seen=seen, skipped=skipped, kept=len(preds))
            continue

        question = str(row.get("question", "")).strip()
        gold_raw = str(row.get("gold", row.get("answer", ""))).strip()
        gold, wrong, unknown, answer_candidates = get_answer_candidates_from_row(row)
        prompt = prompt_cache.setdefault(question, make_prompt(question))
        noise_seed = stable_seed_from_text(row_key) if (args.add_noise or args.add_noise_all) else None

        nc_image = resize_image_if_needed(Image.open(img_path).convert("RGB"), args.resize_max_side)
        ic_image, ic_meta = build_masked_image(
            source_path=img_path,
            detection_path=det_path,
            target_labels=target_labels,
            mask_path=mask_path,
            mask_rle=inline_mask["mask_rle"] if inline_mask is not None else None,
            mask_height=inline_mask["mask_height"] if inline_mask is not None else None,
            mask_width=inline_mask["mask_width"] if inline_mask is not None else None,
            mask_all=args.mask_all,
            add_noise=args.add_noise,
            add_noise_all=args.add_noise_all,
            noise_std=FULL_IMAGE_NOISE_STD,
            noise_seed=noise_seed,
            mask_scale=args.mask_scale,
            noise_scale=args.noise_scale,
        )
        ic_image = resize_image_if_needed(ic_image, args.resize_max_side)

        if debug_mask_dir is not None and len(preds) < 20:
            debug_name = f"{row.get('img_id', idx)}_{len(preds):04d}.png"
            ic_image.save(debug_mask_dir / debug_name)
        processed_image_path = ""
        if processed_pic_dir is not None:
            processed_name = (
                f"{int(idx):06d}_img_{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(row.get('img_id', idx)))}"
                f"_qid_{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(row.get('qid', idx)))}.png"
            )
            processed_image_path = str(processed_pic_dir / processed_name)
            ic_image.save(processed_image_path)

        nc_messages = build_messages_multimodal(nc_image, prompt)
        ic_messages = build_messages_multimodal(ic_image, prompt)
        nc_scores = score_candidate_options_cached(model, processor, nc_messages, answer_candidates)
        ic_scores = score_candidate_options_cached(model, processor, ic_messages, answer_candidates)
        nc_pred_raw = max(nc_scores, key=lambda key: nc_scores[key])
        ic_pred_raw = max(ic_scores, key=lambda key: ic_scores[key])
        nc_pred_norm = normalize_answer(nc_pred_raw)
        ic_pred_norm = normalize_answer(ic_pred_raw)
        _, nc_pred_label = score_map_to_label(nc_scores, gold, wrong, unknown)
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
            "processed_image_path": processed_image_path,
            "ic_target_labels": target_labels,
            "answer_candidates": answer_candidates,
            "save_processed_images": bool(args.save_processed_images),
            "mask_all": bool(args.mask_all),
            "add_noise": bool(args.add_noise),
            "add_noise_all": bool(args.add_noise_all),
            "mask_scale": float(ic_meta.get("mask_scale", args.mask_scale)),
            "noise_scale": float(ic_meta.get("noise_scale", args.noise_scale)),
            "nc_pred_raw": nc_pred_raw,
            "nc_pred_norm": nc_pred_norm,
            "nc_pred_label": nc_pred_label,
            "nc_scores_json": json.dumps(nc_scores, ensure_ascii=False),
            "ic_pred_raw": ic_pred_raw,
            "ic_pred_norm": ic_pred_norm,
            "ic_pred_label": ic_pred_label,
            "ic_scores_json": json.dumps(ic_scores, ensure_ascii=False),
            "nc_correct": nc_pred_label == "gold",
            "ic_is_unknown": ic_pred_label == "unknown",
            "ic_matches_gold": ic_pred_label == "gold",
            "ic_matches_wrong": ic_pred_label == "wrong",
            "changed_vs_nc": nc_pred_norm != ic_pred_norm,
            "mask_bbox": ic_meta.get("mask_bbox"),
            "used_mask_pixels": bool(ic_meta.get("used_mask_pixels", False)),
            "fallback_rectangles": int(ic_meta.get("fallback_rectangles", 0)),
            "noise_std": float(ic_meta.get("noise_std", 0.0) or 0.0),
            "noise_seed": ic_meta.get("noise_seed"),
        }
        preds.append(record)
        tracker.append_record("preds.jsonl", record)
        tracker.mark_done(row_key, {"status": "ok"})
        tracker.update(seen=seen, skipped=skipped, kept=len(preds))

    summary = summarize_records(preds)
    summary.update(
        {
            "model": args.model,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "data_csv": args.data_csv,
            "out_dir": str(out_root),
            "preds_jsonl": str(out_root / "preds.jsonl"),
            "summary_json": str(out_root / "summary.json"),
            "unknown_target": UNKNOWN_ANSWER,
            "save_processed_images": bool(args.save_processed_images),
            "processed_image_dir": str(processed_pic_dir) if processed_pic_dir is not None else "",
            "mask_all": bool(args.mask_all),
            "add_noise": bool(args.add_noise),
            "add_noise_all": bool(args.add_noise_all),
            "mask_scale": float(args.mask_scale),
            "noise_scale": float(args.noise_scale),
            "noise_std": FULL_IMAGE_NOISE_STD if (args.add_noise or args.add_noise_all) else 0.0,
        }
    )

    preds_path = out_root / "preds.jsonl"
    summary_path = out_root / "summary.json"
    with preds_path.open("w", encoding="utf-8") as f:
        for row in preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.finish(seen=seen, skipped=skipped, kept=len(preds), preds_jsonl=str(preds_path), summary_json=str(summary_path))

    print(f"saved_preds={preds_path}")
    print(f"saved_summary={summary_path}")
    print(f"n_records={summary['n_records']}")
    print(f"nc_correct_rate={summary['nc_correct_rate']:.4f}")
    print(f"ic_unknown_rate={summary['ic_unknown_rate']:.4f}")
    print(f"ic_gold_rate={summary['ic_gold_rate']:.4f}")
    print(f"ic_wrong_rate={summary['ic_wrong_rate']:.4f}")
    print(f"ic_other_rate={summary['ic_other_rate']:.4f}")


if __name__ == "__main__":
    main()






