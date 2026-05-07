#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path("/root/logit_lens")
IMAGE_CONFLICT_DIR = REPO_ROOT / "Slake_vqa" / "image_conflict"
if str(IMAGE_CONFLICT_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_CONFLICT_DIR))

import ablate_head as ablate_mod
import eval_slake_mm_image_conflict as noise_eval_mod
from mask_utils import build_masked_image


def resume_key_for_noise_sample(
    row_idx: int,
    question: str,
    gold: str,
    wrong: str,
    noise_mode: str,
    mask_scope: str,
) -> str:
    return f"{row_idx}\t{question}\t{gold}\t{wrong}\t{noise_mode}\t{mask_scope}"


def rebuild_sample_images(
    sample: Dict[str, Any],
    max_image_side: int,
    noise_mode: str,
    mask_scale: float,
    noise_scale: float,
    noise_std: float,
):
    nc_image = ablate_mod.resize_image_max_side(
        Image.open(sample["img_path"]).convert("RGB"),
        max_image_side,
    )
    noise_image, noise_meta = build_masked_image(
        source_path=sample["img_path"],
        detection_path=sample["detection_path"],
        target_labels=sample["ic_target_labels"],
        mask_path=sample["mask_path"] or None,
        mask_rle=sample["mask_rle_obj"],
        mask_height=sample["mask_h_obj"],
        mask_width=sample["mask_w_obj"],
        add_noise=(noise_mode == "local"),
        add_noise_all=(noise_mode == "full"),
        noise_std=noise_std,
        noise_seed=sample["noise_seed"],
        mask_scale=mask_scale,
        noise_scale=noise_scale,
    )
    noise_image = ablate_mod.resize_image_max_side(noise_image, max_image_side)
    return nc_image, noise_image, noise_meta


def save_noise_image(
    out_dir: Path,
    sample: Dict[str, Any],
    noise_image: Image.Image,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"{int(sample['row_idx']):06d}"
        f"_img_{str(sample.get('img_id', sample['row_idx'])).replace('/', '_')}"
        f"_qid_{str(sample.get('qid', sample['row_idx'])).replace('/', '_')}.png"
    )
    out_path = out_dir / file_name
    noise_image.save(out_path)
    return str(out_path)


def build_samples(args, df, model, processor):
    q_col = ablate_mod.infer_col(df, ["question", "query"])
    gold_col = ablate_mod.infer_col(df, ["answer", "gold", "gt_answer", "label"])

    samples: List[Dict[str, Any]] = []
    skipped_oom = 0

    iterable = df.iterrows()
    total = len(df)
    if args.max_examples > 0:
        df = df.head(args.max_examples)
        iterable = df.iterrows()
        total = len(df)

    for idx, row in tqdm(iterable, total=total, desc="Build noise samples"):
        question = ablate_mod.normalize_text(row[q_col])
        gold, wrong, unknown, _answer_candidates = ablate_mod.get_answer_candidates_from_row(row)
        if not gold:
            gold = ablate_mod.normalize_text(row[gold_col]).lower()

        if not question or not gold or not unknown or gold == unknown:
            continue

        target_labels = noise_eval_mod.parse_json_list(row.get("ic_target_labels"))
        img_path = noise_eval_mod.resolve_row_path(row, args.image_root, key="image_path")
        det_path = noise_eval_mod.resolve_row_path(row, args.image_root, key="detection_path")
        mask_path = noise_eval_mod.resolve_row_path(row, args.image_root, key="mask_path")
        inline_mask = noise_eval_mod.extract_inline_mask_data(row)
        if img_path is None:
            continue
        if args.noise_mode == "local" and (
            not target_labels or (inline_mask is None and (det_path is None or mask_path is None))
        ):
            continue

        mask_rle_obj = None
        mask_h_obj = None
        mask_w_obj = None
        if inline_mask is not None:
            mask_rle_obj = inline_mask.get("mask_rle")
            mask_h_obj = inline_mask.get("mask_height")
            mask_w_obj = inline_mask.get("mask_width")

        nc_prompt, noise_prompt = ablate_mod.make_prompts(question=question)
        noise_seed = noise_eval_mod.stable_seed_from_text(
            noise_eval_mod.resume_key_for_row(row, idx)
        )

        try:
            nc_image, noise_image, noise_meta = rebuild_sample_images(
                {
                    "img_path": str(img_path),
                    "detection_path": str(det_path) if det_path is not None else "",
                    "mask_path": str(mask_path) if mask_path is not None else "",
                    "ic_target_labels": target_labels,
                    "mask_rle_obj": mask_rle_obj,
                    "mask_h_obj": mask_h_obj,
                    "mask_w_obj": mask_w_obj,
                    "noise_seed": noise_seed,
                },
                max_image_side=args.max_image_side,
                noise_mode=args.noise_mode,
                mask_scale=args.mask_scale,
                noise_scale=args.noise_scale,
                noise_std=args.noise_std,
            )
            nc_scores = ablate_mod.score_answer_candidates_eval_style(
                model, processor, nc_image, nc_prompt, gold, wrong, unknown, args.trace_mode
            )
            noise_scores = ablate_mod.score_answer_candidates_eval_style(
                model, processor, noise_image, noise_prompt, gold, wrong, unknown, args.trace_mode
            )
            nc_metrics = ablate_mod.compute_scalar_metrics(nc_scores, args.trace_mode)
            noise_metrics = ablate_mod.compute_scalar_metrics(noise_scores, args.trace_mode)
        except Exception as exc:
            if not ablate_mod.is_cuda_oom_error(exc):
                continue
            skipped_oom += 1
            ablate_mod.maybe_empty_cuda_cache(len(samples) + skipped_oom, every=1)
            continue
        finally:
            ablate_mod.maybe_empty_cuda_cache(len(samples) + 1)

        samples.append(
            {
                "row_idx": int(idx),
                "img_id": row.get("img_id", ""),
                "qid": row.get("qid", ""),
                "img_path": str(img_path),
                "question": question,
                "gold": gold,
                "wrong": wrong,
                "unknown": unknown,
                "nc_prompt": nc_prompt,
                "noise_prompt": noise_prompt,
                "detection_path": str(det_path) if det_path is not None else "",
                "mask_path": str(mask_path) if mask_path is not None else "",
                "ic_target_labels": target_labels,
                "mask_rle_obj": mask_rle_obj,
                "mask_h_obj": mask_h_obj,
                "mask_w_obj": mask_w_obj,
                "base_nc_scores": nc_scores,
                "base_noise_scores": noise_scores,
                "base_nc_metrics": nc_metrics,
                "base_noise_metrics": noise_metrics,
                "base_nc_pred": ablate_mod.pred_label(nc_scores),
                "base_noise_pred": ablate_mod.pred_label(noise_scores),
                "base_nc_gold_margin": ablate_mod.nc_gold_margin(nc_scores),
                "noise_seed": noise_seed,
                "base_noise_meta": noise_meta,
            }
        )

    if skipped_oom > 0:
        print(f"[WARN] Skipped {skipped_oom} samples due to CUDA OOM during baseline cache build.")

    return samples


def summarize_metric_changes(records: List[Dict[str, Any]], metrics: List[str]):
    out = {}
    for metric_name in metrics:
        eff_fixed_nc = [r["metric_effect_reduction_fixed_nc"][metric_name] for r in records]
        eff_scope = [r["metric_effect_reduction_mask_scope"][metric_name] for r in records]
        base_ch = [r["metric_abs_base_change"][metric_name] for r in records]
        noise_ch = [r["metric_abs_noise_change"][metric_name] for r in records]
        out[metric_name] = {
            "mean_abs_effect_reduction_fixed_nc": ablate_mod.safe_mean(eff_fixed_nc),
            "mean_abs_effect_reduction_mask_scope": ablate_mod.safe_mean(eff_scope),
            "mean_abs_base_change": ablate_mod.safe_mean(base_ch),
            "mean_abs_noise_change": ablate_mod.safe_mean(noise_ch),
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ablate selected heads on noisy SLAKE image-conflict inputs."
    )
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--selected_heads", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--max_image_side", type=int, default=672)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["conflict"])
    ap.add_argument("--metrics", type=str, default="follow_context")
    ap.add_argument("--mask_scope", type=str, default="ctx_only", choices=["all", "ctx_only"])
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])
    ap.add_argument("--noise_mode", type=str, default="full", choices=["full", "local"])
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--noise_std", type=float, default=48.0)
    ap.add_argument("--random_ablate", action="store_true")
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_noise_images", action="store_true")
    ap.add_argument("--noise_image_dir", type=str, default="")
    args = ap.parse_args()

    args.model, _model_spec = noise_eval_mod.resolve_model_selection(
        model_name=args.model_name,
        model=args.model,
    )
    args.out_json = noise_eval_mod.prefix_model_relative_path(
        args.out_json,
        model_name=args.model_name,
        model=args.model,
    )
    if args.mask_scale <= 0 or args.noise_scale <= 0 or args.noise_std <= 0:
        raise ValueError("--mask_scale, --noise_scale, and --noise_std must be positive.")

    ablate_mod.set_seed(args.seed)
    device = (
        "cuda"
        if (args.device == "auto" and ablate_mod.torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    dtype = ablate_mod.str2dtype(args.dtype)

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    for metric_name in metrics:
        if metric_name not in ablate_mod.ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric: {metric_name}")

    df = pd.read_csv(args.data_csv)
    processor = ablate_mod.load_processor_with_compat(args.model)
    model = noise_eval_mod.load_mm_model(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()

    samples = build_samples(args, df, model, processor)
    if not samples:
        raise RuntimeError("No valid noise samples were built.")

    data_tag = Path(args.data_csv).stem
    tracker = noise_eval_mod.ResumeTracker(
        resume_dir=noise_eval_mod.build_resume_dir(
            project_root=Path(__file__).resolve().parent,
            task_name=(
                f"ablate_noise_random_{data_tag}_{args.noise_mode}"
                if args.random_ablate
                else f"ablate_noise_{data_tag}_{args.noise_mode}"
            ),
            model_name=args.model_name,
            model=args.model,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="ablate_noise_random" if args.random_ablate else "ablate_noise",
        data_csv=args.data_csv,
        out_json=args.out_json,
        mask_scope=args.mask_scope,
        noise_mode=args.noise_mode,
        noise_scale=args.noise_scale,
        noise_std=args.noise_std,
        save_noise_images=bool(args.save_noise_images),
        noise_image_dir=args.noise_image_dir,
    )

    noise_image_dir = None
    if args.save_noise_images:
        if args.noise_image_dir.strip():
            noise_image_dir = Path(args.noise_image_dir)
        else:
            noise_image_dir = Path(args.out_json).with_suffix("")
            noise_image_dir = noise_image_dir.parent / f"{noise_image_dir.name}_images"

    selected_pairs = ablate_mod.parse_selected_heads(args.selected_heads)
    if args.random_ablate:
        selected_pairs = ablate_mod.random_select_heads(
            model=model,
            k=len(selected_pairs),
            seed=args.random_seed,
        )
    layer_to_heads = ablate_mod.pack_layer_to_heads(selected_pairs)
    handles = ablate_mod.install_head_mask_hooks(model, layer_to_heads, keep_mode=args.keep_mode)

    records = tracker.read_records("records.jsonl")
    skipped_rebuild = 0
    first_rebuild_error = ""

    try:
        for sample in tqdm(samples, desc="Ablate selected heads on noise"):
            sample_key = resume_key_for_noise_sample(
                row_idx=sample["row_idx"],
                question=sample["question"],
                gold=sample["gold"],
                wrong=sample["wrong"],
                noise_mode=args.noise_mode,
                mask_scope=args.mask_scope,
            )
            if tracker.is_done(sample_key):
                continue

            try:
                nc_image, noise_image, noise_meta = rebuild_sample_images(
                    sample,
                    max_image_side=args.max_image_side,
                    noise_mode=args.noise_mode,
                    mask_scale=args.mask_scale,
                    noise_scale=args.noise_scale,
                    noise_std=args.noise_std,
                )
            except Exception as exc:
                skipped_rebuild += 1
                if not first_rebuild_error:
                    first_rebuild_error = repr(exc)
                continue

            saved_noise_image_path = ""
            if noise_image_dir is not None:
                saved_noise_image_path = save_noise_image(
                    noise_image_dir,
                    sample,
                    noise_image,
                )

            if args.mask_scope == "all":
                ab_nc_scores = ablate_mod.score_answer_candidates_eval_style(
                    model,
                    processor,
                    nc_image,
                    sample["nc_prompt"],
                    sample["gold"],
                    sample["wrong"],
                    sample["unknown"],
                    args.trace_mode,
                )
                ab_nc_metrics = ablate_mod.compute_scalar_metrics(ab_nc_scores, args.trace_mode)
                ab_nc_pred = ablate_mod.pred_label(ab_nc_scores)
            else:
                ab_nc_scores = copy.deepcopy(sample["base_nc_scores"])
                ab_nc_metrics = copy.deepcopy(sample["base_nc_metrics"])
                ab_nc_pred = sample["base_nc_pred"]

            ab_nc_gold_margin = ablate_mod.nc_gold_margin(ab_nc_scores)
            ab_noise_scores = ablate_mod.score_answer_candidates_eval_style(
                model,
                processor,
                noise_image,
                sample["noise_prompt"],
                sample["gold"],
                sample["wrong"],
                sample["unknown"],
                args.trace_mode,
            )
            ab_noise_metrics = ablate_mod.compute_scalar_metrics(ab_noise_scores, args.trace_mode)
            ab_noise_pred = ablate_mod.pred_label(ab_noise_scores)

            noise_follow_context_gain = float(
                ab_noise_metrics["follow_context"] - sample["base_noise_metrics"]["follow_context"]
            )
            nc_gold_margin_damage = float(
                max(0.0, sample["base_nc_gold_margin"] - ab_nc_gold_margin)
            )
            hallucination_relief = float(noise_follow_context_gain - nc_gold_margin_damage)

            metric_effect_reduction_fixed_nc = {}
            metric_effect_reduction_mask_scope = {}
            metric_abs_base_change = {}
            metric_abs_noise_change = {}
            for metric_name in metrics:
                metric_effect_reduction_fixed_nc[metric_name] = float(
                    ab_noise_metrics[metric_name] - sample["base_noise_metrics"][metric_name]
                )
                metric_effect_reduction_mask_scope[metric_name] = float(
                    ab_noise_metrics[metric_name] - sample["base_noise_metrics"][metric_name]
                )
                metric_abs_base_change[metric_name] = float(
                    abs(ab_nc_metrics[metric_name] - sample["base_nc_metrics"][metric_name])
                )
                metric_abs_noise_change[metric_name] = float(
                    abs(ab_noise_metrics[metric_name] - sample["base_noise_metrics"][metric_name])
                )

            rec = {
                "row_idx": sample["row_idx"],
                "img_id": sample["img_id"],
                "qid": sample["qid"],
                "gold": sample["gold"],
                "wrong": sample["wrong"],
                "base_nc_pred": sample["base_nc_pred"],
                "base_noise_pred": sample["base_noise_pred"],
                "base_ctx_pred": sample["base_noise_pred"],
                "ab_nc_pred": ab_nc_pred,
                "ab_noise_pred": ab_noise_pred,
                "ab_ctx_pred": ab_noise_pred,
                "base_nc_scores": sample["base_nc_scores"],
                "base_noise_scores": sample["base_noise_scores"],
                "base_ctx_scores": sample["base_noise_scores"],
                "ab_nc_scores": ab_nc_scores,
                "ab_noise_scores": ab_noise_scores,
                "ab_ctx_scores": ab_noise_scores,
                "base_nc_metrics": sample["base_nc_metrics"],
                "base_noise_metrics": sample["base_noise_metrics"],
                "base_ctx_metrics": sample["base_noise_metrics"],
                "ab_nc_metrics": ab_nc_metrics,
                "ab_noise_metrics": ab_noise_metrics,
                "ab_ctx_metrics": ab_noise_metrics,
                "base_nc_gold_margin": sample["base_nc_gold_margin"],
                "ab_nc_gold_margin": ab_nc_gold_margin,
                "noise_follow_context_gain": noise_follow_context_gain,
                "ctx_follow_context_gain": noise_follow_context_gain,
                "nc_gold_margin_damage": nc_gold_margin_damage,
                "hallucination_relief": hallucination_relief,
                "noise_seed": sample["noise_seed"],
                "base_noise_meta": sample["base_noise_meta"],
                "ab_noise_meta": noise_meta,
                "saved_noise_image_path": saved_noise_image_path,
                "metric_effect_reduction_fixed_nc": metric_effect_reduction_fixed_nc,
                "metric_effect_reduction_mask_scope": metric_effect_reduction_mask_scope,
                "metric_abs_base_change": metric_abs_base_change,
                "metric_abs_noise_change": metric_abs_noise_change,
                "metric_abs_ctx_change": metric_abs_noise_change,
            }
            records.append(rec)
            tracker.append_record("records.jsonl", rec)
            tracker.mark_done(sample_key, {"row_idx": int(sample["row_idx"])})
            tracker.update(completed=len(records))
            ablate_mod.maybe_empty_cuda_cache(len(records))
    finally:
        ablate_mod.remove_handles(handles)

    all_relief = [r["hallucination_relief"] for r in records]
    all_noise_gain = [r["noise_follow_context_gain"] for r in records]
    all_nc_damage = [r["nc_gold_margin_damage"] for r in records]

    base_nc_correct = 0
    base_noise_correct = 0
    ab_nc_correct = 0
    ab_noise_correct = 0
    base_noise_unknown = 0
    ab_noise_unknown = 0
    for record in records:
        if record["base_nc_pred"] == "gold":
            base_nc_correct += 1
        if record["base_noise_pred"] == "gold":
            base_noise_correct += 1
        if record["ab_nc_pred"] == "gold":
            ab_nc_correct += 1
        if record["ab_noise_pred"] == "gold":
            ab_noise_correct += 1
        if record["base_noise_pred"] == "unknown":
            base_noise_unknown += 1
        if record["ab_noise_pred"] == "unknown":
            ab_noise_unknown += 1

    n_records = len(records)
    metric_summary = summarize_metric_changes(records, metrics)
    summary = {
        "config": {
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "model": args.model,
            "selected_heads": args.selected_heads,
            "trace_mode": args.trace_mode,
            "metrics": metrics,
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "dtype": args.dtype,
            "device": device,
            "max_examples": args.max_examples,
            "max_image_side": args.max_image_side,
            "random_ablate": bool(args.random_ablate),
            "random_seed": args.random_seed,
            "noise_mode": args.noise_mode,
            "mask_scale": float(args.mask_scale),
            "noise_scale": float(args.noise_scale),
            "noise_std": float(args.noise_std),
            "save_noise_images": bool(args.save_noise_images),
            "noise_image_dir": str(noise_image_dir) if noise_image_dir is not None else "",
        },
        "selected_pairs": [{"layer": layer, "head": head} for layer, head in selected_pairs],
        "n_selected_heads": len(selected_pairs),
        "n_samples": n_records,
        "noise_mode": args.noise_mode,
        "mean_hallucination_relief": ablate_mod.safe_mean(all_relief),
        "mean_noise_follow_context_gain": ablate_mod.safe_mean(all_noise_gain),
        "mean_ic_follow_context_gain": ablate_mod.safe_mean(all_noise_gain),
        "mean_nc_gold_margin_damage": ablate_mod.safe_mean(all_nc_damage),
        "pred_nc": {
            "base_acc": float(base_nc_correct / n_records) if n_records > 0 else 0.0,
            "ab_acc": float(ab_nc_correct / n_records) if n_records > 0 else 0.0,
            "acc_delta": float((ab_nc_correct - base_nc_correct) / n_records) if n_records > 0 else 0.0,
        },
        "pred_noise": {
            "base_acc": float(base_noise_correct / n_records) if n_records > 0 else 0.0,
            "ab_acc": float(ab_noise_correct / n_records) if n_records > 0 else 0.0,
            "acc_delta": float((ab_noise_correct - base_noise_correct) / n_records) if n_records > 0 else 0.0,
        },
        "pred_ctx": {
            "base_acc": float(base_noise_correct / n_records) if n_records > 0 else 0.0,
            "ab_acc": float(ab_noise_correct / n_records) if n_records > 0 else 0.0,
            "acc_delta": float((ab_noise_correct - base_noise_correct) / n_records) if n_records > 0 else 0.0,
        },
        "noise_unknown": {
            "base_rate": float(base_noise_unknown / n_records) if n_records > 0 else 0.0,
            "ab_rate": float(ab_noise_unknown / n_records) if n_records > 0 else 0.0,
            "rate_delta": float((ab_noise_unknown - base_noise_unknown) / n_records) if n_records > 0 else 0.0,
        },
        "ctx_unknown": {
            "base_rate": float(base_noise_unknown / n_records) if n_records > 0 else 0.0,
            "ab_rate": float(ab_noise_unknown / n_records) if n_records > 0 else 0.0,
            "rate_delta": float((ab_noise_unknown - base_noise_unknown) / n_records) if n_records > 0 else 0.0,
        },
        "per_metric": metric_summary,
        "records": records,
    }

    ablate_mod.ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    tracker.finish(out_json=args.out_json, n_samples=n_records, n_selected_heads=len(selected_pairs))

    if skipped_rebuild > 0:
        print(f"[WARN] Skipped {skipped_rebuild} samples during noise image rebuild before ablation.")
        if first_rebuild_error:
            print(f"[WARN] First rebuild error: {first_rebuild_error}")

    print(
        json.dumps(
            {
                "saved_to": args.out_json,
                "n_selected_heads": len(selected_pairs),
                "n_samples": n_records,
                "noise_mode": args.noise_mode,
                "mean_hallucination_relief": summary["mean_hallucination_relief"],
                "mean_noise_follow_context_gain": summary["mean_noise_follow_context_gain"],
                "mean_nc_gold_margin_damage": summary["mean_nc_gold_margin_damage"],
                "noise_unknown_rate_delta": summary["noise_unknown"]["rate_delta"],
                "pred_noise_acc_delta": summary["pred_noise"]["acc_delta"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
