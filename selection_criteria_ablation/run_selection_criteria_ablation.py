#!/usr/bin/env python3
"""Run selection-criteria ablations with the existing project evaluators.

This is intentionally a thin orchestration layer:
  1. Read the existing head-scan outputs.
  2. Select CER-only and BCP-only heads with the same count as the current
     Dual selection, up to exact ties/availability.
  3. Feed those selected-head files into the original ablation/eval scripts.

Default behavior is dry-run: write selected heads and runnable commands.
Pass --run to launch the model evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ABLATION_ROOT = Path("/root/logit_lens/selection_criteria_ablation")
DEFAULT_OUTPUT_ROOT = ABLATION_ROOT / "results"
DEFAULT_POSITION = "before_question"
CRITERIA = ("cer-only", "bcp-only")


@dataclass(frozen=True)
class DatasetSplit:
    split: str
    path: Path


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    task: str
    project_root: Path
    model_path: str
    scan_path: Path
    dual_head_file: Path
    dual_result_files: dict[str, Path]
    splits: tuple[DatasetSplit, ...]
    eval_script: str
    dtype: str
    model_name: str = ""
    image_root: str = "."
    trace_mode: str = "conflict"
    metrics: str = "follow_conflict"
    mask_scope: str = "all"
    keep_mode: str = "self"


CONFLICT_ROOT = Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp")
HULU_ROOT = Path("/root/logit_lens/VQA_RAD/Hulu-med")
INTERNVL_ROOT = Path("/root/logit_lens/VQA_RAD/text_conflict")


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen3-4b": ModelConfig(
        key="qwen3-4b",
        display_name="Qwen3-4B",
        task="conflictmedqa",
        project_root=CONFLICT_ROOT,
        model_path="/root/autodl-tmp/qwen3-4B",
        scan_path=CONFLICT_ROOT
        / "result_train/before_question/headscan_rounds_top50_inf/head_scan_summary.json",
        dual_head_file=CONFLICT_ROOT
        / "result_train/before_question/headscan_rounds_top50_inf/head_groups.json",
        dual_result_files={
            "train": CONFLICT_ROOT / "result_train/before_question/summary_train.json",
            "val": CONFLICT_ROOT / "result_train/before_question/summary_val.json",
            "all": CONFLICT_ROOT / "result_train/before_question/summary_all.json",
        },
        splits=(
            DatasetSplit("train", CONFLICT_ROOT / "data/kept_pairs_a12_b10_all_train.jsonl"),
            DatasetSplit("val", CONFLICT_ROOT / "data/kept_pairs_a12_b10_all_val.jsonl"),
        ),
        eval_script="ablate_head_inf.py",
        dtype="bfloat16",
    ),
    "llama32-3b": ModelConfig(
        key="llama32-3b",
        display_name="Llama-3.2-3B-Instruct",
        task="conflictmedqa",
        project_root=CONFLICT_ROOT,
        model_path="/root/autodl-tmp/Llama-3.2-3B-Instruct",
        scan_path=CONFLICT_ROOT
        / "llama32_3b/result/before_question/headscan_rounds_top30_inf/head_scan_summary.json",
        dual_head_file=CONFLICT_ROOT
        / "llama32_3b/result/before_question/headscan_rounds_top30_inf/selected_heads.json",
        dual_result_files={
            "train": CONFLICT_ROOT
            / "llama32_3b/result/before_question/conflict_retest_ablated_heads_inf_before_question_train_summary.json",
            "val": CONFLICT_ROOT
            / "llama32_3b/result/before_question/conflict_retest_ablated_heads_inf_before_question_val_summary.json",
        },
        splits=(
            DatasetSplit(
                "train",
                CONFLICT_ROOT / "llama32_3b/data/kept_pairs_a12_b10_all_train.jsonl",
            ),
            DatasetSplit(
                "val",
                CONFLICT_ROOT / "llama32_3b/data/kept_pairs_a12_b10_all_val.jsonl",
            ),
        ),
        eval_script="ablate_head_inf.py",
        dtype="bfloat16",
    ),
    "hulumed4b": ModelConfig(
        key="hulumed4b",
        display_name="Hulu-med-4B",
        task="vqa_rad",
        project_root=HULU_ROOT,
        model_path="/root/autodl-tmp/Hulu-Med-4B",
        scan_path=HULU_ROOT
        / "result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json",
        dual_head_file=HULU_ROOT
        / "result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json",
        dual_result_files={
            "train": HULU_ROOT
            / "result_train_hulumed4b_before_question/ablate_selected_heads_all_hulumed4b_train.json",
            "val": HULU_ROOT
            / "result_train_hulumed4b_before_question/ablate_selected_heads_all_hulumed4b_val.json",
        },
        splits=(
            DatasetSplit("train", HULU_ROOT / "data/nc_cc_both_correct_rerun_tmp_train.csv"),
            DatasetSplit("val", HULU_ROOT / "data/nc_cc_both_correct_rerun_tmp_val.csv"),
        ),
        eval_script="ablate_head.py",
        dtype="bf16",
    ),
    "internvl35-4b": ModelConfig(
        key="internvl35-4b",
        display_name="InternVL3.5-4B",
        task="vqa_rad",
        project_root=INTERNVL_ROOT,
        model_path="/root/autodl-tmp/InternVL3_5-4B",
        model_name="InternVL3_5-4B",
        scan_path=INTERNVL_ROOT
        / "internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json",
        dual_head_file=INTERNVL_ROOT
        / "internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json",
        dual_result_files={
            "train": INTERNVL_ROOT
            / "internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_train.json",
            "val": INTERNVL_ROOT
            / "internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_val.json",
        },
        splits=(
            DatasetSplit(
                "train",
                INTERNVL_ROOT / "data/internvl35_4b/vqa_rad_nc_cc_both_correct_train.csv",
            ),
            DatasetSplit(
                "val",
                INTERNVL_ROOT / "data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv",
            ),
        ),
        eval_script="ablate_head.py",
        dtype="bf16",
    ),
}

MODEL_ALIASES = {
    "qwen": "qwen3-4b",
    "qwen3_4b": "qwen3-4b",
    "qwen3-4b": "qwen3-4b",
    "llama": "llama32-3b",
    "llama32": "llama32-3b",
    "llama32_3b": "llama32-3b",
    "llama32-3b": "llama32-3b",
    "hulu": "hulumed4b",
    "hulumed": "hulumed4b",
    "hulumed4b": "hulumed4b",
    "hulu-med-4b": "hulumed4b",
    "internvl": "internvl35-4b",
    "internvl35": "internvl35-4b",
    "internvl35_4b": "internvl35-4b",
    "internvl35-4b": "internvl35-4b",
    "internvl3.5-4b": "internvl35-4b",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_head_record(item: dict[str, Any], source: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if "layer" not in item or "head" not in item:
        return None

    cer = maybe_float(item.get("mean_abs_effect_reduction"))
    bcp = maybe_float(item.get("mean_abs_base_delta_change"))
    bcp_key = "mean_abs_base_delta_change"
    if bcp is None:
        bcp = maybe_float(item.get("mean_abs_base_change"))
        bcp_key = "mean_abs_base_change"

    if cer is None or bcp is None:
        return None

    row = dict(item)
    row.update(
        {
            "layer": int(item["layer"]),
            "head": int(item["head"]),
            "cer": cer,
            "bcp": bcp,
            "cer_key": "mean_abs_effect_reduction",
            "bcp_key": bcp_key,
            "source": source,
        }
    )
    row.setdefault("mean_abs_effect_reduction", cer)
    row.setdefault("mean_abs_base_delta_change", bcp)
    return row


def iter_candidate_items(obj: Any, source: str = "root") -> Iterable[tuple[dict[str, Any], str]]:
    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            if isinstance(item, dict):
                yield item, f"{source}[{idx}]"
                yield from iter_candidate_items(item, f"{source}[{idx}]")
        return

    if not isinstance(obj, dict):
        return

    if "layer" in obj and "head" in obj:
        yield obj, source

    candidate_keys = (
        "results",
        "top20",
        "top",
        "heads",
        "selected",
        "selected_heads",
        "conflict_specific",
        "backbone",
        "round_outputs",
        "per_round_summaries",
        "rounds",
        "result",
        "coarse_top20",
        "refined_top20",
        "rerank_top20",
    )
    for key in candidate_keys:
        if key in obj:
            yield from iter_candidate_items(obj[key], f"{source}.{key}")


def load_scan_heads(scan_path: Path) -> list[dict[str, Any]]:
    obj = read_json(scan_path)
    best: dict[tuple[int, int], dict[str, Any]] = {}

    for item, source in iter_candidate_items(obj):
        row = normalize_head_record(item, source)
        if row is None:
            continue
        key = (row["layer"], row["head"])
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        if row["cer"] > old["cer"] or (row["cer"] == old["cer"] and row["bcp"] < old["bcp"]):
            best[key] = row

    heads = list(best.values())
    if not heads:
        raise RuntimeError(f"No usable head records found in {scan_path}")
    return heads


def load_dual_heads(path: Path) -> list[dict[str, int]]:
    obj = read_json(path)
    items = None
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for key in ("selected_heads", "selected", "conflict_specific", "results", "heads", "topk"):
            value = obj.get(key)
            if isinstance(value, list):
                items = value
                break
    if items is None:
        raise RuntimeError(f"Cannot parse Dual selected heads from {path}")

    pairs = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if "layer" not in item or "head" not in item:
            continue
        pair = (int(item["layer"]), int(item["head"]))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append({"layer": pair[0], "head": pair[1]})
    if not pairs:
        raise RuntimeError(f"No valid Dual heads found in {path}")
    return pairs


def select_heads(
    heads: list[dict[str, Any]],
    criterion: str,
    target_count: int,
    tolerance: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if criterion == "cer-only":
        ordered = sorted(heads, key=lambda h: (-h["cer"], h["bcp"], h["layer"], h["head"]))
        metric_key = "cer"
        direction = "descending"
    elif criterion == "bcp-only":
        ordered = sorted(heads, key=lambda h: (h["bcp"], -h["cer"], h["layer"], h["head"]))
        metric_key = "bcp"
        direction = "ascending"
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    selected = ordered[: min(target_count, len(ordered))]
    threshold = selected[-1][metric_key] if selected else None
    count_diff = abs(len(selected) - target_count)
    if count_diff > tolerance:
        raise RuntimeError(
            f"{criterion} selected {len(selected)} heads, target {target_count}, "
            f"tolerance {tolerance}; not enough candidate heads?"
        )

    meta = {
        "criterion": criterion,
        "selection_rule": (
            "sort by mean_abs_effect_reduction descending"
            if criterion == "cer-only"
            else "sort by BCP ascending"
        ),
        "metric_key": metric_key,
        "direction": direction,
        "threshold_at_last_selected": threshold,
        "target_count_from_dual": target_count,
        "selected_count": len(selected),
        "count_diff": count_diff,
        "tolerance": tolerance,
        "n_candidates": len(heads),
    }
    return selected, meta


def write_selected_heads(
    out_dir: Path,
    model: ModelConfig,
    criterion: str,
    selected: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": criterion,
        "criterion": criterion,
        "model": model.display_name,
        "task": model.task,
        "selected_heads": selected,
        "selected": selected,
        "n_selected": len(selected),
        "meta": meta,
    }
    json_path = out_dir / f"{criterion}_selected_heads.json"
    csv_path = out_dir / f"{criterion}_selected_heads.csv"
    write_json(json_path, payload)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "layer",
                "head",
                "CER_mean_abs_effect_reduction",
                "BCP",
                "BCP_source_key",
                "source",
            ]
        )
        for rank, head in enumerate(selected, 1):
            writer.writerow(
                [
                    rank,
                    head["layer"],
                    head["head"],
                    f"{head['cer']:.10g}",
                    f"{head['bcp']:.10g}",
                    head.get("bcp_key", ""),
                    head.get("source", ""),
                ]
            )
    return json_path


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def model_path_for(model: ModelConfig) -> str:
    env_keys = {
        "qwen3-4b": "QWEN3_4B_MODEL_PATH",
        "llama32-3b": "LLAMA32_3B_MODEL_PATH",
        "hulumed4b": "HULUMED4B_MODEL_PATH",
        "internvl35-4b": "INTERNVL35_4B_MODEL_PATH",
    }
    env_key = env_keys.get(model.key)
    if env_key:
        override = os.environ.get(env_key, "").strip()
        if override:
            return override
    return model.model_path


def build_conflict_command(
    model: ModelConfig,
    split: DatasetSplit,
    head_file: Path,
    out_dir: Path,
    criterion: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path]:
    result_dir = out_dir / criterion
    result_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = result_dir / f"{criterion}_{split.split}.jsonl"
    summary_json = result_dir / f"{criterion}_{split.split}_summary.json"
    cmd = [
        args.python,
        model.eval_script,
        "--pairs",
        str(split.path),
        "--model",
        model_path_for(model),
        "--head_file",
        str(head_file),
        "--out",
        str(out_jsonl),
        "--summary_out",
        str(summary_json),
        "--positions",
        args.position,
        "--mask_scope",
        args.mask_scope,
        "--dtype",
        args.conflict_dtype or model.dtype,
        "--device_map",
        args.device_map,
    ]
    if args.max_pairs > 0:
        cmd.extend(["--max_pairs", str(args.max_pairs)])
    if args.enable_thinking:
        cmd.append("--enable_thinking")
    return cmd, out_jsonl, summary_json


def build_vqa_command(
    model: ModelConfig,
    split: DatasetSplit,
    head_file: Path,
    out_dir: Path,
    criterion: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path]:
    result_dir = out_dir / criterion
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"{criterion}_{split.split}.json"
    cmd = [
        args.python,
        model.eval_script,
        "--data_csv",
        str(split.path),
        "--image_root",
        model.image_root,
        "--model",
        model_path_for(model),
        "--selected_heads",
        str(head_file),
        "--out_json",
        str(out_json),
        "--device",
        args.device,
        "--dtype",
        args.vqa_dtype or model.dtype,
        "--trace_mode",
        model.trace_mode,
        "--position",
        args.position,
        "--metrics",
        model.metrics,
        "--mask_scope",
        args.mask_scope,
        "--keep_mode",
        model.keep_mode,
        "--seed",
        str(args.seed),
        "--max_image_side",
        str(args.max_image_side),
    ]
    if model.model_name:
        cmd.extend(["--model_name", model.model_name])
    if args.max_examples > 0:
        cmd.extend(["--max_examples", str(args.max_examples)])
    if args.resume and model.key == "internvl35-4b":
        cmd.append("--resume")
    return cmd, out_json, out_json


def build_command(
    model: ModelConfig,
    split: DatasetSplit,
    head_file: Path,
    out_dir: Path,
    criterion: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path, Path]:
    if model.task == "conflictmedqa":
        return build_conflict_command(model, split, head_file, out_dir, criterion, args)
    if model.task == "vqa_rad":
        return build_vqa_command(model, split, head_file, out_dir, criterion, args)
    raise ValueError(model.task)


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    skip_existing: bool,
    expected_result: Path,
) -> dict[str, Any]:
    if skip_existing and expected_result.exists():
        print(f"[skip] existing result: {expected_result}", flush=True)
        return {
            "status": "skipped_existing",
            "returncode": 0,
            "result": str(expected_result),
            "log": str(log_path),
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[run] {quote_cmd(cmd)}", flush=True)
    print(f"[log] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ cd {cwd}\n")
        log.write(f"$ {quote_cmd(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode == 0:
        print(f"[done] {expected_result}", flush=True)
    else:
        print(f"[failed:{proc.returncode}] see log: {log_path}", flush=True)
    return {
        "status": "finished" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "result": str(expected_result),
        "log": str(log_path),
    }


def copy_dual_metadata(model: ModelConfig, out_dir: Path) -> dict[str, Any]:
    dual_heads = load_dual_heads(model.dual_head_file)
    payload = {
        "criterion": "dual",
        "note": "Dual is not re-evaluated by this runner; these paths point to the existing results.",
        "head_file": str(model.dual_head_file),
        "n_selected_heads": len(dual_heads),
        "selected_heads": dual_heads,
        "result_files": {k: str(v) for k, v in model.dual_result_files.items()},
        "existing_results_found": {k: v.exists() for k, v in model.dual_result_files.items()},
    }
    write_json(out_dir / "dual_existing_result_manifest.json", payload)
    return payload


def resolve_model_keys(raw_models: str) -> list[str]:
    if raw_models.strip().lower() == "all":
        return list(MODEL_CONFIGS.keys())
    keys = []
    for raw in raw_models.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        key = MODEL_ALIASES.get(token)
        if key is None:
            raise ValueError(f"Unknown model '{raw}'. Choices: {', '.join(MODEL_CONFIGS)}")
        if key not in keys:
            keys.append(key)
    return keys


def resolve_requested_splits(raw_splits: str) -> set[str]:
    splits = {x.strip().lower() for x in raw_splits.split(",") if x.strip()}
    if not splits:
        raise ValueError("--splits cannot be empty")
    bad = sorted(splits - {"train", "val"})
    if bad:
        raise ValueError(f"Unsupported split(s): {', '.join(bad)}. Choices: train,val")
    return splits


def validate_paths(model: ModelConfig) -> list[str]:
    missing = []
    for label, path in (
        ("project_root", model.project_root),
        ("scan_path", model.scan_path),
        ("dual_head_file", model.dual_head_file),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    for split in model.splits:
        if not split.path.exists():
            missing.append(f"{split.split}_data: {split.path}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select CER-only/BCP-only heads and evaluate them with existing scripts."
    )
    parser.add_argument("--models", default="all", help="Comma-separated model keys or aliases; default: all")
    parser.add_argument("--position", default=DEFAULT_POSITION)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Actually run model evaluations")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--criteria", default=",".join(CRITERIA))
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits to evaluate: train,val")
    parser.add_argument("--target_tolerance", type=int, default=3)
    parser.add_argument("--device", default="auto", help="VQA_RAD ablation device")
    parser.add_argument("--device_map", default="auto", help="ConflictMedQA HF device_map")
    parser.add_argument("--mask_scope", default="all")
    parser.add_argument("--conflict_dtype", default="", choices=["", "float16", "bfloat16", "float32"])
    parser.add_argument("--vqa_dtype", default="", choices=["", "fp16", "bf16", "fp32"])
    parser.add_argument("--max_pairs", type=int, default=0, help="ConflictMedQA smoke-test cap; 0 means full")
    parser.add_argument("--max_examples", type=int, default=0, help="VQA_RAD smoke-test cap; 0 means full")
    parser.add_argument("--max_image_side", type=int, default=672, help="VQA_RAD image resize cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Use resume mode where supported")
    args = parser.parse_args()

    requested_criteria = tuple(x.strip().lower() for x in args.criteria.split(",") if x.strip())
    for criterion in requested_criteria:
        if criterion not in CRITERIA:
            raise ValueError(f"Unsupported criterion '{criterion}'. Choices: {', '.join(CRITERIA)}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    model_keys = resolve_model_keys(args.models)
    requested_splits = resolve_requested_splits(args.splits)
    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "position": args.position,
        "splits": sorted(requested_splits),
        "dry_run": not args.run,
        "models": {},
    }
    command_lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]

    for model_key in model_keys:
        model = MODEL_CONFIGS[model_key]
        missing = validate_paths(model)
        if missing:
            raise FileNotFoundError(
                f"Missing required files for {model.display_name}:\n- " + "\n- ".join(missing)
            )

        model_out = output_root / model.key / args.position
        select_out = model_out / "selected_heads"
        logs_out = model_out / "logs"
        model_out.mkdir(parents=True, exist_ok=True)

        scan_heads = load_scan_heads(model.scan_path)
        dual_manifest = copy_dual_metadata(model, model_out)
        target_count = int(dual_manifest["n_selected_heads"])

        model_payload: dict[str, Any] = {
            "display_name": model.display_name,
            "task": model.task,
            "project_root": str(model.project_root),
            "scan_path": str(model.scan_path),
            "dual": dual_manifest,
            "criteria": {},
        }

        for criterion in requested_criteria:
            selected, selection_meta = select_heads(
                scan_heads,
                criterion=criterion,
                target_count=target_count,
                tolerance=args.target_tolerance,
            )
            criterion_out = select_out / criterion
            head_file = write_selected_heads(criterion_out, model, criterion, selected, selection_meta)

            criterion_payload: dict[str, Any] = {
                "head_file": str(head_file),
                "selection_meta": selection_meta,
                "runs": {},
            }

            for split in model.splits:
                if split.split not in requested_splits:
                    continue
                cmd, result_path, summary_path = build_command(
                    model, split, head_file, model_out, criterion, args
                )
                rel_log_name = f"{criterion}_{split.split}.log"
                log_path = logs_out / rel_log_name
                command_lines.append(f"cd {shlex.quote(str(model.project_root))}")
                command_lines.append(f"mkdir -p {shlex.quote(str(log_path.parent))}")
                command_lines.append(f"{quote_cmd(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}")
                command_lines.append("")

                run_info = {
                    "command": cmd,
                    "command_text": quote_cmd(cmd),
                    "cwd": str(model.project_root),
                    "result": str(result_path),
                    "summary": str(summary_path),
                    "log": str(log_path),
                    "status": "not_run",
                }
                if args.run:
                    run_info.update(
                        run_command(
                            cmd,
                            cwd=model.project_root,
                            log_path=log_path,
                            skip_existing=args.skip_existing,
                            expected_result=summary_path,
                        )
                    )
                    if run_info.get("returncode", 0) != 0:
                        criterion_payload["runs"][split.split] = run_info
                        model_payload["criteria"][criterion] = criterion_payload
                        manifest["models"][model.key] = model_payload
                        write_json(output_root / "manifest.json", manifest)
                        return int(run_info["returncode"])
                criterion_payload["runs"][split.split] = run_info

            model_payload["criteria"][criterion] = criterion_payload

        manifest["models"][model.key] = model_payload

    write_json(output_root / "manifest.json", manifest)
    command_file = output_root / "run_commands.sh"
    command_file.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    command_file.chmod(0o755)

    print(
        json.dumps(
            {
                "saved_manifest": str(output_root / "manifest.json"),
                "saved_commands": str(command_file),
                "dry_run": not args.run,
                "models": model_keys,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
