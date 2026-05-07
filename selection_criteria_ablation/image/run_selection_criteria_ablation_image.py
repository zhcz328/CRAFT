#!/usr/bin/env python3
"""Run selection-criteria ablations for SLAKE image_conflict.

This runner is intentionally stored outside the original image_conflict project.
It reuses the existing ablation implementation by invoking:
  /root/logit_lens/Slake_vqa/image_conflict/ablate_head.py

Primary comparison metric:
  ctx_unknown_rate_delta
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


RUNNER_ROOT = Path(__file__).resolve().parent
IMAGE_CONFLICT_ROOT = Path("/root/logit_lens/Slake_vqa/image_conflict")
DEFAULT_OUTPUT_ROOT = RUNNER_ROOT / "results"
DEFAULT_POSITION = "image_conflict"
CRITERIA = ("cer-only", "bcp-only")


@dataclass(frozen=True)
class DatasetSplit:
    split: str
    path: Path


@dataclass(frozen=True)
class ModelConfig:
    key: str
    display_name: str
    model_path: str
    model_name: str
    slug: str
    scan_path: Path
    dual_head_file: Path
    dual_result_files: dict[str, Path]
    splits: tuple[DatasetSplit, ...]
    dtype: str = "bf16"
    trace_mode: str = "conflict"
    metrics: str = "follow_context"
    mask_scope: str = "all"
    keep_mode: str = "self"
    image_root: str = "/root/autodl-tmp/data/SLAKE/imgs"


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "hulumed-4b": ModelConfig(
        key="hulumed-4b",
        display_name="hulumed-4b",
        model_path="/root/autodl-tmp/Hulu-Med-4B",
        model_name="hulumed-4b",
        slug="hulumed4b",
        scan_path=IMAGE_CONFLICT_ROOT / "hulumed4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
        dual_head_file=IMAGE_CONFLICT_ROOT / "hulumed4b/result_image_conflict_slake/selected_heads_core_layers.json",
        dual_result_files={
            "train": IMAGE_CONFLICT_ROOT / "hulumed4b/result_image_conflict_slake/ablate/ablate_selected_heads_all_train.json",
            "val": IMAGE_CONFLICT_ROOT / "hulumed4b/result_image_conflict_slake/ablate/ablate_selected_heads_all_val.json",
        },
        splits=(
            DatasetSplit("train", IMAGE_CONFLICT_ROOT / "data/hulumed4b/slake_nc_correct_ic_ready_train.csv"),
            DatasetSplit("val", IMAGE_CONFLICT_ROOT / "data/hulumed4b/slake_nc_correct_ic_ready_val.csv"),
        ),
    ),
    "internvl3_5-4b": ModelConfig(
        key="internvl3_5-4b",
        display_name="internvl3_5-4b",
        model_path="/root/autodl-tmp/InternVL3_5-4B",
        model_name="internvl3_5-4b",
        slug="internvl35_4b",
        scan_path=IMAGE_CONFLICT_ROOT / "internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
        dual_head_file=IMAGE_CONFLICT_ROOT / "internvl35_4b/result_image_conflict_slake/selected_heads_core_layers.json",
        dual_result_files={
            "train": IMAGE_CONFLICT_ROOT / "internvl35_4b/result_image_conflict_slake/ablate/ablate_selected_heads_all_train.json",
            "val": IMAGE_CONFLICT_ROOT / "internvl35_4b/result_image_conflict_slake/ablate/ablate_selected_heads_all_val.json",
        },
        splits=(
            DatasetSplit("train", IMAGE_CONFLICT_ROOT / "data/internvl35_4b/slake_nc_correct_ic_ready_train.csv"),
            DatasetSplit("val", IMAGE_CONFLICT_ROOT / "data/internvl35_4b/slake_nc_correct_ic_ready_val.csv"),
        ),
    ),
}


MODEL_ALIASES = {
    "hulumed-4b": "hulumed-4b",
    "hulumed4b": "hulumed-4b",
    "hulu-med-4b": "hulumed-4b",
    "internvl3_5-4b": "internvl3_5-4b",
    "internvl35_4b": "internvl3_5-4b",
    "internvl35-4b": "internvl3_5-4b",
    "internvl3.5-4b": "internvl3_5-4b",
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
    if not isinstance(item, dict) or "layer" not in item or "head" not in item:
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
            "bcp_key": bcp_key,
            "source": source,
        }
    )
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
    for key in (
        "results",
        "top20",
        "heads",
        "selected",
        "selected_heads",
        "coarse_top20",
        "coarse_rerank_top20",
        "refined_top20",
        "rerank_top20",
    ):
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
        if old is None or row["cer"] > old["cer"] or (row["cer"] == old["cer"] and row["bcp"] < old["bcp"]):
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
        raise RuntimeError(f"Cannot parse selected heads from {path}")
    pairs = []
    seen: set[tuple[int, int]] = set()
    for item in items:
        if not isinstance(item, dict) or "layer" not in item or "head" not in item:
            continue
        pair = (int(item["layer"]), int(item["head"]))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append({"layer": pair[0], "head": pair[1]})
    if not pairs:
        raise RuntimeError(f"No valid selected heads found in {path}")
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
        description = "sort by mean_abs_effect_reduction descending"
    elif criterion == "bcp-only":
        ordered = sorted(heads, key=lambda h: (h["bcp"], -h["cer"], h["layer"], h["head"]))
        metric_key = "bcp"
        direction = "ascending"
        description = "sort by mean_abs_base_change ascending"
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    selected = ordered[: min(target_count, len(ordered))]
    count_diff = abs(len(selected) - target_count)
    if count_diff > tolerance:
        raise RuntimeError(
            f"{criterion} selected {len(selected)} heads, target {target_count}, tolerance {tolerance}"
        )
    meta = {
        "criterion": criterion,
        "selection_rule": description,
        "metric_key": metric_key,
        "direction": direction,
        "threshold_at_last_selected": selected[-1][metric_key] if selected else None,
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
            ["rank", "layer", "head", "CER_mean_abs_effect_reduction", "BCP", "BCP_source_key", "source"]
        )
        for rank, head in enumerate(selected, 1):
            writer.writerow(
                [
                    rank,
                    head["layer"],
                    head["head"],
                    f'{head["cer"]:.10g}',
                    f'{head["bcp"]:.10g}',
                    head.get("bcp_key", ""),
                    head.get("source", ""),
                ]
            )
    return json_path


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def build_command(
    model: ModelConfig,
    split: DatasetSplit,
    head_file: Path,
    out_dir: Path,
    criterion: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path]:
    result_dir = out_dir / criterion
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"{criterion}_{split.split}.json"
    cmd = [
        args.python,
        "ablate_head.py",
        "--data_csv",
        str(split.path),
        "--image_root",
        args.image_root or model.image_root,
        "--model",
        model.model_path,
        "--model_name",
        model.model_name,
        "--selected_heads",
        str(head_file),
        "--trace_mode",
        model.trace_mode,
        "--position",
        args.position,
        "--mask_scale",
        str(args.mask_scale),
        "--metrics",
        model.metrics,
        "--mask_scope",
        args.mask_scope,
        "--keep_mode",
        model.keep_mode,
        "--dtype",
        args.dtype or model.dtype,
        "--device",
        args.device,
        "--out_json",
        str(out_json),
        "--max_image_side",
        str(args.max_image_side),
        "--seed",
        str(args.seed),
    ]
    if args.max_examples > 0:
        cmd.extend(["--max_examples", str(args.max_examples)])
    if args.resume:
        cmd.append("--resume")
    return cmd, out_json


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    skip_existing: bool,
    expected_result: Path,
) -> dict[str, Any]:
    if skip_existing and expected_result.exists():
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
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ cd {cwd}\n")
        log.write(f"$ {quote_cmd(cmd)}\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
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
    bad = sorted(splits - {"train", "val"})
    if bad:
        raise ValueError(f"Unsupported split(s): {', '.join(bad)}. Choices: train,val")
    return splits


def validate_paths(model: ModelConfig) -> list[str]:
    missing = []
    for label, path in (("scan_path", model.scan_path), ("dual_head_file", model.dual_head_file)):
        if not path.exists():
            missing.append(f"{label}: {path}")
    for split in model.splits:
        if not split.path.exists():
            missing.append(f"{split.split}_data: {split.path}")
    return missing


def extract_summary_metrics(summary_path: Path) -> dict[str, Any]:
    data = read_json(summary_path)
    return {
        "summary_path": str(summary_path),
        "n_selected_heads": int(data.get("n_selected_heads", 0) or 0),
        "n_samples": int(data.get("n_samples", 0) or 0),
        "ctx_unknown_rate_delta": maybe_float(data.get("ctx_unknown", {}).get("rate_delta")) or 0.0,
        "ctx_unknown_base_rate": maybe_float(data.get("ctx_unknown", {}).get("base_rate")) or 0.0,
        "ctx_unknown_ab_rate": maybe_float(data.get("ctx_unknown", {}).get("ab_rate")) or 0.0,
        "pred_ctx_acc_delta": maybe_float(data.get("pred_ctx", {}).get("acc_delta")) or 0.0,
        "pred_nc_acc_delta": maybe_float(data.get("pred_nc", {}).get("acc_delta")) or 0.0,
        "mean_hallucination_relief": maybe_float(data.get("mean_hallucination_relief")) or 0.0,
        "mean_ic_follow_context_gain": maybe_float(data.get("mean_ic_follow_context_gain")) or 0.0,
        "mean_nc_gold_margin_damage": maybe_float(data.get("mean_nc_gold_margin_damage")) or 0.0,
    }


def write_comparison_reports(output_root: Path, manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model_key, model_payload in manifest.get("models", {}).items():
        dual_files = model_payload.get("dual", {}).get("result_files", {})
        for split, path_str in dual_files.items():
            path = Path(path_str)
            if path.exists():
                rows.append(
                    {
                        "model": model_key,
                        "criterion": "dual",
                        "split": split,
                        **extract_summary_metrics(path),
                    }
                )
        for criterion, criterion_payload in model_payload.get("criteria", {}).items():
            for split, run_info in criterion_payload.get("runs", {}).items():
                summary_path = Path(run_info["summary"])
                if summary_path.exists():
                    rows.append(
                        {
                            "model": model_key,
                            "criterion": criterion,
                            "split": split,
                            **extract_summary_metrics(summary_path),
                        }
                    )
    rows.sort(key=lambda r: (r["model"], r["split"], -r["ctx_unknown_rate_delta"], r["criterion"]))
    write_json(output_root / "ctx_unknown_rate_delta_comparison.json", rows)
    with (output_root / "ctx_unknown_rate_delta_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "split",
                "criterion",
                "ctx_unknown_rate_delta",
                "ctx_unknown_base_rate",
                "ctx_unknown_ab_rate",
                "pred_ctx_acc_delta",
                "pred_nc_acc_delta",
                "mean_hallucination_relief",
                "mean_ic_follow_context_gain",
                "mean_nc_gold_margin_damage",
                "n_selected_heads",
                "n_samples",
                "summary_path",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model"],
                    row["split"],
                    row["criterion"],
                    f'{row["ctx_unknown_rate_delta"]:.10g}',
                    f'{row["ctx_unknown_base_rate"]:.10g}',
                    f'{row["ctx_unknown_ab_rate"]:.10g}',
                    f'{row["pred_ctx_acc_delta"]:.10g}',
                    f'{row["pred_nc_acc_delta"]:.10g}',
                    f'{row["mean_hallucination_relief"]:.10g}',
                    f'{row["mean_ic_follow_context_gain"]:.10g}',
                    f'{row["mean_nc_gold_margin_damage"]:.10g}',
                    row["n_selected_heads"],
                    row["n_samples"],
                    row["summary_path"],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CER-only/BCP-only head-selection ablations for SLAKE image_conflict."
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--position", default=DEFAULT_POSITION)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--criteria", default=",".join(CRITERIA))
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--target_tolerance", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image_root", default="")
    parser.add_argument("--dtype", default="", choices=["", "fp16", "bf16", "fp32"])
    parser.add_argument("--mask_scope", default="all")
    parser.add_argument("--mask_scale", type=float, default=2.0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--max_image_side", type=int, default=672)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
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
        "runner_root": str(RUNNER_ROOT),
        "image_conflict_root": str(IMAGE_CONFLICT_ROOT),
        "output_root": str(output_root),
        "position": args.position,
        "splits": sorted(requested_splits),
        "dry_run": not args.run,
        "metric_focus": "ctx_unknown_rate_delta",
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
        model_out = output_root / model.slug / args.position
        select_out = model_out / "selected_heads"
        logs_out = model_out / "logs"
        model_out.mkdir(parents=True, exist_ok=True)

        scan_heads = load_scan_heads(model.scan_path)
        dual_manifest = copy_dual_metadata(model, model_out)
        target_count = int(dual_manifest["n_selected_heads"])
        model_payload: dict[str, Any] = {
            "display_name": model.display_name,
            "slug": model.slug,
            "scan_path": str(model.scan_path),
            "dual": dual_manifest,
            "criteria": {},
        }

        for criterion in requested_criteria:
            selected, selection_meta = select_heads(
                scan_heads, criterion, target_count, args.target_tolerance
            )
            head_file = write_selected_heads(select_out / criterion, model, criterion, selected, selection_meta)
            criterion_payload: dict[str, Any] = {
                "head_file": str(head_file),
                "selection_meta": selection_meta,
                "runs": {},
            }
            for split in model.splits:
                if split.split not in requested_splits:
                    continue
                cmd, summary_path = build_command(model, split, head_file, model_out, criterion, args)
                log_path = logs_out / f"{criterion}_{split.split}.log"
                command_lines.append(f"cd {shlex.quote(str(IMAGE_CONFLICT_ROOT))}")
                command_lines.append(f"mkdir -p {shlex.quote(str(log_path.parent))}")
                command_lines.append(f"{quote_cmd(cmd)} 2>&1 | tee {shlex.quote(str(log_path))}")
                command_lines.append("")
                run_info = {
                    "command": cmd,
                    "command_text": quote_cmd(cmd),
                    "cwd": str(IMAGE_CONFLICT_ROOT),
                    "summary": str(summary_path),
                    "log": str(log_path),
                    "status": "not_run",
                }
                if args.run:
                    run_info.update(
                        run_command(
                            cmd,
                            IMAGE_CONFLICT_ROOT,
                            log_path,
                            args.skip_existing,
                            summary_path,
                        )
                    )
                    if run_info.get("returncode", 0) != 0:
                        criterion_payload["runs"][split.split] = run_info
                        model_payload["criteria"][criterion] = criterion_payload
                        manifest["models"][model.key] = model_payload
                        write_json(output_root / "manifest.json", manifest)
                        write_comparison_reports(output_root, manifest)
                        return int(run_info["returncode"])
                criterion_payload["runs"][split.split] = run_info
            model_payload["criteria"][criterion] = criterion_payload
        manifest["models"][model.key] = model_payload

    write_json(output_root / "manifest.json", manifest)
    command_file = output_root / "run_commands.sh"
    command_file.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    command_file.chmod(0o755)
    write_comparison_reports(output_root, manifest)

    print(
        json.dumps(
            {
                "saved_manifest": str(output_root / "manifest.json"),
                "saved_commands": str(command_file),
                "saved_comparison_json": str(output_root / "ctx_unknown_rate_delta_comparison.json"),
                "saved_comparison_csv": str(output_root / "ctx_unknown_rate_delta_comparison.csv"),
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
