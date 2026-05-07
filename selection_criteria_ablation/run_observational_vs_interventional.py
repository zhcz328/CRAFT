#!/usr/bin/env python3
"""Run observational head selection and compare against existing interventional ablations.

This script reuses the existing project evaluators and current interventional
results. It adds one new criterion:

  observational:
    Rank heads by the absolute Pearson correlation between
    - natural attention mass from the final prompt query token to the injected
      evidence block, and
    - whether the model follows the conflict answer under the conflict prompt.

Selection is performed on the train split. Evaluation is done on the val split.
Interventional numbers are read from the existing result files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer

from run_selection_criteria_ablation import (
    MODEL_CONFIGS,
    build_command,
    load_dual_heads,
    load_scan_heads,
    model_path_for,
    quote_cmd,
    read_json,
    resolve_model_keys,
    run_command,
    validate_paths,
    write_json,
)


ABLATION_ROOT = Path("/root/logit_lens/selection_criteria_ablation")
DEFAULT_OUTPUT_ROOT = ABLATION_ROOT / "results_observational_vs_interventional"
DEFAULT_POSITION = "before_question"
DEFAULT_PYTHON = "/root/miniconda3/envs/latentmas/bin/python"


def pearson_abs(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n <= 1:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = 0.0
    dx = 0.0
    dy = 0.0
    for x, y in zip(xs[:n], ys[:n]):
        a = x - mx
        b = y - my
        num += a * b
        dx += a * a
        dy += b * b
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return abs(num / math.sqrt(dx * dy))


def find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle)
    for start in range(last + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def best_span_from_candidates(
    input_ids: list[int],
    token_candidates: list[list[int]],
) -> tuple[int, int] | None:
    for cand in token_candidates:
        match = find_subsequence(input_ids, cand)
        if match is not None:
            return match
    return None


def candidate_pairs_from_scan(scan_heads: list[dict[str, Any]]) -> dict[int, list[int]]:
    layer2heads: dict[int, set[int]] = defaultdict(set)
    for row in scan_heads:
        layer2heads[int(row["layer"])].add(int(row["head"]))
    return {layer: sorted(heads) for layer, heads in sorted(layer2heads.items())}


def write_selected_heads(
    out_dir: Path,
    selected: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "observational",
        "criterion": "observational",
        "selected_heads": selected,
        "selected": selected,
        "n_selected": len(selected),
        "meta": meta,
    }
    json_path = out_dir / "observational_selected_heads.json"
    csv_path = out_dir / "observational_selected_heads.csv"
    write_json(json_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "layer",
                "head",
                "observational_score",
                "mean_feature",
                "positive_rate",
                "n_samples",
            ]
        )
        for rank, row in enumerate(selected, 1):
            writer.writerow(
                [
                    rank,
                    row["layer"],
                    row["head"],
                    f'{row["observational_score"]:.10g}',
                    f'{row["mean_feature"]:.10g}',
                    f'{row["positive_rate"]:.10g}',
                    row["n_samples"],
                ]
            )
    return json_path


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_attn_tensor(obj: Any) -> torch.Tensor | None:
    if torch.is_tensor(obj) and obj.dim() == 4:
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            found = find_attn_tensor(item)
            if found is not None:
                return found
    if hasattr(obj, "attn_weights"):
        found = find_attn_tensor(getattr(obj, "attn_weights"))
        if found is not None:
            return found
    return None


def capture_attentions_with_hooks(model, attn_modules: dict[int, Any], forward_kwargs: dict[str, Any]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            attn = find_attn_tensor(output)
            if attn is not None:
                captured[layer_idx] = attn.detach().float().cpu()
        return hook

    try:
        for layer_idx, module in attn_modules.items():
            handles.append(module.register_forward_hook(make_hook(int(layer_idx))))
        with torch.no_grad():
            model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def find_self_attn_modules_generic(model) -> dict[int, Any]:
    layer2attn: dict[int, Any] = {}
    for name, module in model.named_modules():
        if not name.endswith(".self_attn"):
            continue
        m = re.search(r"(?:^|\.)(\d+)\.self_attn$", name)
        if m is None:
            continue
        layer2attn[int(m.group(1))] = module
    return layer2attn


def capture_attention_mass_with_hooks(
    model,
    attn_modules: dict[int, Any],
    candidate_layer_heads: dict[int, list[int]],
    query_pos: int,
    span: tuple[int, int],
    forward_kwargs: dict[str, Any],
) -> dict[tuple[int, int], float]:
    masses: dict[tuple[int, int], float] = {}
    handles = []

    def make_hook(layer_idx: int):
        heads = candidate_layer_heads.get(layer_idx, [])

        def hook(_module, _inputs, output):
            attn = find_attn_tensor(output)
            if attn is None:
                return
            if attn.dim() != 4 or attn.shape[0] == 0:
                return

            q_len = int(attn.shape[-2])
            k_len = int(attn.shape[-1])
            if q_len <= 0 or k_len <= 0:
                return

            q = min(max(int(query_pos), 0), q_len - 1)
            start = min(max(int(span[0]), 0), k_len)
            end = min(max(int(span[1]), start), k_len)
            if start >= end:
                return

            head_tensor = attn[0]
            valid_heads = [h for h in heads if 0 <= int(h) < int(head_tensor.shape[0])]
            if not valid_heads:
                return

            vals = (
                head_tensor[valid_heads, q, start:end]
                .sum(dim=-1)
                .detach()
                .float()
                .cpu()
                .tolist()
            )
            for head_idx, val in zip(valid_heads, vals):
                masses[(layer_idx, int(head_idx))] = float(val)

        return hook

    try:
        for layer_idx, module in attn_modules.items():
            handles.append(module.register_forward_hook(make_hook(int(layer_idx))))
        with torch.no_grad():
            model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    return masses


def load_conflict_module(project_root: Path):
    return import_module_from_path(
        f"ablate_head_inf_{project_root.name}",
        project_root / "ablate_head_inf.py",
    )


def compute_conflictmedqa_observational(
    model_key: str,
    project_root: Path,
    model_path: str,
    pairs_path: Path,
    candidate_layer_heads: dict[int, list[int]],
    device_map: str,
    dtype_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mod = load_conflict_module(project_root)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()

    pairs = mod.read_jsonl(str(pairs_path))
    feature_values: dict[tuple[int, int], list[float]] = {
        (layer, head): []
        for layer, heads in candidate_layer_heads.items()
        for head in heads
    }
    labels: list[float] = []
    skipped_span = 0
    skipped_attn = 0
    layer2attn = find_self_attn_modules_generic(model)
    attn_modules = {layer: layer2attn[layer] for layer in candidate_layer_heads.keys() if layer in layer2attn}

    for rec in pairs:
        for side in ("correct", "wrong"):
            s = rec[side]
            base_prompt = s["prompt"]
            gold_yes = side == "correct"
            conflict_yes = not gold_yes
            gold = mod.YES if gold_yes else mod.NO
            wrong = mod.YES if conflict_yes else mod.NO

            conflict_user = mod.inject_evidence(base_prompt, mod.make_evidence(conflict_yes), DEFAULT_POSITION)
            conflict_prompt = mod.wrap_as_chat(tokenizer, conflict_user, enable_thinking=False)

            enc = tokenizer(conflict_prompt, return_tensors="pt")
            input_ids = enc["input_ids"][0].tolist()
            evidence_ids = tokenizer(mod.make_evidence(conflict_yes), add_special_tokens=False).input_ids
            span = best_span_from_candidates(input_ids, [evidence_ids])
            if span is None:
                skipped_span += 1
                continue

            enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
            query_pos = int(enc["attention_mask"][0].sum().item()) - 1

            masses: dict[tuple[int, int], float] = {}
            try:
                masses = capture_attention_mass_with_hooks(
                    model,
                    attn_modules,
                    candidate_layer_heads=candidate_layer_heads,
                    query_pos=query_pos,
                    span=span,
                    forward_kwargs={
                        **enc,
                        "output_attentions": False,
                        "use_cache": False,
                        "return_dict": True,
                    },
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not masses:
                skipped_attn += 1
                del enc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            pred_conflict, _yes_lp, _no_lp, _delta = mod.score_yes_no_fast(model, tokenizer, conflict_prompt)
            labels.append(1.0 if pred_conflict == wrong else 0.0)

            for layer_idx, heads in candidate_layer_heads.items():
                for head_idx in heads:
                    key = (layer_idx, head_idx)
                    if key in masses:
                        feature_values[key].append(float(masses[key]))

            del masses
            del enc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        results = []
        positive_rate = sum(labels) / len(labels) if labels else 0.0
        for (layer, head), values in feature_values.items():
            if len(values) != len(labels) or not values:
                continue
            score = pearson_abs(values, labels)
            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "observational_score": score,
                    "mean_feature": float(sum(values) / len(values)),
                    "positive_rate": positive_rate,
                    "n_samples": len(values),
                }
            )
        results.sort(
            key=lambda row: (
                -row["observational_score"],
                -row["mean_feature"],
                row["layer"],
                row["head"],
            )
        )
        meta = {
            "model_key": model_key,
            "task": "conflictmedqa",
            "position": DEFAULT_POSITION,
            "feature": "final-query attention mass to inserted evidence block",
            "target": "follow_conflict label on conflict prompt (1 if prediction equals conflict answer)",
            "score": "absolute Pearson correlation",
            "pairs_path": str(pairs_path),
            "n_labels": len(labels),
            "positive_rate": positive_rate,
            "candidate_head_count": len(feature_values),
            "skipped_span": skipped_span,
            "skipped_attn": skipped_attn,
        }
        return results, meta
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_vqa_module(model_key: str, project_root: Path):
    sys.path.insert(0, str(project_root))
    try:
        return import_module_from_path(f"ablate_head_{model_key}", project_root / "ablate_head.py")
    finally:
        try:
            sys.path.remove(str(project_root))
        except ValueError:
            pass


def build_vqa_model(ablate_mod, model_path: str, dtype_name: str):
    dtype = ablate_mod.str2dtype(dtype_name)
    # Reuse the exact loader path used by VQA ablation scripts when available
    # (notably for InternVL wrappers / token-id wiring).
    shared_loader = getattr(ablate_mod, "shared_load_mm_model", None)
    if callable(shared_loader):
        return shared_loader(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )

    load_errs = []
    model = None
    for cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            load_errs.append(f"{cls.__name__}: {repr(exc)}")
    if model is None:
        raise RuntimeError("Failed to load model. " + " | ".join(load_errs))
    return model


def compute_vqa_observational(
    model_key: str,
    project_root: Path,
    model_path: str,
    data_csv: Path,
    image_root: str,
    candidate_layer_heads: dict[int, list[int]],
    dtype_name: str,
    trace_mode: str,
    max_image_side: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mod = load_vqa_module(model_key, project_root)
    processor = mod.load_processor_with_compat(model_path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_vqa_model(mod, model_path, dtype_name).to(device)
    model.eval()

    df = mod.pd.read_csv(data_csv)
    resolved_image_root = str((project_root / image_root).resolve())
    args = SimpleNamespace(
        image_root=resolved_image_root,
        max_examples=0,
        max_image_side=max_image_side,
        position=DEFAULT_POSITION,
        trace_mode=trace_mode,
    )
    samples = mod.build_samples(args, df, processor, tokenizer, model, device)

    feature_values: dict[tuple[int, int], list[float]] = {
        (layer, head): []
        for layer, heads in candidate_layer_heads.items()
        for head in heads
    }
    labels: list[float] = []
    skipped_span = 0
    skipped_attn = 0

    layer2attn, _ = mod._find_self_attn_modules(model)
    attn_modules = {layer: layer2attn[layer] for layer in candidate_layer_heads.keys() if layer in layer2attn}

    for sample in samples:
        forward_inputs = None
        masses: dict[tuple[int, int], float] = {}
        try:
            evidence_block = mod.EVIDENCE_TMPL.format(ans=sample["wrong"] if trace_mode == "conflict" else sample["gold"])
            raw_input_ids = sample["ctx_inputs"]["input_ids"][0].tolist()
            token_candidates = [
                tokenizer(evidence_block, add_special_tokens=False).input_ids,
                tokenizer(" " + sample["wrong"], add_special_tokens=False).input_ids,
                tokenizer(sample["wrong"], add_special_tokens=False).input_ids,
            ]
            span = best_span_from_candidates(raw_input_ids, token_candidates)
            if span is None:
                skipped_span += 1
                continue

            target_dtype = mod.infer_vision_input_dtype(model)
            forward_inputs = mod.to_device(sample["ctx_inputs"], next(model.parameters()).device, float_dtype=target_dtype)

            query_pos = int(sample["ctx_inputs"]["attention_mask"][0].sum().item()) - 1
            try:
                masses = capture_attention_mass_with_hooks(
                    model,
                    attn_modules,
                    candidate_layer_heads=candidate_layer_heads,
                    query_pos=query_pos,
                    span=span,
                    forward_kwargs={
                        **forward_inputs,
                        "output_attentions": False,
                        "use_cache": False,
                        "return_dict": True,
                    },
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not masses:
                skipped_attn += 1
                continue

            labels.append(1.0 if sample["base_ctx_pred"] == "wrong" else 0.0)
            for layer_idx, heads in candidate_layer_heads.items():
                for head_idx in heads:
                    key = (layer_idx, head_idx)
                    if key in masses:
                        feature_values[key].append(float(masses[key]))
        finally:
            if isinstance(sample, dict):
                sample.clear()
            if forward_inputs is not None:
                del forward_inputs
            if masses:
                del masses
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        results = []
        positive_rate = sum(labels) / len(labels) if labels else 0.0
        for (layer, head), values in feature_values.items():
            if len(values) != len(labels) or not values:
                continue
            score = pearson_abs(values, labels)
            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "observational_score": score,
                    "mean_feature": float(sum(values) / len(values)),
                    "positive_rate": positive_rate,
                    "n_samples": len(values),
                }
            )
        results.sort(
            key=lambda row: (
                -row["observational_score"],
                -row["mean_feature"],
                row["layer"],
                row["head"],
            )
        )
        meta = {
            "model_key": model_key,
            "task": "vqa_rad",
            "position": DEFAULT_POSITION,
            "feature": "final-query attention mass to inserted evidence block",
            "target": "follow_conflict label on conflict prompt (1 if prediction equals conflict answer)",
            "score": "absolute Pearson correlation",
            "data_csv": str(data_csv),
            "n_labels": len(labels),
            "positive_rate": positive_rate,
            "candidate_head_count": len(feature_values),
            "skipped_span": skipped_span,
            "skipped_attn": skipped_attn,
        }
        return results, meta
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_observational_heads(
    model_key: str,
    model,
    scan_heads: list[dict[str, Any]],
    target_count: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_layer_heads = candidate_pairs_from_scan(scan_heads)

    if model.task == "conflictmedqa":
        results, meta = compute_conflictmedqa_observational(
            model_key=model_key,
            project_root=model.project_root,
            model_path=model_path_for(model),
            pairs_path=next(split.path for split in model.splits if split.split == "train"),
            candidate_layer_heads=candidate_layer_heads,
            device_map=args.device_map,
            dtype_name=args.conflict_dtype or model.dtype,
        )
    elif model.task == "vqa_rad":
        results, meta = compute_vqa_observational(
            model_key=model_key,
            project_root=model.project_root,
            model_path=model_path_for(model),
            data_csv=next(split.path for split in model.splits if split.split == "train"),
            image_root=model.image_root,
            candidate_layer_heads=candidate_layer_heads,
            dtype_name=args.vqa_dtype or model.dtype,
            trace_mode=model.trace_mode,
            max_image_side=args.max_image_side,
        )
    else:  # pragma: no cover - no third task type here
        raise ValueError(model.task)

    selected = results[: min(target_count, len(results))]
    selection_meta = {
        "criterion": "observational",
        "selection_rule": "sort by absolute Pearson correlation descending",
        "metric_key": "observational_score",
        "direction": "descending",
        "threshold_at_last_selected": selected[-1]["observational_score"] if selected else None,
        "target_count_from_dual": target_count,
        "selected_count": len(selected),
        "n_candidates": len(results),
        **meta,
    }
    return selected, selection_meta


def cfr_reduction_points_from_summary(model_key: str, model, summary_path: Path) -> float:
    obj = read_json(summary_path)
    if model.task == "conflictmedqa":
        if "summary_by_position" in obj:
            pos = obj["summary_by_position"][DEFAULT_POSITION]
            if "follow_conflict_rate_before" in pos and "follow_conflict_rate_after" in pos:
                before = float(pos["follow_conflict_rate_before"])
                after = float(pos["follow_conflict_rate_after"])
                return 100.0 * (before - after)
            if "follow_conflict_rate" in pos:
                split = "val" if "val" in summary_path.stem else "train"
                sibling = summary_path.with_name(f"standard_metrics_{split}.json")
                if sibling.exists():
                    sibling_obj = read_json(sibling)
                    before = float(sibling_obj["baseline"]["attack_success_rate"])
                    after = float(sibling_obj["selected_ablation"]["metrics"]["attack_success_rate"])
                    return 100.0 * (before - after)
        baseline = obj.get("baseline", {})
        selected = obj.get("selected_ablation", {}).get("metrics", {})
        if baseline and selected:
            before = float(baseline["attack_success_rate"])
            after = float(selected["attack_success_rate"])
            return 100.0 * (before - after)
        raise KeyError(f"Unsupported conflictmedqa summary format: {summary_path}")

    if model.task == "vqa_rad":
        pred_ctx = obj["pred_ctx"]
        base_cfr = 1.0 - float(pred_ctx["base_acc"])
        ab_cfr = 1.0 - float(pred_ctx["ab_acc"])
        return 100.0 * (base_cfr - ab_cfr)

    raise ValueError(model.task)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select observational heads on train, evaluate on val, and compare with existing interventional ablations."
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--position", default=DEFAULT_POSITION)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--mask_scope", default="all")
    parser.add_argument("--conflict_dtype", default="", choices=["", "float16", "bfloat16", "float32"])
    parser.add_argument("--vqa_dtype", default="", choices=["", "fp16", "bf16", "fp32"])
    parser.add_argument("--max_image_side", type=int, default=672)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_keys = resolve_model_keys(args.models)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "position": DEFAULT_POSITION,
        "models": {},
        "dry_run": not args.run,
    }
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    comparison_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        model = MODEL_CONFIGS[model_key]
        missing = validate_paths(model)
        if missing:
            raise FileNotFoundError(
                f"Missing required files for {model.display_name}:\n- " + "\n- ".join(missing)
            )

        model_out = output_root / model.key / DEFAULT_POSITION
        model_out.mkdir(parents=True, exist_ok=True)
        logs_out = model_out / "logs"
        heads_out = model_out / "selected_heads" / "observational"
        obs_scan_path = model_out / "observational_scan.json"

        scan_heads = load_scan_heads(model.scan_path)
        dual_heads = load_dual_heads(model.dual_head_file)
        target_count = len(dual_heads)

        if args.skip_existing and obs_scan_path.exists():
            obs_scan = read_json(obs_scan_path)
            selected = obs_scan["selected_heads"]
            selection_meta = obs_scan["meta"]
        else:
            selected, selection_meta = select_observational_heads(
                model_key=model_key,
                model=model,
                scan_heads=scan_heads,
                target_count=target_count,
                args=args,
            )
            obs_scan = {
                "model": model.display_name,
                "task": model.task,
                "position": DEFAULT_POSITION,
                "selected_heads": selected,
                "meta": selection_meta,
            }
            write_json(obs_scan_path, obs_scan)

        obs_head_file = write_selected_heads(heads_out, selected, selection_meta)

        val_split = next(split for split in model.splits if split.split == "val")
        cmd, result_path, summary_path = build_command(
            model,
            val_split,
            obs_head_file,
            model_out,
            "observational",
            args,
        )
        log_path = logs_out / "observational_val.log"
        command_lines.append(f"cd {model.project_root}")
        command_lines.append(f"{quote_cmd(cmd)} 2>&1 | tee {str(log_path)}")
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
                write_json(output_root / "manifest.json", manifest)
                return int(run_info["returncode"])

        interventional_summary = model.dual_result_files["val"]
        observational_reduction = (
            cfr_reduction_points_from_summary(model_key, model, summary_path)
            if summary_path.exists()
            else None
        )
        interventional_reduction = cfr_reduction_points_from_summary(model_key, model, interventional_summary)

        comparison_row = {
            "model_key": model.key,
            "display_name": model.display_name,
            "task": model.task,
            "position": DEFAULT_POSITION,
            "n_selected_heads": target_count,
            "observational_reduction_points": observational_reduction,
            "interventional_reduction_points": interventional_reduction,
            "observational_summary": str(summary_path),
            "interventional_summary": str(interventional_summary),
            "observational_head_file": str(obs_head_file),
            "interventional_head_file": str(model.dual_head_file),
        }
        comparison_rows.append(comparison_row)

        manifest["models"][model.key] = {
            "display_name": model.display_name,
            "task": model.task,
            "observational": {
                "scan_path": str(obs_scan_path),
                "head_file": str(obs_head_file),
                "selection_meta": selection_meta,
                "run": run_info,
                "reduction_points": observational_reduction,
            },
            "interventional": {
                "head_file": str(model.dual_head_file),
                "summary": str(interventional_summary),
                "reduction_points": interventional_reduction,
            },
        }
        write_json(output_root / "manifest.json", manifest)

    compare_json = output_root / "observational_vs_interventional_comparison.json"
    compare_csv = output_root / "observational_vs_interventional_comparison.csv"
    write_json(compare_json, comparison_rows)
    with compare_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_key",
                "display_name",
                "task",
                "position",
                "n_selected_heads",
                "observational_reduction_points",
                "interventional_reduction_points",
                "observational_summary",
                "interventional_summary",
                "observational_head_file",
                "interventional_head_file",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    command_file = output_root / "run_commands.sh"
    command_file.write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    command_file.chmod(0o755)

    print(
        json.dumps(
            {
                "saved_manifest": str(output_root / "manifest.json"),
                "saved_compare_json": str(compare_json),
                "saved_compare_csv": str(compare_csv),
                "saved_commands": str(command_file),
                "models": model_keys,
                "dry_run": not args.run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
