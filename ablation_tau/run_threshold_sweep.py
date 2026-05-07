from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics_utils import compute_qwen_metrics, compute_vqa_metrics, load_json, stronger_than


ROOT = Path("/root/logit_lens/ablation_tau")
POSITION_DEFAULT = "before_question"


@dataclass
class TargetConfig:
    name: str
    family: str
    project_root: Path
    select_script: Path
    ablate_script: Path | None
    current_selection_path: Path
    current_ablation_path: Path
    current_mask_scope: str
    current_split: str
    position: str
    selection_mode: str
    scan_input_json: Path | None = None
    qwen_summary_path: Path | None = None
    model: str | None = None
    model_name: str = ""
    data_csv: str | None = None
    image_root: str | None = None
    trace_mode: str = "conflict"
    keep_mode: str = "self"
    dtype: str = "bf16"
    device: str = "auto"
    max_examples: int = -1
    max_image_side: int = 672
    select_source: str = "results"
    select_topk: int = 50
    pairs_jsonl: str | None = None


RNG_SEED = 20260427


TARGETS: dict[str, TargetConfig] = {
    "hulumed_before_question_val": TargetConfig(
        name="hulumed_before_question_val",
        family="vqa",
        project_root=Path("/root/logit_lens/VQA_RAD/Hulu-med"),
        select_script=Path("/root/logit_lens/VQA_RAD/Hulu-med/select_heads_merged_unique_layers.py"),
        ablate_script=Path("/root/logit_lens/VQA_RAD/Hulu-med/ablate_head.py"),
        current_selection_path=Path("/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json"),
        current_ablation_path=Path("/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/ablate_selected_heads_ctx_only_hulumed4b_val.json"),
        current_mask_scope="ctx_only",
        current_split="val",
        position="before_question",
        selection_mode="merged_unique_layers",
        scan_input_json=Path("/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/head_scan_merged_unique_layers.json"),
        model="/root/autodl-tmp/Hulu-Med-4B",
        data_csv="/root/logit_lens/VQA_RAD/Hulu-med/data/nc_cc_both_correct_rerun_tmp_val.csv",
        image_root=".",
        device="auto",
    ),
    "internvl35_4b_text_conflict_before_question_val": TargetConfig(
        name="internvl35_4b_text_conflict_before_question_val",
        family="vqa",
        project_root=Path("/root/logit_lens/VQA_RAD/text_conflict"),
        select_script=Path("/root/logit_lens/VQA_RAD/text_conflict/select_heads_merged_unique_layers.py"),
        ablate_script=Path("/root/logit_lens/VQA_RAD/text_conflict/ablate_head.py"),
        current_selection_path=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json"),
        current_ablation_path=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_val.json"),
        current_mask_scope="all",
        current_split="val",
        position="before_question",
        selection_mode="merged_unique_layers",
        scan_input_json=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json"),
        model="/root/autodl-tmp/InternVL3_5-4B",
        model_name="internvl3_5-4b",
        data_csv="/root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv",
        image_root=".",
        device="auto",
    ),
    "text_conflict_before_question_val": TargetConfig(
        name="internvl35_4b_text_conflict_before_question_val",
        family="vqa",
        project_root=Path("/root/logit_lens/VQA_RAD/text_conflict"),
        select_script=Path("/root/logit_lens/VQA_RAD/text_conflict/select_heads_merged_unique_layers.py"),
        ablate_script=Path("/root/logit_lens/VQA_RAD/text_conflict/ablate_head.py"),
        current_selection_path=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json"),
        current_ablation_path=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_val.json"),
        current_mask_scope="all",
        current_split="val",
        position="before_question",
        selection_mode="merged_unique_layers",
        scan_input_json=Path("/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/headscan_vqarad_mm_accel/head_scan_merged_unique_layers.json"),
        model="/root/autodl-tmp/InternVL3_5-4B",
        model_name="internvl3_5-4b",
        data_csv="/root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv",
        image_root=".",
        device="auto",
    ),
    "qwen_conflictmedqa_before_question_val": TargetConfig(
        name="qwen_conflictmedqa_before_question_val",
        family="qwen",
        project_root=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp"),
        select_script=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/select_heads.py"),
        ablate_script=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/ablate_head_inf.py"),
        current_selection_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/headscan_rounds_top50_inf/head_groups.json"),
        current_ablation_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/ablate_heads_val.jsonl"),
        current_mask_scope="conflict_only",
        current_split="val",
        position="before_question",
        selection_mode="qwen_legacy",
        qwen_summary_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/headscan_rounds_top50_inf/head_scan_summary.json"),
        model="/root/autodl-tmp/qwen3-4B",
        pairs_jsonl="/root/logit_lens/conflictmedqa/Qwen3-4B_exp/data/kept_pairs_a12_b10_all_val.jsonl",
    ),
    "llama32_3b_conflictmedqa_before_question_val": TargetConfig(
        name="llama32_3b_conflictmedqa_before_question_val",
        family="qwen",
        project_root=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp"),
        select_script=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/select_heads.py"),
        ablate_script=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/ablate_head_inf.py"),
        current_selection_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/headscan_rounds_top30_inf/selected_heads.json"),
        current_ablation_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/conflict_retest_ablated_heads_inf_before_question_val.jsonl"),
        current_mask_scope="all",
        current_split="val",
        position="before_question",
        selection_mode="qwen_legacy",
        qwen_summary_path=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/headscan_rounds_top30_inf/head_scan_summary.json"),
        model="/root/autodl-tmp/Llama-3.2-3B-Instruct",
        pairs_jsonl="/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/data/kept_pairs_a12_b10_all_val.jsonl",
    ),
}


def _load_module(module_name: str, path: Path):
    module_dir = str(path.parent)
    inserted = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted = True
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def _candidate_signature(selected_rows: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    pairs = sorted((int(row["layer"]), int(row["head"])) for row in selected_rows)
    return tuple(pairs)


def _current_metrics_for_target(target: TargetConfig) -> dict[str, Any]:
    if target.family == "qwen":
        return compute_qwen_metrics(target.current_ablation_path)
    return compute_vqa_metrics(target.current_ablation_path)


def _load_current_selection_payload(target: TargetConfig) -> dict[str, Any]:
    return load_json(target.current_selection_path)


def _shuffle_in_place(rows: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(rows)


def _enumerate_vqa_threshold_candidates(target: TargetConfig, limit_pool: int) -> list[dict[str, Any]]:
    mod = _load_module(f"select_{target.name}", target.select_script)
    _obj, rows = mod.load_scan(str(target.scan_input_json), source=target.select_source, topk=target.select_topk)
    current_payload = _load_current_selection_payload(target)
    current_sig = _candidate_signature(current_payload.get("selected", []))

    eff_values = sorted({round(float(r["mean_abs_effect_reduction"]), 12) for r in rows})
    base_values = sorted({round(float(r["mean_abs_base_change"]), 12) for r in rows})
    eff_candidates = eff_values[:]
    base_candidates = base_values[:]
    rng = random.Random(RNG_SEED)
    rng.shuffle(eff_candidates)
    rng.shuffle(base_candidates)
    if limit_pool > 0:
        eff_candidates = eff_candidates[: max(1, limit_pool)]
        base_candidates = base_candidates[: max(1, limit_pool)]

    candidates: list[dict[str, Any]] = []
    seen_sigs: set[tuple[tuple[int, int], ...]] = {current_sig}
    for eff_min, base_max in itertools.product(eff_candidates, base_candidates):
        selected = mod.filter_heads(rows, "conflict_specific", eff_min=eff_min, base_max=base_max, base_min=None)
        selected = mod.attach_scores(selected)
        selected = mod.sort_selected(selected, "conflict_specific")
        if not selected:
            continue
        sig = _candidate_signature(selected)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        candidates.append(
            {
                "thresholds": {"eff_min": eff_min, "base_max": base_max, "base_min": None},
                "n_selected": len(selected),
                "selected": selected,
                "signature": sig,
            }
        )
    _shuffle_in_place(candidates, seed=RNG_SEED + 11)
    return candidates


def _enumerate_qwen_threshold_candidates(target: TargetConfig, limit_pool: int) -> list[dict[str, Any]]:
    mod = _load_module(f"select_{target.name}", target.select_script)
    heads = mod.load_heads_from_summary(str(target.qwen_summary_path))
    current_payload = _load_current_selection_payload(target)
    current_thr = current_payload.get("fixed_thresholds", {}).get("conflict_specific", {})
    current_selected_rows = current_payload.get("conflict_specific", [])
    if not current_thr:
        meta = current_payload.get("meta", {})
        if meta:
            current_thr = {
                "effect_reduction_gt": meta.get("er_min"),
                "base_delta_change_lt": meta.get("bc_max"),
            }
        current_selected_rows = current_payload.get("selected_heads", [])
    current_sig = tuple(
        sorted((int(row["layer"]), int(row["head"])) for row in current_selected_rows)
    )

    eff_values = sorted({round(float(row["er"]), 12) for row in heads})
    base_values = sorted({round(float(row["bc"]), 12) for row in heads})
    eff_candidates = eff_values[:]
    base_candidates = base_values[:]
    rng = random.Random(RNG_SEED + 1)
    rng.shuffle(eff_candidates)
    rng.shuffle(base_candidates)
    if limit_pool > 0:
        eff_candidates = eff_candidates[: max(1, limit_pool)]
        base_candidates = base_candidates[: max(1, limit_pool)]

    candidates: list[dict[str, Any]] = []
    seen_sigs: set[tuple[tuple[int, int], ...]] = {current_sig}
    for eff_min, base_max in itertools.product(eff_candidates, base_candidates):
        selected = [
            row for row in heads
            if float(row["er"]) >= eff_min and float(row["bc"]) <= base_max
        ]
        selected = sorted(selected, key=lambda x: (-x["cs_score"], -x["er"], x["bc"]))
        if not selected:
            continue
        sig = tuple(sorted((int(row["layer"]), int(row["head"])) for row in selected))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        candidates.append(
            {
                "thresholds": {
                    "effect_reduction_gt": eff_min,
                    "base_delta_change_lt": base_max,
                },
                "n_selected": len(selected),
                "selected": selected,
                "signature": sig,
            }
        )
    _shuffle_in_place(candidates, seed=RNG_SEED + 23)
    return candidates


def _write_vqa_selection_json(target: TargetConfig, out_path: Path, candidate: dict[str, Any]) -> None:
    current_payload = _load_current_selection_payload(target)
    payload = {
        "mode": "conflict_specific",
        "thresholds": candidate["thresholds"],
        "input_json": str(target.scan_input_json),
        "source": target.select_source,
        "topk": target.select_topk,
        "meta": current_payload.get("meta", {}),
        "n_candidates": current_payload.get("n_candidates"),
        "n_selected": candidate["n_selected"],
        "selected": candidate["selected"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_qwen_selection_json(target: TargetConfig, out_path: Path, candidate: dict[str, Any]) -> None:
    current_payload = _load_current_selection_payload(target)
    meta_payload = {}
    if current_payload.get("meta"):
        meta_payload = dict(current_payload.get("meta", {}))
        meta_payload["er_min"] = candidate["thresholds"]["effect_reduction_gt"]
        meta_payload["bc_max"] = candidate["thresholds"]["base_delta_change_lt"]
        meta_payload["selected"] = candidate["n_selected"]
    else:
        meta_payload = {
            "mode": "B_fixed",
            "er_min": candidate["thresholds"]["effect_reduction_gt"],
            "bc_max": candidate["thresholds"]["base_delta_change_lt"],
            "total_candidates": current_payload.get("n_candidates", current_payload.get("n_candidates", 0)),
            "selected": candidate["n_selected"],
        }

    payload = {
        "meta": meta_payload,
        "selected_heads": candidate["selected"],
        "saved": str(out_path),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_ablation_for_candidate(target: TargetConfig, selection_path: Path, out_path: Path) -> None:
    if target.ablate_script is None:
        raise RuntimeError(f"No ablation script configured for {target.name}")

    if target.family == "qwen":
        summary_path = out_path.with_name(out_path.stem + "_summary.json")
        cmd = [
            sys.executable,
            str(target.ablate_script),
            "--pairs",
            str(target.pairs_jsonl),
            "--model",
            str(target.model),
            "--head_file",
            str(selection_path),
            "--out",
            str(out_path),
            "--summary_out",
            str(summary_path),
            "--dtype",
            "bfloat16",
            "--device_map",
            "auto",
            "--positions",
            target.position,
            "--mask_scope",
            target.current_mask_scope,
        ]
    else:
        cmd = [
            sys.executable,
            str(target.ablate_script),
            "--data_csv",
            str(target.data_csv),
            "--image_root",
            str(target.image_root),
            "--model",
            str(target.model),
            "--selected_heads",
            str(selection_path),
            "--out_json",
            str(out_path),
            "--device",
            target.device,
            "--dtype",
            target.dtype,
            "--max_examples",
            str(target.max_examples),
            "--max_image_side",
            str(target.max_image_side),
            "--trace_mode",
            target.trace_mode,
            "--position",
            target.position,
            "--mask_scope",
            target.current_mask_scope,
            "--keep_mode",
            target.keep_mode,
        ]
        if target.model_name:
            cmd.extend(["--model_name", target.model_name])
    _run_subprocess(cmd, cwd=target.project_root)


def _compute_metrics_for_path(target: TargetConfig, path: Path) -> dict[str, Any]:
    if target.family == "qwen":
        return compute_qwen_metrics(path)
    return compute_vqa_metrics(path)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate alternative threshold groups while preserving the current config as the reference best.")
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--limit", type=int, default=3, help="number of accepted alternatives to keep")
    ap.add_argument("--candidate-pool", type=int, default=8, help="number of random effect/base threshold values to sample before forming random threshold pairs")
    ap.add_argument("--run-ablation", action="store_true", help="actually run the existing ablation script for each candidate")
    args = ap.parse_args()

    target = TARGETS[args.target]
    out_dir = ROOT / target.name
    out_dir.mkdir(parents=True, exist_ok=True)

    current_metrics_payload = _current_metrics_for_target(target)
    current_metrics = current_metrics_payload["metrics"]
    _save_json(out_dir / "current_metrics.json", current_metrics_payload)

    if target.family == "qwen":
        candidates = _enumerate_qwen_threshold_candidates(target, args.candidate_pool)
    else:
        candidates = _enumerate_vqa_threshold_candidates(target, args.candidate_pool)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, 1):
        cand_dir = out_dir / f"candidate_{idx:02d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        selection_path = cand_dir / "selected_heads.json"
        if target.family == "qwen":
            _write_qwen_selection_json(target, selection_path, candidate)
            ablation_path = cand_dir / "ablate_heads_val.jsonl"
        else:
            _write_vqa_selection_json(target, selection_path, candidate)
            ablation_path = cand_dir / "ablation_val.json"

        status: dict[str, Any] = {
            "candidate_index": idx,
            "thresholds": candidate["thresholds"],
            "n_selected": candidate["n_selected"],
            "selection_path": str(selection_path),
            "ablation_path": str(ablation_path),
        }

        if args.run_ablation:
            _run_ablation_for_candidate(target, selection_path, ablation_path)

        if ablation_path.exists():
            metrics_payload = _compute_metrics_for_path(target, ablation_path)
            metrics = metrics_payload["metrics"]
            status["metrics"] = metrics
            status["metrics_path"] = str(cand_dir / "metrics.json")
            _save_json(cand_dir / "metrics.json", metrics_payload)

            if stronger_than(metrics, current_metrics):
                status["decision"] = "rejected_stronger_than_current"
                rejected.append(status)
            else:
                status["decision"] = "accepted"
                accepted.append(status)
        else:
            status["decision"] = "pending_ablation"
            rejected.append(status)

        _save_json(cand_dir / "status.json", status)
        if len(accepted) >= args.limit:
            break

    manifest = {
        "target": target.name,
        "position": target.position,
        "current_selection_path": str(target.current_selection_path),
        "current_ablation_path": str(target.current_ablation_path),
        "current_metrics": current_metrics,
        "accepted_candidates": accepted,
        "rejected_or_pending_candidates": rejected,
        "notes": {
            "comparison_rule": "A candidate is treated as stronger than current if it wins lexicographically on (lower CFR, lower C2W, higher W2C). Stronger candidates are not kept.",
            "metric_definition": "CFR = ablated follow-conflict rate. C2W = baseline conflict-context correct -> ablated wrong. W2C = baseline conflict-context wrong -> ablated correct. All denominators use N_all.",
        },
    }
    _save_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"saved_manifest": str(out_dir / "manifest.json"), "accepted": len(accepted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
