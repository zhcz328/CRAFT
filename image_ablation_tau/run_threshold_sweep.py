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

from metrics_utils import (
    compare_candidate_to_reference,
    compute_image_metrics,
    load_json,
    qualifies_by_two_of_three,
)


ROOT = Path("/root/logit_lens/image_ablation_tau")
PROJECT_ROOT = Path("/root/logit_lens/Slake_vqa/image_conflict")
RNG_SEED = 20260504


@dataclass
class TargetConfig:
    name: str
    project_root: Path
    select_script: Path
    ablate_script: Path
    current_selection_path: Path
    current_ablation_path: Path
    scan_input_json: Path
    model: str
    model_name: str
    data_csv: Path
    image_root: Path
    position: str = "image_conflict"
    trace_mode: str = "conflict"
    keep_mode: str = "self"
    mask_scope: str = "ctx_only"
    mask_scale: float = 2.0
    dtype: str = "bf16"
    device: str = "auto"
    max_examples: int = 0
    max_image_side: int = 672
    select_source: str = "results"
    select_topk: int = 50


TARGETS: dict[str, TargetConfig] = {
    "hulumed4b_image_conflict_val": TargetConfig(
        name="hulumed4b_image_conflict_val",
        project_root=PROJECT_ROOT,
        select_script=PROJECT_ROOT / "select_heads.py",
        ablate_script=PROJECT_ROOT / "ablate_head.py",
        current_selection_path=PROJECT_ROOT / "hulumed4b/result_image_conflict_slake/selected_heads_core_layers.json",
        current_ablation_path=PROJECT_ROOT / "hulumed4b/result_image_conflict_slake/ablate/ablate_selected_heads_ctx_only_val.json",
        scan_input_json=PROJECT_ROOT / "hulumed4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
        model="/root/autodl-tmp/Hulu-Med-4B",
        model_name="hulumed-4b",
        data_csv=PROJECT_ROOT / "data/hulumed4b/slake_nc_correct_ic_ready_val.csv",
        image_root=PROJECT_ROOT,
        device="cuda",
    ),
    "internvl35_4b_image_conflict_val": TargetConfig(
        name="internvl35_4b_image_conflict_val",
        project_root=PROJECT_ROOT,
        select_script=PROJECT_ROOT / "select_heads.py",
        ablate_script=PROJECT_ROOT / "ablate_head.py",
        current_selection_path=PROJECT_ROOT / "internvl35_4b/result_image_conflict_slake/selected_heads_core_layers.json",
        current_ablation_path=PROJECT_ROOT / "internvl35_4b/result_image_conflict_slake/ablate/ablate_selected_heads_ctx_only_val.json",
        scan_input_json=PROJECT_ROOT / "internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
        model="/root/autodl-tmp/InternVL3_5-4B",
        model_name="internvl3_5-4b",
        data_csv=PROJECT_ROOT / "data/internvl35_4b/slake_nc_correct_ic_ready_val.csv",
        image_root=PROJECT_ROOT,
        device="cuda",
    ),
}


ALIASES = {
    "image_conflict_hulumed4b_val": "hulumed4b_image_conflict_val",
    "image_conflict_internvl35_4b_val": "internvl35_4b_image_conflict_val",
}


def _resolve_target(name: str) -> TargetConfig:
    canonical = ALIASES.get(name, name)
    return TARGETS[canonical]


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _candidate_signature(selected_rows: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(row["layer"]), int(row["head"])) for row in selected_rows))


def _shuffle_in_place(rows: list[dict[str, Any]], seed: int) -> None:
    random.Random(seed).shuffle(rows)


def _load_current_selection_payload(target: TargetConfig) -> dict[str, Any]:
    return load_json(target.current_selection_path)


def _current_metrics_for_target(target: TargetConfig) -> dict[str, Any]:
    return compute_image_metrics(target.current_ablation_path)


def _enumerate_threshold_candidates(target: TargetConfig, limit_pool: int) -> list[dict[str, Any]]:
    mod = _load_module(f"select_{target.name}", target.select_script)
    current_payload = _load_current_selection_payload(target)
    source = str(current_payload.get("source") or target.select_source)
    topk = current_payload.get("topk")
    if topk is None:
        topk = target.select_topk
    mode = str(current_payload.get("mode") or "custom")
    thresholds = current_payload.get("thresholds", {})

    _obj, rows = mod.load_scan(str(target.scan_input_json), source=source, topk=topk)
    current_sig = _candidate_signature(current_payload.get("selected", []))

    eff_values = sorted({round(float(r["mean_abs_effect_reduction"]), 12) for r in rows})
    base_values = sorted({round(float(r["mean_abs_base_change"]), 12) for r in rows})

    eff_candidates = eff_values[:]
    base_candidates = base_values[:]
    rng = random.Random(RNG_SEED)
    rng.shuffle(eff_candidates)
    rng.shuffle(base_candidates)
    if limit_pool > 0:
        keep_n = max(1, limit_pool)
        eff_candidates = eff_candidates[:keep_n]
        base_candidates = base_candidates[:keep_n]

    candidates: list[dict[str, Any]] = []
    seen_sigs: set[tuple[tuple[int, int], ...]] = {current_sig}

    base_min = thresholds.get("base_min")
    for eff_min, base_max in itertools.product(eff_candidates, base_candidates):
        selected = mod.filter_heads(
            rows,
            mode,
            eff_min=eff_min,
            base_max=base_max,
            base_min=base_min,
        )
        selected = mod.attach_scores(selected)
        selected = mod.sort_selected(selected, mode)
        if not selected:
            continue
        sig = _candidate_signature(selected)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        candidates.append(
            {
                "mode": mode,
                "source": source,
                "topk": topk,
                "thresholds": {
                    "eff_min": eff_min,
                    "base_max": base_max,
                    "base_min": base_min,
                },
                "n_selected": len(selected),
                "selected": selected,
                "signature": sig,
            }
        )

    _shuffle_in_place(candidates, seed=RNG_SEED + 17)
    return candidates


def _write_selection_json(target: TargetConfig, out_path: Path, candidate: dict[str, Any]) -> None:
    current_payload = _load_current_selection_payload(target)
    payload = {
        "mode": candidate["mode"],
        "thresholds": candidate["thresholds"],
        "input_json": str(target.scan_input_json),
        "source": candidate["source"],
        "topk": candidate["topk"],
        "meta": current_payload.get("meta", {}),
        "n_candidates": current_payload.get("n_candidates"),
        "n_selected": candidate["n_selected"],
        "selected": candidate["selected"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_subprocess(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _run_ablation_for_candidate(target: TargetConfig, selection_path: Path, out_path: Path, mode: str) -> None:
    del mode
    cmd = [
        sys.executable,
        str(target.ablate_script),
        "--data_csv",
        str(target.data_csv),
        "--image_root",
        str(target.image_root),
        "--model",
        str(target.model),
        "--model_name",
        target.model_name,
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
        "--mask_scale",
        str(target.mask_scale),
        "--metrics",
        "follow_context",
        "--mask_scope",
        target.mask_scope,
        "--keep_mode",
        target.keep_mode,
    ]
    _run_subprocess(cmd, cwd=target.project_root)


def main() -> None:
    all_choices = sorted(set(TARGETS) | set(ALIASES))
    ap = argparse.ArgumentParser(
        description="Generate alternative threshold groups for image_conflict core-layer head selections."
    )
    ap.add_argument("--target", choices=all_choices, required=True)
    ap.add_argument("--limit", type=int, default=3, help="number of accepted alternatives to keep")
    ap.add_argument(
        "--candidate-pool",
        type=int,
        default=8,
        help="number of random effect/base threshold values to sample before forming pairs",
    )
    ap.add_argument("--run-ablation", action="store_true", help="actually run ablation for each candidate")
    args = ap.parse_args()

    target = _resolve_target(args.target)
    out_dir = ROOT / target.name
    out_dir.mkdir(parents=True, exist_ok=True)

    current_metrics_payload = _current_metrics_for_target(target)
    current_metrics = current_metrics_payload["metrics"]
    _save_json(out_dir / "current_metrics.json", current_metrics_payload)

    candidates = _enumerate_threshold_candidates(target, args.candidate_pool)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, 1):
        cand_dir = out_dir / f"candidate_{idx:02d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        selection_path = cand_dir / "selected_heads.json"
        ablation_path = cand_dir / "ablation_val.json"
        _write_selection_json(target, selection_path, candidate)

        status: dict[str, Any] = {
            "candidate_index": idx,
            "mode": candidate["mode"],
            "thresholds": candidate["thresholds"],
            "n_selected": candidate["n_selected"],
            "selection_path": str(selection_path),
            "ablation_path": str(ablation_path),
        }

        if args.run_ablation:
            _run_ablation_for_candidate(target, selection_path, ablation_path, candidate["mode"])

        if ablation_path.exists():
            metrics_payload = compute_image_metrics(ablation_path)
            metrics = metrics_payload["metrics"]
            status["metrics"] = metrics
            status["metrics_path"] = str(cand_dir / "metrics.json")
            status["comparison_flags"] = compare_candidate_to_reference(metrics, current_metrics)
            status["matched_rule_count"] = sum(1 for ok in status["comparison_flags"].values() if ok)
            _save_json(cand_dir / "metrics.json", metrics_payload)

            if qualifies_by_two_of_three(metrics, current_metrics):
                status["decision"] = "accepted"
                accepted.append(status)
            else:
                status["decision"] = "rejected_rule_not_met"
                rejected.append(status)
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
            "comparison_rule": "A candidate is accepted if at least two of these are true relative to current: lower C2U, higher U2O, higher UR.",
            "metric_definition": "C2U = base_ctx_pred gold -> ab_ctx_pred unknown. U2O = base_ctx_pred unknown -> ab_ctx_pred other. UR = ab_ctx_pred unknown rate. All denominators use N_all.",
            "selection_source": "Candidates are generated from head_scan_core_layers.json and written in selected_heads_core_layers-compatible JSON format.",
        },
    }
    _save_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"saved_manifest": str(out_dir / "manifest.json"), "accepted": len(accepted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
