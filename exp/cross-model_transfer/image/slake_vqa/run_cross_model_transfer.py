#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "exp" / "cross-model_transfer" / "image" / "slake_vqa"
LATENTMAS_PYTHON = Path("/root/miniconda3/envs/latentmas/bin/python")


@dataclass(frozen=True)
class TransferJob:
    name: str
    target_model: str
    target_model_path: str
    script_path: str
    workdir: str
    data_csv: str
    image_root: str
    selected_heads: str
    out_json: str
    position: str = "image_conflict"
    trace_mode: str = "conflict"
    metrics: str = "follow_context"
    mask_scope: str = "ctx_only"
    keep_mode: str = "self"
    dtype: str = "bf16"
    device: str = "cuda"
    max_examples: int = 0
    max_image_side: int = 672
    seed: int = 0
    mask_scale: float = 2.0

    def command(self) -> list[str]:
        python_exe = str(LATENTMAS_PYTHON if LATENTMAS_PYTHON.exists() else Path(sys.executable))
        return [
            python_exe,
            self.script_path,
            "--data_csv",
            self.data_csv,
            "--image_root",
            self.image_root,
            "--model",
            self.target_model_path,
            "--model_name",
            self.target_model,
            "--selected_heads",
            self.selected_heads,
            "--out_json",
            self.out_json,
            "--device",
            self.device,
            "--dtype",
            self.dtype,
            "--max_examples",
            str(self.max_examples),
            "--max_image_side",
            str(self.max_image_side),
            "--seed",
            str(self.seed),
            "--trace_mode",
            self.trace_mode,
            "--position",
            self.position,
            "--metrics",
            self.metrics,
            "--mask_scope",
            self.mask_scope,
            "--keep_mode",
            self.keep_mode,
            "--mask_scale",
            str(self.mask_scale),
        ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_core_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    pred_nc = payload.get("pred_nc", {})
    pred_ctx = payload.get("pred_ctx", {})
    ctx_unknown = payload.get("ctx_unknown", {})
    return {
        "n_selected_heads": payload.get("n_selected_heads"),
        "n_samples": payload.get("n_samples"),
        "mean_hallucination_relief": payload.get("mean_hallucination_relief"),
        "mean_ic_follow_context_gain": payload.get("mean_ic_follow_context_gain"),
        "mean_nc_gold_margin_damage": payload.get("mean_nc_gold_margin_damage"),
        "pred_nc_base_acc": pred_nc.get("base_acc"),
        "pred_nc_ab_acc": pred_nc.get("ab_acc"),
        "pred_nc_acc_delta": pred_nc.get("acc_delta"),
        "pred_ctx_base_acc": pred_ctx.get("base_acc"),
        "pred_ctx_ab_acc": pred_ctx.get("ab_acc"),
        "pred_ctx_acc_delta": pred_ctx.get("acc_delta"),
        "ctx_unknown_base_rate": ctx_unknown.get("base_rate"),
        "ctx_unknown_ab_rate": ctx_unknown.get("ab_rate"),
        "ctx_unknown_rate_delta": ctx_unknown.get("rate_delta"),
        "follow_context_mean_abs_effect_reduction_fixed_nc": payload.get("per_metric", {})
        .get("follow_context", {})
        .get("mean_abs_effect_reduction_fixed_nc"),
        "follow_context_mean_abs_effect_reduction_mask_scope": payload.get("per_metric", {})
        .get("follow_context", {})
        .get("mean_abs_effect_reduction_mask_scope"),
        "follow_context_mean_abs_base_change": payload.get("per_metric", {})
        .get("follow_context", {})
        .get("mean_abs_base_change"),
        "follow_context_mean_abs_ctx_change": payload.get("per_metric", {})
        .get("follow_context", {})
        .get("mean_abs_ctx_change"),
    }


def write_summary(jobs: list[TransferJob]) -> Path:
    summary_rows = []
    for job in jobs:
        payload = load_json(Path(job.out_json))
        row = {
            "job": job.name,
            "target_model": job.target_model,
            "source_heads": job.selected_heads,
            "out_json": job.out_json,
            "mask_scope": job.mask_scope,
            "position": job.position,
            "metrics": job.metrics,
        }
        row.update(extract_core_metrics(payload))
        summary_rows.append(row)

    summary = {
        "experiment": "cross_model_transfer_image_slake_image_conflict",
        "jobs": [asdict(job) for job in jobs],
        "results": summary_rows,
    }
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_conflict_dir = ROOT / "Slake_vqa" / "image_conflict"
    script_path = str(image_conflict_dir / "ablate_head.py")

    jobs = [
        TransferJob(
            name="internvl_heads_on_hulumed_val_ctx_only",
            target_model="hulumed-4b",
            target_model_path="/root/autodl-tmp/Hulu-Med-4B",
            script_path=script_path,
            workdir=str(image_conflict_dir),
            data_csv=str(image_conflict_dir / "data" / "hulumed4b" / "slake_nc_correct_ic_ready_val.csv"),
            image_root=".",
            selected_heads=str(
                image_conflict_dir
                / "internvl35_4b"
                / "result_image_conflict_slake"
                / "selected_heads_core_layers.json"
            ),
            out_json=str(OUT_DIR / "internvl_heads_on_hulumed_val_ctx_only.json"),
        ),
        TransferJob(
            name="hulumed_heads_on_internvl_val_ctx_only",
            target_model="internvl3_5-4b",
            target_model_path="/root/autodl-tmp/InternVL3_5-4B",
            script_path=script_path,
            workdir=str(image_conflict_dir),
            data_csv=str(image_conflict_dir / "data" / "internvl35_4b" / "slake_nc_correct_ic_ready_val.csv"),
            image_root=".",
            selected_heads=str(
                image_conflict_dir
                / "hulumed4b"
                / "result_image_conflict_slake"
                / "selected_heads_core_layers.json"
            ),
            out_json=str(OUT_DIR / "hulumed_heads_on_internvl_val_ctx_only.json"),
        ),
    ]

    run_log = []
    for job in jobs:
        cmd = job.command()
        print(f"[run] {job.name}")
        print(" ".join(cmd))
        proc = subprocess.run(cmd, cwd=job.workdir, text=True, capture_output=True)
        log_entry = {
            "job": job.name,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": cmd,
            "python_executable": cmd[0],
        }
        run_log.append(log_entry)

        log_path = OUT_DIR / f"{job.name}.log.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)

        if proc.returncode != 0:
            fail_path = OUT_DIR / "run_log.json"
            with fail_path.open("w", encoding="utf-8") as f:
                json.dump(run_log, f, ensure_ascii=False, indent=2)
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)

    run_log_path = OUT_DIR / "run_log.json"
    with run_log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)

    summary_path = write_summary(jobs)
    print(f"[done] summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
