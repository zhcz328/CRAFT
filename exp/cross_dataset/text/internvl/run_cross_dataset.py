#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "exp" / "cross_dataset" / "text" / "internvl"
LATENTMAS_PYTHON = Path("/root/miniconda3/envs/latentmas/bin/python")


@dataclass(frozen=True)
class TransferJob:
    name: str
    source_dataset: str
    target_dataset: str
    split: str
    workdir: str
    script_path: str
    data_csv: str
    image_root: str
    target_model: str
    target_model_path: str
    selected_heads: str
    out_json: str
    position: str = "before_question"
    trace_mode: str = "conflict"
    metrics: str = "follow_conflict"
    mask_scope: str = "all"
    keep_mode: str = "self"
    dtype: str = "bf16"
    device: str = "cuda"
    max_examples: int = 0
    max_image_side: int = 672
    seed: int = 0

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
            "--dtype",
            self.dtype,
            "--device",
            self.device,
            "--max_examples",
            str(self.max_examples),
            "--max_image_side",
            str(self.max_image_side),
            "--seed",
            str(self.seed),
            "--out_json",
            self.out_json,
        ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_core_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    pred_nc = payload.get("pred_nc", {})
    pred_ctx = payload.get("pred_ctx", {})
    per_metric = payload.get("per_metric", {}).get("follow_conflict", {})
    return {
        "n_selected_heads": payload.get("n_selected_heads"),
        "n_samples": payload.get("n_samples"),
        "mean_abs_effect_reduction_fixed_nc": payload.get("mean_abs_effect_reduction_fixed_nc"),
        "mean_abs_effect_reduction_mask_scope": payload.get("mean_abs_effect_reduction_mask_scope"),
        "mean_abs_base_change": payload.get("mean_abs_base_change"),
        "mean_abs_ctx_change": payload.get("mean_abs_ctx_change"),
        "pred_nc_base_acc": pred_nc.get("base_acc"),
        "pred_nc_ab_acc": pred_nc.get("ab_acc"),
        "pred_nc_acc_delta": pred_nc.get("acc_delta"),
        "pred_ctx_base_acc": pred_ctx.get("base_acc"),
        "pred_ctx_ab_acc": pred_ctx.get("ab_acc"),
        "pred_ctx_acc_delta": pred_ctx.get("acc_delta"),
        "follow_conflict_metric_fixed_nc": per_metric.get("mean_abs_effect_reduction_fixed_nc"),
        "follow_conflict_metric_mask_scope": per_metric.get("mean_abs_effect_reduction_mask_scope"),
        "follow_conflict_metric_base_change": per_metric.get("mean_abs_base_change"),
        "follow_conflict_metric_ctx_change": per_metric.get("mean_abs_ctx_change"),
    }


def write_summary(jobs: list[TransferJob]) -> Path:
    summary_rows = []
    for job in jobs:
        payload = load_json(Path(job.out_json))
        row = {
            "job": job.name,
            "source_dataset": job.source_dataset,
            "target_dataset": job.target_dataset,
            "split": job.split,
            "source_heads": job.selected_heads,
            "out_json": job.out_json,
            "mask_scope": job.mask_scope,
            "position": job.position,
            "metrics": job.metrics,
        }
        row.update(extract_core_metrics(payload))
        summary_rows.append(row)

    summary = {
        "experiment": "cross_dataset_text_internvl_before_question",
        "jobs": [asdict(job) for job in jobs],
        "results": summary_rows,
    }
    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary_path


def build_jobs(splits: list[str]) -> list[TransferJob]:
    internvl_model_name = "internvl3_5-4b"
    internvl_model_path = "/root/autodl-tmp/InternVL3_5-4B"
    slake_project = ROOT / "Slake_vqa" / "text_conflict"
    vqarad_project = ROOT / "VQA_RAD" / "text_conflict"

    jobs: list[TransferJob] = []
    for split in splits:
        jobs.append(
            TransferJob(
                name=f"slake_heads_on_vqarad_{split}",
                source_dataset="slake_vqa",
                target_dataset="vqa_rad",
                split=split,
                workdir=str(vqarad_project),
                script_path=str(vqarad_project / "ablate_head.py"),
                data_csv=str(vqarad_project / "data" / "internvl35_4b" / f"vqa_rad_nc_cc_both_correct_{split}.csv"),
                image_root=".",
                target_model=internvl_model_name,
                target_model_path=internvl_model_path,
                selected_heads=str(
                    ROOT
                    / "Slake_vqa"
                    / "text_conflict"
                    / "internvl35_4b"
                    / "result_before_question_slake"
                    / "selected_heads_merged_unique_layers.json"
                ),
                out_json=str(OUT_DIR / f"slake_heads_on_vqarad_all_{split}.json"),
            )
        )
        jobs.append(
            TransferJob(
                name=f"vqarad_heads_on_slake_{split}",
                source_dataset="vqa_rad",
                target_dataset="slake_vqa",
                split=split,
                workdir=str(slake_project),
                script_path=str(slake_project / "ablate_head.py"),
                data_csv=str(slake_project / "data" / "internvl35_4b" / f"slake_nc_cc_both_correct_{split}.csv"),
                image_root=".",
                target_model=internvl_model_name,
                target_model_path=internvl_model_path,
                selected_heads=str(
                    ROOT
                    / "VQA_RAD"
                    / "text_conflict"
                    / "internvl35_4b"
                    / "result_before_question_vqarad"
                    / "selected_heads_merged_unique_layers.json"
                ),
                out_json=str(OUT_DIR / f"vqarad_heads_on_slake_all_{split}.json"),
            )
        )
    return jobs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val"])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.splits)

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
