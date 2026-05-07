from __future__ import annotations

import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path("/root/logit_lens")
PATCH_VALIDATE_ROOT = ROOT / "PIC" / "patch_validate"
RESULT_ROOT = PATCH_VALIDATE_ROOT / "result"
TRACE_ROOT = RESULT_ROOT / "traces"
FIGURE_ROOT = RESULT_ROOT / "figures"
METRIC_ROOT = RESULT_ROOT / "metrics"
MANIFEST_ROOT = RESULT_ROOT / "manifests"
LOG_ROOT = RESULT_ROOT / "logs"

PATCH_VARIANTS: List[str] = ["k2", "k4", "k8", "k16", "k32", "all_token"]
LOCAL_K_VARIANTS: List[str] = ["k2", "k4", "k8", "k16", "k32"]


def ensure_result_dirs() -> None:
    for path in [RESULT_ROOT, TRACE_ROOT, FIGURE_ROOT, METRIC_ROOT, MANIFEST_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def shell_join(parts: Sequence[str]) -> str:
    quoted: List[str] = []
    for part in parts:
        if part == "$PYTHON_BIN":
            quoted.append(part)
        else:
            quoted.append(shlex.quote(str(part)))
    return " ".join(quoted)


def variant_to_patch_params(variant: str, default_local_k: int) -> Tuple[int, bool]:
    if variant == "all_token":
        return default_local_k, True
    if not variant.startswith("k"):
        raise ValueError(f"Unknown patch variant: {variant}")
    return int(variant[1:]), False


def variant_sort_key(variant: str) -> Tuple[int, int]:
    if variant == "all_token":
        return (1, 10**9)
    if variant.startswith("k"):
        return (0, int(variant[1:]))
    return (2, 10**9)


@dataclass(frozen=True)
class TraceSpec:
    model_key: str
    display_name: str
    modality: str
    setting_label: str
    project_root: Path
    script_name: str
    runner_kind: str
    position: str
    default_local_k: int
    curve_metric: Optional[str]
    model_path: str
    dtype: str
    compare_variant: str
    pairs_path: Optional[Path] = None
    data_csv: Optional[Path] = None
    image_root: Optional[Path] = None
    model_name: str = ""
    trace_mode: str = "conflict"
    max_pairs: int = 120
    limit: int = 0

    def output_dir(self, variant: str) -> Path:
        return TRACE_ROOT / self.model_key / self.position / variant

    def output_paths(self, variant: str) -> Dict[str, Path]:
        out_dir = self.output_dir(variant)
        if self.runner_kind == "conflictmedqa_text":
            return {
                "trace_json": out_dir / "layer_trace.json",
                "plot_primary": out_dir / "layer_trace_scores.png",
                "scan_plan_json": out_dir / "scan_plan.json",
            }
        if self.runner_kind == "multimodal_fastcache":
            return {
                "trace_json": out_dir / "trace_conflict.json",
                "plot_prefix": out_dir / "trace_conflict",
                "scan_plan_json": out_dir / "trace_conflict_scan_plan.json",
            }
        raise ValueError(f"Unsupported runner_kind: {self.runner_kind}")

    def command(
        self,
        variant: str,
        python_bin: str = "$PYTHON_BIN",
        device_override: Optional[str] = None,
    ) -> str:
        patch_k, patch_all_tokens = variant_to_patch_params(variant, self.default_local_k)
        outputs = self.output_paths(variant)
        script_path = self.project_root / self.script_name

        if self.runner_kind == "conflictmedqa_text":
            if self.pairs_path is None:
                raise ValueError(f"{self.model_key} is missing pairs_path")
            args = [
                python_bin,
                str(script_path.name),
                "--pairs",
                str(self.pairs_path),
                "--model",
                self.model_path,
                "--position",
                self.position,
                "--patch_k",
                str(patch_k),
                "--max_pairs",
                str(self.max_pairs),
                "--dtype",
                self.dtype,
                "--device",
                device_override or "auto",
                "--out",
                str(outputs["trace_json"]),
                "--plot_out",
                str(outputs["plot_primary"]),
                "--plan_out",
                str(outputs["scan_plan_json"]),
            ]
            if patch_all_tokens:
                args.append("--patch_all_token")
            return f"cd {shlex.quote(str(self.project_root))} && {shell_join(args)}"

        if self.runner_kind == "multimodal_fastcache":
            if self.data_csv is None or self.image_root is None:
                raise ValueError(f"{self.model_key} is missing data_csv/image_root")
            args = [
                python_bin,
                str(script_path.name),
                "--data_csv",
                str(self.data_csv),
                "--image_root",
                str(self.image_root),
                "--model",
                self.model_path,
                "--device",
                device_override or "auto",
                "--dtype",
                self.dtype,
                "--trace_mode",
                self.trace_mode,
                "--position",
                self.position,
                "--patch_k",
                str(patch_k),
                "--limit",
                str(self.limit),
                "--metrics",
                "follow_conflict",
                "--out",
                str(outputs["trace_json"]),
                "--plot_prefix",
                str(outputs["plot_prefix"]),
                "--scan_plan_out",
                str(outputs["scan_plan_json"]),
                "--scan_plan_metric",
                "follow_conflict",
            ]
            if self.model_name:
                args.extend(["--model_name", self.model_name])
            if patch_all_tokens:
                args.append("--patch_all_tokens")
            return f"cd {shlex.quote(str(self.project_root))} && {shell_join(args)}"

        raise ValueError(f"Unsupported runner_kind: {self.runner_kind}")

    def to_manifest_entry(self, variant: str, device_override: Optional[str] = None) -> Dict[str, Any]:
        patch_k, patch_all_tokens = variant_to_patch_params(variant, self.default_local_k)
        outputs = self.output_paths(variant)
        payload = {
            "model_key": self.model_key,
            "display_name": self.display_name,
            "modality": self.modality,
            "setting_label": self.setting_label,
            "project_root": str(self.project_root),
            "script_name": self.script_name,
            "runner_kind": self.runner_kind,
            "position": self.position,
            "variant": variant,
            "patch_k": patch_k,
            "patch_all_token": patch_all_tokens,
            "device": device_override or "auto",
            "default_local_k": self.default_local_k,
            "compare_variant": self.compare_variant,
            "curve_metric": self.curve_metric,
            "trace_json": str(outputs["trace_json"]),
            "scan_plan_json": str(outputs["scan_plan_json"]),
            "command": self.command(variant, device_override=device_override),
        }
        plot_prefix = outputs.get("plot_prefix")
        if plot_prefix is not None:
            payload["plot_prefix"] = str(plot_prefix)
        else:
            payload["plot_primary"] = str(outputs["plot_primary"])
        return payload


def get_trace_specs() -> List[TraceSpec]:
    return [
        TraceSpec(
            model_key="qwen3_4b",
            display_name="Qwen3-4B",
            modality="text",
            setting_label="Text-only setting",
            project_root=ROOT / "conflictmedqa" / "Qwen3-4B_exp",
            script_name="layer_trace.py",
            runner_kind="conflictmedqa_text",
            position="before_question",
            default_local_k=8,
            curve_metric=None,
            model_path="/root/autodl-tmp/qwen3-4B",
            dtype="bfloat16",
            compare_variant="k8",
            pairs_path=ROOT
            / "conflictmedqa"
            / "Qwen3-4B_exp"
            / "data"
            / "kept_pairs_a12_b10_all_train.jsonl",
        ),
        TraceSpec(
            model_key="llama32_3b",
            display_name="Llama3.2-3B",
            modality="text",
            setting_label="Text-only setting",
            project_root=ROOT / "conflictmedqa" / "Qwen3-4B_exp",
            script_name="layer_trace.py",
            runner_kind="conflictmedqa_text",
            position="before_question",
            default_local_k=8,
            curve_metric=None,
            model_path="/root/autodl-tmp/Llama-3.2-3B-Instruct",
            dtype="bfloat16",
            compare_variant="k8",
            pairs_path=ROOT
            / "conflictmedqa"
            / "Qwen3-4B_exp"
            / "llama32_3b"
            / "data"
            / "kept_pairs_a12_b10_all_train.jsonl",
        ),
        TraceSpec(
            model_key="hulumed4b",
            display_name="Hulu-Med-4B",
            modality="multimodal",
            setting_label="Multimodal setting",
            project_root=ROOT / "VQA_RAD" / "Hulu-med",
            script_name="layer_trace_vqarad_mm_current_fixed_fastcache.py",
            runner_kind="multimodal_fastcache",
            position="before_question",
            default_local_k=16,
            curve_metric="follow_conflict",
            model_path="/root/autodl-tmp/Hulu-Med-4B",
            dtype="fp16",
            compare_variant="k16",
            data_csv=ROOT / "VQA_RAD" / "Hulu-med" / "data" / "nc_cc_both_correct_rerun_tmp_train.csv",
            image_root=Path("."),
        ),
        TraceSpec(
            model_key="internvl35_4b",
            display_name="InternVL3.5-4B",
            modality="multimodal",
            setting_label="Multimodal setting",
            project_root=ROOT / "VQA_RAD" / "text_conflict",
            script_name="layer_trace_vqarad_mm_current_fixed_fastcache.py",
            runner_kind="multimodal_fastcache",
            position="before_question",
            default_local_k=16,
            curve_metric="follow_conflict",
            model_path="/root/autodl-tmp/InternVL3_5-4B",
            model_name="InternVL3_5-4B",
            dtype="fp16",
            compare_variant="k16",
            data_csv=ROOT
            / "VQA_RAD"
            / "text_conflict"
            / "data"
            / "internvl35_4b"
            / "vqa_rad_nc_cc_both_correct_train.csv",
            image_root=Path("."),
        ),
    ]


def get_trace_spec_map() -> Dict[str, TraceSpec]:
    return {spec.model_key: spec for spec in get_trace_specs()}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_curve(payload: Dict[str, Any], metric: Optional[str]) -> List[float]:
    if isinstance(payload.get("layer_score_mean"), list):
        values = payload["layer_score_mean"]
    elif isinstance(payload.get("layer_score_mean"), dict):
        curves = payload["layer_score_mean"]
        metric_name = metric or next(iter(curves))
        values = curves[metric_name]
    elif isinstance(payload.get("layer_scores"), dict):
        curves = payload["layer_scores"]
        metric_name = metric or next(iter(curves))
        values = curves[metric_name]
    else:
        raise KeyError("No supported layer score field found in trace payload.")

    out: List[float] = []
    for value in values:
        scalar = float(value)
        out.append(scalar if math.isfinite(scalar) else float("nan"))
    return out


def load_curve(trace_json: Path, metric: Optional[str]) -> List[float]:
    return extract_curve(load_json(trace_json), metric=metric)


def normalize_curve_minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)

    arr = arr.copy()
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))
    span = max_val - min_val
    if span <= 1e-12:
        arr[finite] = 0.0
        arr[~finite] = 0.0
        return arr

    arr = (arr - min_val) / span
    arr[~finite] = 0.0
    return arr


def compute_smoothness(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def compute_consensus_curve(curves: Sequence[Sequence[float]]) -> np.ndarray:
    normalized = [normalize_curve_minmax(curve) for curve in curves]
    if not normalized:
        return np.zeros(0, dtype=float)
    stacked = np.stack(normalized, axis=0)
    return np.median(stacked, axis=0)


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray(list(a), dtype=float)
    xb = np.asarray(list(b), dtype=float)
    if xa.size != xb.size or xa.size == 0:
        return 0.0
    xa = xa - float(np.mean(xa))
    xb = xb - float(np.mean(xb))
    denom = float(np.linalg.norm(xa) * np.linalg.norm(xb))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(xa, xb) / denom)


def compute_robustness(
    values: Sequence[float],
    consensus_curve: Sequence[float],
    method: str = "consensus_l1",
) -> float:
    normalized = normalize_curve_minmax(values)
    consensus = np.asarray(list(consensus_curve), dtype=float)

    if normalized.size != consensus.size:
        raise ValueError(
            f"Curve length mismatch for robustness: {normalized.size} vs {consensus.size}"
        )

    if method == "consensus_l1":
        return float(1.0 - np.mean(np.abs(normalized - consensus)))
    if method == "pearson_to_consensus":
        return pearson_corr(normalized, consensus)
    raise ValueError(f"Unsupported robustness method: {method}")


def write_markdown_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
