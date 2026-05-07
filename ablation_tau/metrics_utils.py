from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def _lexicographic_strength_tuple(metrics: dict[str, float]) -> tuple[float, float, float]:
    # Lower CFR/C2W is better; higher W2C is better.
    return (
        float(metrics["CFR"]),
        float(metrics["C2W"]),
        -float(metrics["W2C"]),
    )


def stronger_than(candidate: dict[str, float], reference: dict[str, float]) -> bool:
    return _lexicographic_strength_tuple(candidate) < _lexicographic_strength_tuple(reference)


def summarize_flip_metrics_from_pred_labels(
    rows: list[dict[str, Any]],
    base_field: str,
    ablated_field: str,
) -> dict[str, Any]:
    total = len(rows)
    cfr_count = 0
    c2w_count = 0
    w2c_count = 0

    for row in rows:
        base_pred = str(row.get(base_field, ""))
        ab_pred = str(row.get(ablated_field, ""))
        if ab_pred == "wrong":
            cfr_count += 1
        if base_pred == "gold" and ab_pred == "wrong":
            c2w_count += 1
        if base_pred == "wrong" and ab_pred == "gold":
            w2c_count += 1

    metrics = {
        "n_all": total,
        "counts": {
            "follow_conflict": cfr_count,
            "correct_to_wrong": c2w_count,
            "wrong_to_correct": w2c_count,
        },
        "CFR": _pct(cfr_count, total),
        "C2W": _pct(c2w_count, total),
        "W2C": _pct(w2c_count, total),
    }
    metrics["strength_tuple"] = list(_lexicographic_strength_tuple(metrics))
    return metrics


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            rows.append(obj)
    return rows


def compute_vqa_metrics(ablation_json: str | Path) -> dict[str, Any]:
    payload = load_json(ablation_json)
    rows = payload.get("records", [])
    metrics = summarize_flip_metrics_from_pred_labels(
        rows=rows,
        base_field="base_ctx_pred",
        ablated_field="ab_ctx_pred",
    )
    return {
        "format": "vqa_ablation_json",
        "path": str(ablation_json),
        "config": payload.get("config", {}),
        "n_selected_heads": payload.get("n_selected_heads"),
        "metrics": metrics,
    }


def _normalize_qwen_ctx_label(row: dict[str, Any], field: str) -> str:
    pred = str(row.get(field, {}).get("pred", "")).strip().lower()
    gold = str(row.get("gold", "")).strip().lower()
    return "gold" if pred == gold else "wrong"


def _qwen_example_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["pair_id"]), str(row["side"]), str(row["position"]))


def compute_qwen_metrics(ablation_jsonl: str | Path) -> dict[str, Any]:
    rows = read_jsonl(ablation_jsonl)
    root_dir = Path(ablation_jsonl).resolve().parents[2]
    baseline_rows_path = root_dir / "result_all_positions" / "conflict_positions_val.jsonl"
    if not baseline_rows_path.exists():
        fallback = root_dir / "result_all_positions" / "conflict_positions.jsonl"
        baseline_rows_path = fallback

    normalized_rows: list[dict[str, str]] = []
    has_inline_before_after = bool(rows) and all(
        ("conflict_before" in row and "conflict_after" in row)
        for row in rows
    )

    if has_inline_before_after:
        normalized_rows = [
            {
                "base_ctx_pred": _normalize_qwen_ctx_label(row, "conflict_before"),
                "ab_ctx_pred": _normalize_qwen_ctx_label(row, "conflict_after"),
            }
            for row in rows
        ]
    elif baseline_rows_path.exists():
        baseline_rows = read_jsonl(baseline_rows_path)
        baseline_map = {_qwen_example_key(row): row for row in baseline_rows}
        for row in rows:
            key = _qwen_example_key(row)
            baseline_row = baseline_map.get(key)
            if baseline_row is None:
                continue
            base_ctx_pred = _normalize_qwen_ctx_label(baseline_row, "conflict")
            ab_ctx_pred = _normalize_qwen_ctx_label(row, "conflict")
            normalized_rows.append(
                {
                    "base_ctx_pred": base_ctx_pred,
                    "ab_ctx_pred": ab_ctx_pred,
                }
            )
    else:
        normalized_rows = [
            {
                "base_ctx_pred": _normalize_qwen_ctx_label(row, "conflict_before"),
                "ab_ctx_pred": _normalize_qwen_ctx_label(row, "conflict_after"),
            }
            for row in rows
        ]

    metrics = summarize_flip_metrics_from_pred_labels(
        rows=normalized_rows,
        base_field="base_ctx_pred",
        ablated_field="ab_ctx_pred",
    )
    return {
        "format": "qwen_ablation_jsonl",
        "path": str(ablation_jsonl),
        "n_rows": len(rows),
        "baseline_rows_path": str(baseline_rows_path) if baseline_rows_path.exists() else None,
        "metrics": metrics,
    }
