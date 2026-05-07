from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm_label(value: Any) -> str:
    return str(value or "").strip().lower()


def summarize_image_flip_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    c2u_count = 0
    u2o_count = 0
    ur_count = 0

    for row in rows:
        base_pred = _norm_label(row.get("base_ctx_pred"))
        ab_pred = _norm_label(row.get("ab_ctx_pred"))

        if base_pred == "gold" and ab_pred == "unknown":
            c2u_count += 1
        if base_pred == "unknown" and ab_pred == "other":
            u2o_count += 1
        if ab_pred == "unknown":
            ur_count += 1

    metrics = {
        "n_all": total,
        "counts": {
            "correct_to_unknown": c2u_count,
            "unknown_to_other": u2o_count,
            "unknown_rate": ur_count,
        },
        "C2U": _pct(c2u_count, total),
        "U2O": _pct(u2o_count, total),
        "UR": _pct(ur_count, total),
    }
    metrics["rule_tuple"] = [float(metrics["C2U"]), float(metrics["U2O"]), float(metrics["UR"])]
    return metrics


def compute_image_metrics(ablation_json: str | Path) -> dict[str, Any]:
    payload = load_json(ablation_json)
    rows = payload.get("records", [])
    metrics = summarize_image_flip_metrics(rows)
    return {
        "format": "image_conflict_ablation_json",
        "path": str(ablation_json),
        "config": payload.get("config", {}),
        "n_selected_heads": payload.get("n_selected_heads"),
        "sample_filter": payload.get("sample_filter"),
        "metrics": metrics,
    }


def compare_candidate_to_reference(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> dict[str, bool]:
    return {
        "C2U_lower": float(candidate["C2U"]) < float(reference["C2U"]),
        "U2O_higher": float(candidate["U2O"]) > float(reference["U2O"]),
        "UR_higher": float(candidate["UR"]) > float(reference["UR"]),
    }


def qualifies_by_two_of_three(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> bool:
    flags = compare_candidate_to_reference(candidate, reference)
    return sum(1 for ok in flags.values() if ok) >= 2
