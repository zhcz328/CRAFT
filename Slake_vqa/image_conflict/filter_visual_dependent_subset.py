#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ORGAN_TERMS = (
    "small bowel",
    "femoral head",
    "parotid",
    "mandible",
    "kidney",
    "spleen",
    "liver",
    "heart",
    "colon",
    "brain",
    "lung",
    "ear",
    "eye",
)


def norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def parse_json_list(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return []


def question_organs(question: str) -> list[str]:
    text = norm_text(question)
    found: list[str] = []
    for term in ORGAN_TERMS:
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
            found.append(term)
    return found


def organ_alignment(row: dict[str, str]) -> bool:
    organs = question_organs(row.get("question", ""))
    if not organs:
        return False
    det_text = " ".join(parse_json_list(row.get("detection_labels", ""))).lower()
    tgt_text = " ".join(parse_json_list(row.get("ic_target_labels", ""))).lower()
    return any(term in det_text or term in tgt_text for term in organs)


def classify_row(row: dict[str, str]) -> tuple[bool, str]:
    content_type = str(row.get("content_type", "")).strip()
    question_type = str(row.get("ic_question_type", "")).strip()
    question = norm_text(row.get("question", ""))
    aligned = organ_alignment(row)
    has_targets = bool(parse_json_list(row.get("ic_target_labels", "")))

    if content_type in {"Modality", "Plane"}:
        return False, "exclude_global_type"

    if content_type == "Position":
        if question_type in {"target_location", "lung_part_location", "side_target_position"} and has_targets:
            return True, "position_targeted"
        return False, "exclude_global_type"

    if content_type == "Organ":
        if question_type in {"organ_presence", "organ_health"} and aligned:
            return True, "organ_aligned"
        return False, "organ_mismatch_or_global"

    if content_type == "Abnormality":
        if question_type in {"explicit_disease_presence", "tumor_attribute"}:
            return True, "explicit_abnormality"
        if "which organ is abnormal" in question and aligned:
            return True, "abnormal_organ_choice"
        return False, "abnormality_too_generic"

    if content_type in {"Color", "Shape", "Quantity"}:
        return True, "local_visual_type"

    if content_type == "Size":
        return (True, "size_aligned") if aligned else (False, "size_mismatch")

    return False, "other"


def build_default_paths(in_csv: Path) -> tuple[Path, Path]:
    stem = in_csv.stem
    parent = in_csv.parent
    return (
        parent / f"{stem}_visual_local.csv",
        parent / f"{stem}_visual_local_summary.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter image_conflict CSV rows down to questions that rely more on local visual evidence."
    )
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--summary_json", default="")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv, summary_json = build_default_paths(in_csv)
    if args.out_csv:
        out_csv = Path(args.out_csv)
    if args.summary_json:
        summary_json = Path(args.summary_json)

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    kept_rows: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    kept_content_types: Counter[str] = Counter()

    extra_fields = []
    if "visual_local_keep" not in fieldnames:
        extra_fields.append("visual_local_keep")
    if "visual_local_reason" not in fieldnames:
        extra_fields.append("visual_local_reason")

    for row in rows:
        keep, reason = classify_row(row)
        reason_counts[reason] += 1
        if not keep:
            continue
        new_row = dict(row)
        new_row["visual_local_keep"] = "True"
        new_row["visual_local_reason"] = reason
        kept_rows.append(new_row)
        kept_content_types[str(row.get("content_type", "")).strip()] += 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra_fields)
        writer.writeheader()
        writer.writerows(kept_rows)

    summary = {
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "n_input_rows": len(rows),
        "n_kept_rows": len(kept_rows),
        "keep_rate": (len(kept_rows) / len(rows)) if rows else 0.0,
        "reason_counts": dict(reason_counts),
        "kept_content_types": dict(kept_content_types),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_csv={out_csv}")
    print(f"saved_summary={summary_json}")
    print(f"n_input_rows={len(rows)}")
    print(f"n_kept_rows={len(kept_rows)}")
    print(f"keep_rate={summary['keep_rate']:.4f}")


if __name__ == "__main__":
    main()
