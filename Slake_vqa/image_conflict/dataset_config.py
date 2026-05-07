from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SLAKE_DATASET = "slake_vqa"
HEAL_DATASET = "heal-medvqa"


def normalize_dataset_name(raw: str = "") -> str:
    token = str(raw or SLAKE_DATASET).strip().lower()
    if token in {"slake", "slake_vqa", "slake-vqa"}:
        return SLAKE_DATASET
    if token in {"heal", "heal_medvqa", "heal-medvqa", "healmedvqa"}:
        return HEAL_DATASET
    return SLAKE_DATASET


def dataset_save_root(project_root: str | Path, dataset: str) -> Path:
    key = normalize_dataset_name(dataset)
    if key == HEAL_DATASET:
        return Path("/root/logit_lens/heal-medvqa")
    return Path(project_root)


def has_inline_mask_data(row: Mapping[str, Any]) -> bool:
    raw_h = str(row.get("mask_h", "") or "").strip()
    raw_w = str(row.get("mask_w", "") or "").strip()
    if not raw_h or not raw_w:
        return False
    try:
        return int(raw_h) > 0 and int(raw_w) > 0 and "mask_rle" in row
    except Exception:
        return False


def extract_inline_mask_data(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not has_inline_mask_data(row):
        return None
    raw_rle = row.get("mask_rle", [])
    if isinstance(raw_rle, str):
        text = raw_rle.strip()
        if not text:
            rle = []
        else:
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            if not isinstance(parsed, list):
                return None
            rle = [int(v) for v in parsed]
    elif isinstance(raw_rle, list):
        rle = [int(v) for v in raw_rle]
    else:
        return None
    try:
        mask_h = int(str(row.get("mask_h", "")).strip())
        mask_w = int(str(row.get("mask_w", "")).strip())
    except Exception:
        return None
    return {
        "mask_rle": rle,
        "mask_height": mask_h,
        "mask_width": mask_w,
    }
