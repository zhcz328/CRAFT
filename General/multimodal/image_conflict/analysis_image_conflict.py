from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from mask_utils import UNKNOWN_ANSWER, build_masked_image


SYSTEM_PROMPT = "You are a helpful visual question answering assistant."
BASE_RULE = (
    "Answer the question using the image and your general world knowledge.\n"
    "If the image does not contain enough information to answer the question, output ONLY unknown.\n"
    "Output ONLY the final answer.\n\n"
)


def norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def build_prompt(question: str) -> str:
    return BASE_RULE + f"Question: {str(question).strip()}\nAnswer:"


def build_nc_prompt(question: str) -> str:
    return build_prompt(question)


def build_ic_prompt(question: str) -> str:
    return build_prompt(question)


def build_path_candidates(raw: Any, image_root: str) -> list[Path]:
    text = str(raw or "").strip()
    if not text:
        return []

    root = Path(image_root)
    path = Path(text)
    candidates = [path, root / path, root / path.name]
    parts = list(path.parts)
    for idx in range(1, len(parts)):
        candidates.append(root / Path(*parts[idx:]))

    win_match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if win_match:
        drive = win_match.group(1).lower()
        tail = win_match.group(2).replace("\\", "/")
        candidates.append(Path(f"/mnt/{drive}/{tail}"))

    wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", text)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        tail = wsl_match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{tail}"))

    return candidates


def resolve_row_path(row: Any, image_root: str, key: str) -> Path | None:
    seen: set[str] = set()
    for candidate in build_path_candidates(row.get(key, ""), image_root):
        key_name = str(candidate)
        if key_name in seen:
            continue
        seen.add(key_name)
        if candidate.exists():
            return candidate
    return None


def parse_json_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
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


def dedupe_answer_candidates(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = norm_text(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def get_gold_wrong_from_row(row: Any) -> tuple[str, str]:
    gold = norm_text(row.get("gold", row.get("answer", "")))
    wrong = norm_text(row.get("wrong", row.get("wrong_answer", row.get("unknown_target", UNKNOWN_ANSWER))))
    if not wrong:
        wrong = UNKNOWN_ANSWER
    return gold, wrong


def get_unknown_from_row(row: Any) -> str:
    unknown = norm_text(row.get("unknown_target", UNKNOWN_ANSWER))
    return unknown or UNKNOWN_ANSWER


def get_answer_candidates_from_row(row: Any) -> tuple[str, str, str, list[str]]:
    gold, wrong = get_gold_wrong_from_row(row)
    unknown = get_unknown_from_row(row)
    candidates = dedupe_answer_candidates(gold, wrong, unknown)
    return gold, wrong, unknown, candidates


def score_map_to_label(score_map: dict[str, float], gold: str, wrong: str, unknown: str) -> tuple[str, str]:
    if not score_map:
        return "", "other"
    pred_answer = max(score_map, key=lambda key: score_map[key])
    pred_norm = norm_text(pred_answer)
    if pred_norm == norm_text(gold):
        return pred_answer, "gold"
    if pred_norm == norm_text(unknown):
        return pred_answer, "unknown"
    if pred_norm == norm_text(wrong):
        return pred_answer, "wrong"
    return pred_answer, "other"


def resize_image_max_side(image: Image.Image, max_side: int = 672) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def load_nc_ic_images(
    row: Any,
    image_root: str,
    max_side: int = 672,
    mask_scale: float = 1.0,
) -> tuple[Path, Path, Path, list[str], Image.Image, Image.Image]:
    img_path = resolve_row_path(row, image_root, "image_path")
    det_path = resolve_row_path(row, image_root, "detection_path")
    mask_path = resolve_row_path(row, image_root, "mask_path")
    target_labels = parse_json_list(row.get("ic_target_labels"))

    if img_path is None or det_path is None:
        raise FileNotFoundError("Missing one or more image-conflict assets.")
    if not target_labels:
        raise ValueError("Missing ic_target_labels for image-conflict sample.")

    nc_image = resize_image_max_side(Image.open(img_path).convert("RGB"), max_side=max_side)
    ic_image, _ic_meta = build_masked_image(
        source_path=img_path,
        detection_path=det_path,
        target_labels=target_labels,
        mask_path=mask_path,
        image_id=str(row.get("img_id", "")).strip() or None,
        mask_scale=float(mask_scale),
    )
    ic_image = resize_image_max_side(ic_image, max_side=max_side)
    return img_path, det_path, mask_path, target_labels, nc_image, ic_image


def normalize_position(position: str) -> str:
    return "image_conflict"


def normalize_trace_mode(trace_mode: str) -> str:
    return "conflict"
