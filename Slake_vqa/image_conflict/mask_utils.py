import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw


UNKNOWN_ANSWER = "unknown"

_STOP_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "body",
    "contain",
    "contains",
    "do",
    "does",
    "image",
    "in",
    "is",
    "look",
    "looks",
    "of",
    "or",
    "patient",
    "picture",
    "show",
    "shows",
    "study",
    "the",
    "this",
    "what",
    "where",
    "which",
}

_NON_OBJECT_ANSWERS = {
    "",
    "yes",
    "no",
    "left",
    "right",
    "upper",
    "lower",
    "none",
    "normal",
    "abnormal",
    UNKNOWN_ANSWER,
}

_DIRECTION_TOKENS = {"left", "right", "upper", "lower"}

_PHRASE_ALIASES = {
    "brian": "brain",
    "kidneys": "kidney",
    "lungs": "lung",
    "livers": "liver",
    "spleens": "spleen",
    "hearts": "heart",
    "eyes": "eye",
    "ears": "ear",
    "teeth": "tooth",
    "temporal lobes": "temporal lobe",
    "tumours": "tumor",
    "infiltration": "infiltrate",
    "pleural infiltration": "effusion",
}

_BASE_QUERY_EXPANSIONS = {
    "kidney": {"left kidney", "right kidney"},
    "lung": {"left lung", "right lung"},
    "mandible": {"left mandible", "right mandible"},
    "parotid": {"left parotid", "right parotid"},
    "ear": {"left ear", "right ear"},
    "eye": {"left eye", "right eye"},
    "femoral head": {"left femoral head", "right femoral head"},
}

_PROXY_QUERY_EXPANSIONS = {
    "liver": {"liver cancer"},
    "lung": {"lung cancer", "nodule", "mass", "pneumonia", "atelectasis", "infiltrate", "effusion", "pneumothorax"},
    "heart": {"cardiomegaly"},
    "brain": {"brain edema", "brain enhancing tumor", "brain non enhancing tumor"},
    "brain tumor": {"brain enhancing tumor", "brain non enhancing tumor"},
    "tumor": {"brain enhancing tumor", "brain non enhancing tumor", "liver cancer", "lung cancer"},
}

_MASKABLE_CONTENT_TYPES = {"organ", "abnormality", "size", "position", "color", "quantity", "shape"}

_NON_GROUNDABLE_GENERIC_TARGETS = {
    "",
    "organ",
    "organs",
    "abnormality",
    "abnormalities",
    "kind of abnormality",
    "kinds of abnormality",
    "kind of abnormalities",
    "kinds of abnormalities",
}


def norm_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _canonical_text(text: str) -> str:
    value = norm_text(text)
    if not value:
        return ""
    for src, dst in sorted(_PHRASE_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", dst, value)
    return " ".join(value.split())


def normalize_answer(text: str) -> str:
    return norm_text(text)


def is_unknown_text(text: str) -> bool:
    norm = normalize_answer(text)
    if not norm:
        return False
    unknown_phrases = {
        "unknown",
        "cannot determine",
        "can not determine",
        "cannot tell",
        "can not tell",
        "not sure",
        "unsure",
        "insufficient information",
        "not enough information",
        "cannot answer",
        "can not answer",
        "unable to determine",
        "unanswerable",
    }
    if norm in unknown_phrases:
        return True
    return any(phrase in norm for phrase in unknown_phrases if " " in phrase)


def answers_match(prediction: str, gold: str) -> bool:
    pred = normalize_answer(prediction)
    ref = normalize_answer(gold)
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    yes_set = {"yes", "yeah", "y", "present", "contained", "contains"}
    no_set = {"no", "n", "absent", "not present", "not contained", "does not contain"}
    if ref in yes_set:
        return pred in yes_set
    if ref in no_set:
        return pred in no_set
    return False


def tokenize_text(text: str) -> List[str]:
    return [tok for tok in _canonical_text(text).split() if tok and tok not in _STOP_TOKENS]


def read_detection_map(path: str | Path) -> Dict[str, List[float]]:
    det_path = Path(path)
    if not det_path.exists():
        return {}
    raw = json.loads(det_path.read_text(encoding="utf-8"))
    out: Dict[str, List[float]] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if isinstance(value, (list, tuple)) and len(value) == 4:
                    out[str(key)] = [float(v) for v in value]
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, (list, tuple)) and len(value) == 4:
                out[str(key)] = [float(v) for v in value]
    return out


def split_answer_candidates(answer: str) -> List[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    parts = re.split(r"\s*,\s*|\s+or\s+|\s*/\s*|\s*;\s*", text)
    return [part.strip() for part in parts if part.strip()]


def has_inline_mask_data(row: Any) -> bool:
    if row is None:
        return False
    raw_h = str(row.get("mask_h", "") or "").strip()
    raw_w = str(row.get("mask_w", "") or "").strip()
    if not raw_h or not raw_w:
        return False
    try:
        return int(raw_h) > 0 and int(raw_w) > 0 and "mask_rle" in row
    except Exception:
        return False


def extract_inline_mask_data(row: Any) -> dict[str, Any] | None:
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


def decode_rle_mask(rle: Sequence[int], height: int, width: int) -> Image.Image:
    flat = np.zeros(height * width, dtype=np.uint8)
    values = [int(v) for v in rle]
    if len(values) % 2 != 0:
        raise ValueError(f"Invalid inline mask RLE length: {len(values)}")
    for start, length in zip(values[0::2], values[1::2]):
        if int(length) <= 0:
            continue
        s = int(start)
        e = min(flat.size, s + int(length))
        if s < 0 or s >= flat.size:
            continue
        flat[s:e] = 255
    arr = flat.reshape((height, width), order="F")
    return Image.fromarray(arr, mode="L")


def scale_binary_mask(mask: Image.Image, scale: float) -> Image.Image:
    scale = max(float(scale), 1e-6)
    if abs(scale - 1.0) < 1e-8:
        return mask

    bbox = mask.getbbox()
    if bbox is None:
        return mask

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    scaled_w = max(1, int(round(width * scale)))
    scaled_h = max(1, int(round(height * scale)))

    crop = mask.crop((x1, y1, x2, y2))
    if crop.size != (scaled_w, scaled_h):
        crop = crop.resize((scaled_w, scaled_h), resample=Image.Resampling.NEAREST)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = int(round(cx - scaled_w / 2.0))
    ny1 = int(round(cy - scaled_h / 2.0))
    nx2 = nx1 + scaled_w
    ny2 = ny1 + scaled_h

    canvas = Image.new("L", mask.size, 0)

    src_x1 = max(0, -nx1)
    src_y1 = max(0, -ny1)
    src_x2 = scaled_w - max(0, nx2 - mask.size[0])
    src_y2 = scaled_h - max(0, ny2 - mask.size[1])
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return canvas

    clipped = crop.crop((src_x1, src_y1, src_x2, src_y2))
    paste_x = max(0, nx1)
    paste_y = max(0, ny1)
    canvas.paste(clipped, (paste_x, paste_y))
    return canvas


def _strip_direction_prefix(text: str) -> str:
    tokens = _canonical_text(text).split()
    while tokens and tokens[0] in _DIRECTION_TOKENS:
        tokens = tokens[1:]
    return " ".join(tokens)


def _generic_variants(label: str) -> List[str]:
    norm = _canonical_text(label)
    variants = {norm}
    tokens = norm.split()
    base = _strip_direction_prefix(norm)
    if base and base != norm:
        variants.add(base)
    if len(tokens) >= 2 and tokens[-1] in {"tumor", "cancer", "kidney", "lung", "eye", "ear", "mandible", "parotid", "lobe"}:
        variants.add(tokens[-1])
    return sorted(variants)


def _label_variants(label: str) -> set[str]:
    variants = set(_generic_variants(label))
    norm = _canonical_text(label)
    if norm:
        variants.add(norm)
    return {item for item in variants if item}


def _query_variants(query: str, allow_proxy: bool = False) -> set[str]:
    norm = _canonical_text(query)
    if not norm:
        return set()
    variants = {norm}
    base = _strip_direction_prefix(norm)
    if base:
        variants.add(base)
        variants.update(_BASE_QUERY_EXPANSIONS.get(base, set()))
        if allow_proxy:
            variants.update(_PROXY_QUERY_EXPANSIONS.get(base, set()))
    variants.update(_BASE_QUERY_EXPANSIONS.get(norm, set()))
    if allow_proxy:
        variants.update(_PROXY_QUERY_EXPANSIONS.get(norm, set()))
    return {item for item in variants if item}


def _tokens_match(query: str, label: str) -> bool:
    q_tokens = tokenize_text(query)
    l_tokens = tokenize_text(label)
    if not q_tokens or not l_tokens:
        return False
    q_set = set(q_tokens)
    l_set = set(l_tokens)
    return q_set.issubset(l_set) or l_set.issubset(q_set)


def _match_score(query: str, label: str, allow_proxy: bool = False) -> int:
    query_norm = _canonical_text(query)
    label_norm = _canonical_text(label)
    if not query_norm or not label_norm:
        return 0

    label_variants = _label_variants(label)
    query_variants = _query_variants(query, allow_proxy=allow_proxy)

    if query_norm == label_norm:
        return 100
    if query_norm in label_variants:
        return 95
    if any(query_variant in label_variants for query_variant in query_variants):
        return 90
    if _tokens_match(query_norm, label_norm):
        return 80

    if not allow_proxy:
        return 0

    if any(query_variant == label_norm for query_variant in query_variants):
        return 85
    if any(query_variant and query_variant in label_norm for query_variant in query_variants):
        return 70
    if any(label_variant and query_norm in label_variant for label_variant in label_variants):
        return 60
    if _tokens_match(query_norm, label_norm):
        return 50
    return 0


def match_detection_labels(query: str, available_labels: Sequence[str], allow_proxy: bool = False) -> List[str]:
    scored: list[tuple[int, str]] = []
    for label in available_labels:
        score = _match_score(query, label, allow_proxy=allow_proxy)
        if score > 0:
            scored.append((score, label))
    if not scored:
        return []
    best_score = max(score for score, _label in scored)
    return [label for score, label in scored if score == best_score]


def _build_non_maskable_metadata(question_type: str, skip_reason: str, detection_labels: Sequence[str]) -> Dict[str, Any]:
    return {
        "question_type": question_type,
        "maskable": False,
        "match_policy": "drop",
        "skip_reason": skip_reason,
        "query_targets": [],
        "target_labels": [],
        "resolved": False,
        "detection_label_count": len(list(detection_labels)),
    }


def _extract_quoted_target(question_norm: str, pattern: str) -> str:
    match = re.match(pattern, question_norm)
    if not match:
        return ""
    return _canonical_text(match.group("target"))


def _clean_query_target(target: str) -> str:
    value = _canonical_text(target)
    if not value:
        return ""
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    value = re.sub(r"^(?:existing|existed|shown|showed)\s+", "", value)
    value = re.sub(r"\s+(?:are\s+there|are\s+shown|have\s+existed|does\s+the\s+image\s+have)$", "", value)
    return value.strip()


def classify_question(
    question: str,
    gold_answer: str,
    content_type: str = "",
) -> Dict[str, Any]:
    question_norm = _canonical_text(question)
    answer_norm = _canonical_text(gold_answer)
    content_norm = _canonical_text(content_type)

    if content_norm not in _MASKABLE_CONTENT_TYPES:
        return _build_non_maskable_metadata(
            question_type=f"non_maskable_{content_norm or 'other'}",
            skip_reason="content_type_not_visual_target",
            detection_labels=[],
        )

    if content_norm == "modality":
        return _build_non_maskable_metadata("non_maskable_modality", "modality_question", [])
    if content_norm == "plane":
        return _build_non_maskable_metadata("non_maskable_plane", "plane_question", [])

    if question_norm.startswith("does the picture contain "):
        target = _extract_quoted_target(question_norm, r"^does the picture contain (?P<target>.+)$")
        if answer_norm != "yes":
            return _build_non_maskable_metadata("organ_presence_negative", "negative_presence_question", [])
        return {
            "question_type": "organ_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [target] if target else [],
        }

    if question_norm.startswith("is there "):
        target = _extract_quoted_target(
            question_norm,
            r"^is there (?P<target>.+?)(?: in (?:this|the) (?:image|picture))?$",
        )
        if answer_norm != "yes":
            return _build_non_maskable_metadata("organ_presence_negative", "negative_presence_question", [])
        return {
            "question_type": "organ_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [target] if target else [],
        }

    if question_norm.startswith("are is there "):
        target = _extract_quoted_target(
            question_norm,
            r"^are is there (?P<target>.+?)(?: in (?:this|the) (?:image|picture))?$",
        )
        if answer_norm != "yes":
            return _build_non_maskable_metadata("organ_presence_negative", "negative_presence_question", [])
        return {
            "question_type": "organ_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [target] if target else [],
        }

    match = re.match(r"^does (?:the )?(?P<target>.+?) (?:exist|appear) in (?:this|the) (?:image|picture)$", question_norm)
    if match:
        if answer_norm != "yes":
            return _build_non_maskable_metadata("organ_presence_negative", "negative_presence_question", [])
        return {
            "question_type": "organ_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    match = re.match(r"^can (?P<target>.+?) be observed\b", question_norm)
    if match:
        if answer_norm != "yes":
            return _build_non_maskable_metadata("explicit_disease_negative", "negative_disease_presence", [])
        return {
            "question_type": "explicit_disease_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    match = re.match(r"^does the patient have (?P<target>.+)$", question_norm)
    if match:
        if answer_norm != "yes":
            return _build_non_maskable_metadata("explicit_disease_negative", "negative_disease_presence", [])
        target = _canonical_text(match.group("target"))
        if target in {"any abnormality", "any abnormalities"}:
            return {
                "question_type": "explicit_disease_presence",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": ["abnormality", "tumor", "cancer", "nodule", "mass", "pneumonia", "atelectasis", "infiltrate", "effusion", "pneumothorax", "cardiomegaly"],
            }
        return {
            "question_type": "explicit_disease_presence",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [target],
        }

    match = re.match(r"^is the (?P<target>.+?) healthy$", question_norm)
    if match:
        return {
            "question_type": "organ_health",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    match = re.match(r"^are the (?P<target>.+?) healthy$", question_norm)
    if match:
        return {
            "question_type": "organ_health",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    match = re.match(r"^does (?:the )?(?P<target>.+?) look (?:abnormal|normal)$", question_norm)
    if match:
        return {
            "question_type": "organ_health",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    match = re.match(r"^is the (?P<target>.+?) abnormal$", question_norm)
    if match:
        return {
            "question_type": "organ_health",
            "maskable": True,
            "match_policy": "question_proxy",
            "skip_reason": "",
            "query_targets": [_canonical_text(match.group("target"))],
        }

    if any(
        phrase in question_norm
        for phrase in (
            "is this image normal",
            "does this image look abnormal",
            "does this image look normal",
            "are there abnormalities in this image",
            "does the patient have any abnormalities",
        )
    ):
        return _build_non_maskable_metadata("generic_abnormality", "generic_abnormality_question", [])

    if any(
        phrase in question_norm
        for phrase in (
            "which is bigger",
            "which is the bigger",
            "which is the biggest",
            "what is the largest organ",
            "which is smaller",
            "which is the smallest",
        )
    ):
        answer_targets = [_canonical_text(part) for part in split_answer_candidates(gold_answer)]
        answer_targets = [target for target in answer_targets if target and target not in {"both", "neither"}]
        if not answer_targets:
            return _build_non_maskable_metadata("size_compare", "missing_answer_target", [])
        return {
            "question_type": "size_compare",
            "maskable": True,
            "match_policy": "answer_proxy",
            "skip_reason": "",
            "query_targets": answer_targets,
        }

    match = re.match(r"^which side of (?P<target>.+?) is abnormal in this image (?:left or right)$", question_norm)
    if match and answer_norm in {"left", "right"}:
        return {
            "question_type": "side_specific_position",
            "maskable": True,
            "match_policy": "question_strict",
            "skip_reason": "",
            "query_targets": [f"{answer_norm} {_canonical_text(match.group('target'))}"],
        }

    if any(phrase in question_norm for phrase in ("which hemisphere is abnormal", "which lobe is abnormal")):
        return _build_non_maskable_metadata("side_specific_position", "side_specific_without_groundable_target", [])

    if question_norm.startswith("do the organs in the image exist in the "):
        return _build_non_maskable_metadata("non_maskable_position", "body_region_question", [])

    if content_norm == "position":
        match = re.match(r"^where is the (?P<target>.+?)(?: in this image| in the image| located| located in this image)?$", question_norm)
        if match:
            target = _clean_query_target(match.group("target"))
            if target not in _NON_GROUNDABLE_GENERIC_TARGETS and answer_norm not in {"not seen", "none"}:
                return {
                    "question_type": "target_location",
                    "maskable": True,
                    "match_policy": "question_proxy",
                    "skip_reason": "",
                    "query_targets": [target] if target else [],
                }

        match = re.match(r"^where is/are the (?P<target>.+?)(?: located| located in this picture| located in this image)?$", question_norm)
        if match:
            target = _clean_query_target(match.group("target"))
            if target not in _NON_GROUNDABLE_GENERIC_TARGETS:
                return {
                    "question_type": "target_location",
                    "maskable": True,
                    "match_policy": "question_proxy",
                    "skip_reason": "",
                    "query_targets": [target] if target else [],
                }

        match = re.match(r"^what part of the lung is the (?P<target>.+?) located in$", question_norm)
        if match:
            target = _clean_query_target(match.group("target"))
            return {
                "question_type": "lung_part_location",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": [target] if target else [],
            }

        match = re.match(r"^which side of (?P<target>.+?) is abnormal in this image(?:,left or right)?$", question_norm)
        if match and answer_norm in {"left", "right"}:
            target = _clean_query_target(match.group("target"))
            return {
                "question_type": "side_target_position",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": [f"{answer_norm} {target}", target] if target else [],
            }

    if content_norm == "quantity":
        quantity_patterns = (
            r"^how many (?P<target>.+?) are there in this image$",
            r"^how many (?P<target>.+?) in this image$",
            r"^how many (?P<target>.+?) are shown in this image$",
            r"^how many (?P<target>.+?) have existed in this image$",
            r"^how many (?P<target>.+?) does the image have$",
            r"^how many existing (?P<target>.+?) in this image$",
            r"^how many existing (?P<target>.+?) are there in this image$",
        )
        for pattern in quantity_patterns:
            match = re.match(pattern, question_norm)
            if not match:
                continue
            target = _clean_query_target(match.group("target"))
            if target in _NON_GROUNDABLE_GENERIC_TARGETS:
                return _build_non_maskable_metadata("non_maskable_quantity", "generic_quantity_target", [])
            return {
                "question_type": "quantity_count",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": [target] if target else [],
            }
        return _build_non_maskable_metadata("non_maskable_quantity", "quantity_question", [])

    if content_norm == "shape":
        shape_patterns = (
            r"^what is the shape of (?P<target>.+?) in (?:this image|the picture)$",
            r"^what is the shape of (?P<target>.+?) about this patient$",
        )
        for pattern in shape_patterns:
            match = re.match(pattern, question_norm)
            if not match:
                continue
            target = _clean_query_target(match.group("target"))
            if target in _NON_GROUNDABLE_GENERIC_TARGETS:
                return _build_non_maskable_metadata("non_maskable_shape", "generic_shape_target", [])
            return {
                "question_type": "shape_attribute",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": [target] if target else [],
            }
        return _build_non_maskable_metadata("non_maskable_shape", "shape_question", [])

    if content_norm == "color":
        if "tumor enhancing" in question_norm or "brain tumor white or gray" in question_norm:
            return {
                "question_type": "tumor_attribute",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": ["tumor"],
            }
        color_patterns = (
            r"^what color does (?P<target>.+?) show in the picture$",
            r"^what color do (?P<target>.+?) show in the picture$",
            r"^what color is the (?P<target>.+?) in (?:this image|the picture)$",
            r"^what color are the (?P<target>.+?) in (?:this image|the picture)$",
            r"^what is the color of (?P<target>.+?) in this image$",
            r"^what color are (?P<target>.+?) in the picture$",
            r"^what color do (?P<target>.+?) show in this image$",
            r"^what density is the (?P<target>.+?)$",
        )
        for pattern in color_patterns:
            match = re.match(pattern, question_norm)
            if not match:
                continue
            target = _clean_query_target(match.group("target"))
            if target in _NON_GROUNDABLE_GENERIC_TARGETS:
                return _build_non_maskable_metadata("non_maskable_color", "generic_color_target", [])
            return {
                "question_type": "color_attribute",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": [target] if target else [],
            }
        if question_norm in {"is the abnormality hyperdense or hypodense", "what is the color of abnormality in this image"}:
            return {
                "question_type": "abnormality_color",
                "maskable": True,
                "match_policy": "question_proxy",
                "skip_reason": "",
                "query_targets": ["abnormality", "tumor", "cancer", "nodule", "mass"],
            }
        return _build_non_maskable_metadata("non_maskable_color", "color_question", [])

    return _build_non_maskable_metadata("unhandled_closed_question", "no_matching_question_template", [])


def resolve_target_metadata(
    question: str,
    gold_answer: str,
    detection_labels: Sequence[str],
    content_type: str = "",
    proxy_strategy: str = "strict",
) -> Dict[str, Any]:
    labels = list(detection_labels)
    meta = classify_question(question=question, gold_answer=gold_answer, content_type=content_type)
    meta["detection_label_count"] = len(labels)
    meta["proxy_strategy"] = proxy_strategy

    if not meta.get("maskable"):
        meta["target_labels"] = []
        meta["resolved"] = False
        return meta

    if not labels:
        meta["target_labels"] = []
        meta["resolved"] = False
        meta["skip_reason"] = "missing_detection_labels"
        return meta

    match_policy = str(meta.get("match_policy", "question_strict"))
    allow_proxy = proxy_strategy == "proxy" and match_policy.endswith("proxy")
    queries = [target for target in meta.get("query_targets", []) if str(target).strip()]
    if not queries:
        meta["target_labels"] = []
        meta["resolved"] = False
        meta["skip_reason"] = meta.get("skip_reason") or "missing_query_target"
        return meta
    resolved_labels: List[str] = []

    for query in queries:
        matches = match_detection_labels(query, labels, allow_proxy=allow_proxy)
        for label in matches:
            if label not in resolved_labels:
                resolved_labels.append(label)

    meta["target_labels"] = resolved_labels
    meta["resolved"] = bool(resolved_labels)
    if meta["maskable"] and not meta["resolved"] and not meta.get("skip_reason"):
        meta["skip_reason"] = "no_matching_detection_label"
    return meta


def resolve_target_labels(
    question: str,
    gold_answer: str,
    detection_labels: Sequence[str],
    content_type: str = "",
    proxy_strategy: str = "strict",
) -> List[str]:
    meta = resolve_target_metadata(
        question=question,
        gold_answer=gold_answer,
        detection_labels=detection_labels,
        content_type=content_type,
        proxy_strategy=proxy_strategy,
    )
    return list(meta.get("target_labels", []))


def _bbox_to_ints(
    bbox: Sequence[float],
    width: int,
    height: int,
    scale: float = 1.0,
) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox]
    scale = max(float(scale), 1e-6)
    scaled_w = max(1.0, w * scale)
    scaled_h = max(1.0, h * scale)
    cx = x + w / 2.0
    cy = y + h / 2.0
    x1 = int(round(cx - scaled_w / 2.0))
    y1 = int(round(cy - scaled_h / 2.0))
    x2 = int(round(cx + scaled_w / 2.0))
    y2 = int(round(cy + scaled_h / 2.0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    return x1, y1, x2, y2


def build_target_mask(
    image_size: tuple[int, int],
    detection_map: Dict[str, Sequence[float]],
    target_labels: Iterable[str],
    mask_path: str | Path | None = None,
    scale: float = 1.0,
) -> tuple[Image.Image, Dict[str, int | bool]]:
    width, height = image_size
    union_mask = Image.new("L", (width, height), 0)
    full_mask = None
    if mask_path and Path(mask_path).exists():
        full_mask = Image.open(mask_path).convert("L")
        if full_mask.size != (width, height):
            full_mask = full_mask.resize((width, height))

    draw = ImageDraw.Draw(union_mask)
    used_mask_pixels = False
    fallback_rectangles = 0
    selected = 0
    for label in target_labels:
        bbox = detection_map.get(label)
        if bbox is None:
            continue
        selected += 1
        base_x1, base_y1, base_x2, base_y2 = _bbox_to_ints(bbox, width, height, scale=1.0)
        x1, y1, x2, y2 = _bbox_to_ints(bbox, width, height, scale=scale)
        if full_mask is not None and base_x2 > base_x1 and base_y2 > base_y1:
            crop = full_mask.crop((base_x1, base_y1, base_x2, base_y2))
            nonzero = ImageChops.lighter(crop, Image.new("L", crop.size, 0))
            if nonzero.getbbox():
                crop = crop.point(lambda px: 255 if px > 0 else 0)
                patch = Image.new("L", (width, height), 0)
                if crop.size != (x2 - x1, y2 - y1):
                    crop = crop.resize((max(1, x2 - x1), max(1, y2 - y1)), resample=Image.Resampling.NEAREST)
                patch.paste(crop, (x1, y1))
                union_mask = ImageChops.lighter(union_mask, patch)
                used_mask_pixels = True
                continue
        draw.rectangle((x1, y1, x2, y2), fill=255)
        fallback_rectangles += 1

    return union_mask, {
        "selected_label_count": selected,
        "used_mask_pixels": used_mask_pixels,
        "fallback_rectangles": fallback_rectangles,
    }


def apply_mask_to_image(
    image: Image.Image,
    target_mask: Image.Image,
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    overlay = Image.new("RGB", image.size, fill_rgb)
    return Image.composite(overlay, image, target_mask)


def apply_gaussian_noise_to_image(
    image: Image.Image,
    noise_std: float = 64.0,
    noise_seed: int | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    if noise_std <= 0:
        return image.copy()
    image_np = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=image_np.shape)
    noisy = np.clip(image_np + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def apply_noise_to_image(
    image: Image.Image,
    target_mask: Image.Image,
    noise_std: float = 64.0,
    noise_seed: int | None = None,
) -> Image.Image:
    image = image.convert("RGB")
    noisy = apply_gaussian_noise_to_image(image, noise_std=noise_std, noise_seed=noise_seed)
    return Image.composite(noisy, image, target_mask)


def build_masked_image(
    source_path: str | Path,
    detection_path: str | Path | None,
    target_labels: Sequence[str],
    mask_path: str | Path | None = None,
    mask_rle: Sequence[int] | None = None,
    mask_height: int | None = None,
    mask_width: int | None = None,
    fill_rgb: tuple[int, int, int] = (0, 0, 0),
    mask_all: bool = False,
    add_noise: bool = False,
    add_noise_all: bool = False,
    noise_std: float = 64.0,
    noise_seed: int | None = None,
    mask_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> tuple[Image.Image, Dict[str, object]]:
    source = Image.open(source_path).convert("RGB")
    target_labels = list(target_labels)
    inline_mask = None
    if mask_rle is not None and mask_height and mask_width:
        inline_mask = decode_rle_mask(mask_rle, int(mask_height), int(mask_width))
        if source.size != inline_mask.size:
            source = source.resize(inline_mask.size, resample=Image.Resampling.BILINEAR)
    if mask_all:
        target_mask = Image.new("L", source.size, 255)
        mask_meta = {
            "selected_label_count": len(target_labels),
            "used_mask_pixels": False,
            "fallback_rectangles": 0,
            "mask_all": True,
            "add_noise": False,
            "add_noise_all": False,
            "noise_std": 0.0,
            "noise_seed": None,
            "mask_scale": float(mask_scale),
            "noise_scale": float(noise_scale),
        }
    elif add_noise_all:
        masked = apply_gaussian_noise_to_image(source, noise_std=noise_std, noise_seed=noise_seed)
        meta = {
            "target_labels": target_labels,
            "mask_bbox": None,
            "selected_label_count": len(target_labels),
            "used_mask_pixels": False,
            "fallback_rectangles": 0,
            "mask_all": False,
            "add_noise": False,
            "add_noise_all": True,
            "noise_std": float(noise_std),
            "noise_seed": noise_seed,
            "mask_scale": float(mask_scale),
            "noise_scale": float(noise_scale),
        }
        return masked, meta
    else:
        if inline_mask is not None:
            target_scale = noise_scale if add_noise else mask_scale
            target_mask = scale_binary_mask(inline_mask, target_scale)
            mask_meta = {
                "selected_label_count": len(target_labels),
                "used_mask_pixels": bool(target_mask.getbbox()),
                "fallback_rectangles": 0,
            }
        else:
            if detection_path is None:
                raise FileNotFoundError("Missing detection_path for partial image mask.")
            detection_map = read_detection_map(detection_path)
            target_mask, mask_meta = build_target_mask(
                image_size=source.size,
                detection_map=detection_map,
                target_labels=target_labels,
                mask_path=mask_path,
                scale=noise_scale if add_noise else mask_scale,
            )
        mask_meta["mask_all"] = False
        mask_meta["add_noise"] = bool(add_noise)
        mask_meta["add_noise_all"] = False
        mask_meta["noise_std"] = float(noise_std) if add_noise else 0.0
        mask_meta["noise_seed"] = noise_seed if add_noise else None
        mask_meta["mask_scale"] = float(mask_scale)
        mask_meta["noise_scale"] = float(noise_scale)
    if add_noise:
        masked = apply_noise_to_image(source, target_mask, noise_std=noise_std, noise_seed=noise_seed)
    else:
        masked = apply_mask_to_image(source, target_mask, fill_rgb=fill_rgb)
    meta = {
        "target_labels": target_labels,
        "mask_bbox": target_mask.getbbox(),
        **mask_meta,
    }
    return masked, meta



