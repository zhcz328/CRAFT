from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

IMAGE_EXT = ".jpg"
UNKNOWN_ANSWER = "unknown"

CLOSED_YN_PREFIXES = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ",
    "can ", "could ", "has ", "have ", "had ", "will ", "would ",
)


def norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())


def ensure_image_bytes_saved(image_field: Any, image_id: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"{slugify(image_id)}{IMAGE_EXT}"
    out_path = out_dir / img_name

    payload = image_field
    if isinstance(image_field, dict):
        payload = image_field.get("bytes")

    if isinstance(payload, (bytes, bytearray)):
        if not out_path.exists():
            out_path.write_bytes(payload)
        return str(out_path)

    raise ValueError(f"Unsupported image payload for image_id={image_id}")


def _iter_parquet_rows(parquet_path: Path) -> Iterable[dict[str, Any]]:
    pf = pq.ParquetFile(parquet_path)
    for rg_idx in range(pf.num_row_groups):
        batch = pf.read_row_group(rg_idx)
        for row in batch.to_pylist():
            yield row


def infer_closed(question: str, answer: str, answer_type: str = "", question_type: str = "") -> tuple[bool, str]:
    q = norm_text(question)
    a = norm_text(answer)
    if not q or not a:
        return False, "missing_question_or_answer"
    if a in {"yes", "no"}:
        return True, "yes_no_answer"
    if any(q.startswith(prefix) for prefix in CLOSED_YN_PREFIXES):
        return True, "yes_no_pattern"
    return False, "not_yes_no"


def choose_wrong_answer(gold: str, family: str, family_pool: dict[str, list[str]], global_pool: list[str]) -> tuple[str, str]:
    g = norm_text(gold)
    if g == "yes":
        return "no", "flip_yes_no"
    if g == "no":
        return "yes", "flip_yes_no"
    return UNKNOWN_ANSWER, "fallback_unknown"


def _build_answer_pools(records: list[dict[str, Any]]) -> tuple[dict[str, list[str]], list[str]]:
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    global_counter: Counter[str] = Counter()
    for rec in records:
        answer = norm_text(rec.get("gold", rec.get("answer", "")))
        family = norm_text(rec.get("question_family", "other"))
        if not answer:
            continue
        by_family[family][answer] += 1
        global_counter[answer] += 1

    family_sorted = {family: [ans for ans, _ in counter.most_common()] for family, counter in by_family.items()}
    global_sorted = [ans for ans, _ in global_counter.most_common()]
    return family_sorted, global_sorted


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_vqav2_rows(dataset_root: Path) -> list[dict[str, Any]]:
    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    rows: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        rows.extend(_iter_parquet_rows(parquet_path))
    return rows


def _load_gqa_image_map(dataset_root: Path) -> dict[str, Any]:
    parquet_files = sorted(dataset_root.glob("val-0000*-of-00003.parquet"))
    if not parquet_files:
        parquet_files = sorted(dataset_root.glob("val-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet image shards found under {dataset_root}")

    image_map: dict[str, Any] = {}
    for parquet_path in parquet_files:
        # only image-only shards contain image bytes
        for row in _iter_parquet_rows(parquet_path):
            image_id = str(row.get("id", "")).strip()
            image = row.get("image")
            if image_id and image is not None:
                image_map[image_id] = image
    return image_map


def _extract_annotation_values(annotations: dict[str, Any]) -> list[str]:
    if not isinstance(annotations, dict):
        return []
    values: list[str] = []
    for key in ("answer", "question", "fullAnswer"):
        items = annotations.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value", "")).strip()
            if value:
                values.append(value)
        if values:
            break
    dedup: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def _load_gqa_question_records(dataset_root: Path) -> tuple[list[dict[str, Any]], str]:
    qa_parquet = dataset_root / "val-00000-of-00001.parquet"
    if not qa_parquet.exists():
        raise FileNotFoundError(f"Missing GQA QA parquet: {qa_parquet}")

    rows: list[dict[str, Any]] = []
    for raw in _iter_parquet_rows(qa_parquet):
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        image_id = str(raw.get("imageId", "")).strip()
        qid = str(raw.get("id", "")).strip()
        annotations = raw.get("annotations") if isinstance(raw.get("annotations"), dict) else {}
        target_values = _extract_annotation_values(annotations)
        question_type = ""
        types = raw.get("types")
        if isinstance(types, dict):
            question_type = str(types.get("semantic", "")).strip()

        if not question or not answer or not image_id:
            continue

        rows.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "image_id": image_id,
                "question_type": question_type,
                "answer_type": "yes/no" if norm_text(answer) in {"yes", "no"} else "",
                "split": "valid",
                "source": "qa_parquet_annotations",
                "annotation_values": target_values,
            }
        )
    return rows, str(qa_parquet)


def _load_gqa_scene_graph(dataset_root: Path) -> dict[str, Any]:
    scene_graph_path = dataset_root / "val_sceneGraphs.json"
    if not scene_graph_path.exists():
        raise FileNotFoundError(f"Missing scene graph file: {scene_graph_path}")
    return json.loads(scene_graph_path.read_text(encoding="utf-8"))


def _filter_target_values_by_scene_graph(image_id: str, target_values: list[str], scene_graph: dict[str, Any]) -> list[str]:
    graph = scene_graph.get(str(image_id), {})
    objects = graph.get("objects", {}) if isinstance(graph, dict) else {}
    valid = []
    for value in target_values:
        if str(value) in objects:
            valid.append(str(value))
    if valid:
        return valid

    # fallback: try from semantic argument IDs if annotations missing; keep empty here
    return []


def prepare_vqav2_closed_with_wrong(dataset_root: Path, out_csv: Path, image_dir_name: str = "images") -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    rows = _load_vqav2_rows(dataset_root)

    image_dir = dataset_root / image_dir_name
    prepared: list[dict[str, Any]] = []
    skipped_open = 0
    skipped_non_yes_no = 0

    for raw in rows:
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("multiple_choice_answer", "")).strip()
        question_type = str(raw.get("question_type", "")).strip()
        answer_type = str(raw.get("answer_type", "")).strip()

        is_closed, _ = infer_closed(question=question, answer=answer, answer_type=answer_type, question_type=question_type)
        if not is_closed:
            skipped_open += 1
            continue

        answer_norm = norm_text(answer)
        if answer_norm not in {"yes", "no"}:
            skipped_non_yes_no += 1
            continue

        image_id = str(raw.get("image_id", "")).strip()
        question_id = str(raw.get("question_id", "")).strip()
        image_path = ensure_image_bytes_saved(raw.get("image"), image_id=image_id, out_dir=image_dir)

        prepared.append(
            {
                "id": f"valid:{question_id or image_id}",
                "qid": question_id,
                "img_id": image_id,
                "img_name": f"{slugify(image_id)}{IMAGE_EXT}",
                "image_path": image_path,
                "detection_path": "",
                "mask_path": "",
                "question": question,
                "answer": answer_norm,
                "gold": answer_norm,
                "wrong": "",
                "category": norm_text(question_type) or "other",
                "content_type": question_type,
                "answer_type": answer_type,
                "q_lang": "en",
                "location": "",
                "modality": "",
                "base_type": "",
                "triple": "",
                "split": "valid",
                "dataset_name": "VQAv2_validation",
                "question_type": question_type,
                "question_family": norm_text(question_type) or "other",
                "closed_reason": "yes_no_answer_only",
                "wrong_reason": "",
                "source_dataset": "VQAv2_validation",
                "ic_target_labels": "[]",
                "ic_target_label_count": 0,
            }
        )

    for rec in prepared:
        rec["wrong"] = "no" if rec["gold"] == "yes" else "yes"
        rec["wrong_reason"] = "flip_yes_no"

    _write_csv(out_csv, prepared)
    summary = {
        "dataset_name": "VQAv2_validation",
        "dataset_root": str(dataset_root),
        "out_csv": str(out_csv),
        "n_input_rows": len(rows),
        "n_closed_rows": len(prepared),
        "n_yes_no_rows": len(prepared),
        "n_skipped_open_rows": skipped_open,
        "n_skipped_non_yes_no_rows": skipped_non_yes_no,
        "image_dir": str(image_dir),
    }
    _write_summary(out_csv.with_suffix(".summary.json"), summary)
    return summary


def prepare_gqa_closed_with_wrong(
    dataset_root: Path,
    out_csv: Path,
    image_dir_name: str = "images",
    allow_scene_graph_synthetic: bool = True,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    image_map = _load_gqa_image_map(dataset_root)
    question_rows, question_source = _load_gqa_question_records(dataset_root)
    scene_graph = _load_gqa_scene_graph(dataset_root)
    scene_graph_path = dataset_root / "val_sceneGraphs.json"

    image_dir = dataset_root / image_dir_name
    prepared: list[dict[str, Any]] = []
    skipped_open = 0
    skipped_non_yes_no = 0
    skipped_missing_image = 0
    skipped_missing_target_value = 0

    for raw in question_rows:
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        question_type = str(raw.get("question_type", "")).strip()
        answer_type = str(raw.get("answer_type", "")).strip()

        is_closed, _ = infer_closed(question=question, answer=answer, answer_type=answer_type, question_type=question_type)
        if not is_closed:
            skipped_open += 1
            continue

        answer_norm = norm_text(answer)
        if answer_norm not in {"yes", "no"}:
            skipped_non_yes_no += 1
            continue

        image_id = str(raw.get("image_id", "")).strip()
        payload = image_map.get(image_id)
        if payload is None:
            skipped_missing_image += 1
            continue

        annotation_values = [str(v) for v in raw.get("annotation_values", [])]
        target_values = _filter_target_values_by_scene_graph(image_id=image_id, target_values=annotation_values, scene_graph=scene_graph)
        if not target_values:
            skipped_missing_target_value += 1
            continue

        image_path = ensure_image_bytes_saved(payload, image_id=image_id, out_dir=image_dir)
        qid = str(raw.get("qid", "")).strip() or f"valid:{image_id}:{slugify(question)[:48]}"
        split = str(raw.get("split", "valid")).strip().lower() or "valid"

        prepared.append(
            {
                "id": f"{split}:{qid}",
                "qid": qid,
                "img_id": image_id,
                "img_name": f"{slugify(image_id)}{IMAGE_EXT}",
                "image_path": image_path,
                "detection_path": str(scene_graph_path),
                "mask_path": "",
                "question": question,
                "answer": answer_norm,
                "gold": answer_norm,
                "wrong": "",
                "category": norm_text(question_type) or "other",
                "content_type": question_type,
                "answer_type": answer_type,
                "q_lang": "en",
                "location": "",
                "modality": "",
                "base_type": "",
                "triple": "",
                "split": split,
                "dataset_name": "GQA",
                "question_type": question_type,
                "question_family": norm_text(question_type) or "other",
                "closed_reason": "yes_no_answer_only",
                "wrong_reason": "",
                "source_dataset": str(raw.get("source", "qa_parquet_annotations")),
                "ic_target_labels": json.dumps(target_values, ensure_ascii=False),
                "ic_target_label_count": len(target_values),
                "annotation_values": json.dumps(annotation_values, ensure_ascii=False),
            }
        )

    for rec in prepared:
        rec["wrong"] = "no" if rec["gold"] == "yes" else "yes"
        rec["wrong_reason"] = "flip_yes_no"

    _write_csv(out_csv, prepared)
    summary = {
        "dataset_name": "GQA",
        "dataset_root": str(dataset_root),
        "out_csv": str(out_csv),
        "n_image_rows": len(image_map),
        "n_question_rows": len(question_rows),
        "n_closed_rows": len(prepared),
        "n_yes_no_rows": len(prepared),
        "n_skipped_open_rows": skipped_open,
        "n_skipped_non_yes_no_rows": skipped_non_yes_no,
        "n_skipped_missing_image_rows": skipped_missing_image,
        "n_skipped_missing_target_value_rows": skipped_missing_target_value,
        "question_source": question_source,
        "scene_graph_path": str(scene_graph_path),
        "image_dir": str(image_dir),
    }
    _write_summary(out_csv.with_suffix(".summary.json"), summary)
    return summary
