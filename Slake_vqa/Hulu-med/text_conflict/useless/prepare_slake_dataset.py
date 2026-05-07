import argparse
import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent


def normalize_split_name(name):
    lowered = str(name).strip().lower()
    if lowered in {"val", "valid", "validation"}:
        return "valid"
    return lowered


def load_json_records(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "records", "items", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported JSON layout in {path}")


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def resolve_dataset_root(project_root, raw_arg):
    candidate = (project_root / raw_arg).resolve()
    if candidate.exists():
        return candidate

    fallback = (project_root / "../../data").resolve()
    if fallback.exists():
        return fallback

    return candidate


def resolve_image_root(project_root, dataset_root, raw_arg):
    if raw_arg:
        return (project_root / raw_arg).resolve()

    candidates = [
        dataset_root / "imgs" / "imgs",
        dataset_root / "imgs",
        dataset_root,
    ]
    resolved = first_existing_path(candidates)
    if resolved is not None:
        return resolved.resolve()
    return candidates[0].resolve()


def resolve_split_json(dataset_root, split_name, override):
    if override:
        return Path(override)

    candidates = {
        "train": ["train.json", "train.jsonl"],
        "valid": ["validation.json", "valid.json", "val.json", "validation.jsonl"],
        "test": ["test.json", "test.jsonl"],
    }[split_name]
    for rel_name in candidates:
        candidate = dataset_root / rel_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {split_name} split JSON under {dataset_root}")


def make_output_row(record, split_name, image_root, project_root):
    img_name = str(record.get("img_name", "")).strip()
    img_id = str(record.get("img_id", "")).strip()
    image_path = first_existing_path(
        [
            Path(img_name) if img_name else None,
            image_root / img_name if img_name else None,
            image_root.parent / img_name if img_name else None,
        ]
    )
    if image_path is None and img_name:
        image_path = image_root / img_name

    if image_path is not None:
        try:
            stored_image_path = str(image_path.resolve().relative_to(project_root.resolve()))
        except Exception:
            stored_image_path = str(image_path.resolve() if image_path.exists() else image_path)
    else:
        stored_image_path = ""

    qid = str(record.get("qid", "")).strip()
    split_key = normalize_split_name(split_name)
    row_id = f"{split_key}:{qid}" if qid else f"{split_key}:{img_id}:{record.get('question', '')}"

    return {
        "id": row_id,
        "qid": qid,
        "img_id": img_id,
        "img_name": img_name,
        "image_path": stored_image_path,
        "question": str(record.get("question", "")).strip(),
        "answer": str(record.get("answer", "")).strip(),
        "gold": str(record.get("answer", "")).strip(),
        "category": str(record.get("content_type", "")).strip().lower() or "other",
        "content_type": str(record.get("content_type", "")).strip(),
        "answer_type": str(record.get("answer_type", "")).strip(),
        "q_lang": str(record.get("q_lang", "")).strip(),
        "location": str(record.get("location", "")).strip(),
        "modality": str(record.get("modality", "")).strip(),
        "base_type": str(record.get("base_type", "")).strip(),
        "triple": json.dumps(record.get("triple", ""), ensure_ascii=False),
        "split": split_key,
    }


def filter_record(record, english_only, closed_only):
    q_lang = str(record.get("q_lang", "")).strip().lower()
    answer_type = str(record.get("answer_type", "")).strip().upper()
    base_type = str(record.get("base_type", "")).strip().lower()
    content_type = str(record.get("content_type", "")).strip().lower()
    if english_only and q_lang not in {"", "en"}:
        return False
    if closed_only and answer_type != "CLOSED":
        return False
    if base_type == "kvqa" or content_type == "kg":
        return False
    return True


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = [
            "id",
            "qid",
            "img_id",
            "img_name",
            "image_path",
            "question",
            "answer",
            "gold",
            "category",
            "content_type",
            "answer_type",
            "q_lang",
            "location",
            "modality",
            "base_type",
            "triple",
            "split",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Convert Hugging Face SLAKE JSON splits into Hulu-med CSV inputs.")
    ap.add_argument("--dataset_root", default="../../data", help="Folder containing train/validation/test JSON and extracted images.")
    ap.add_argument("--image_root", default="", help="Override extracted image root. Defaults to <dataset_root>/imgs when present.")
    ap.add_argument("--train_json", default="")
    ap.add_argument("--valid_json", default="")
    ap.add_argument("--test_json", default="")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--include_non_english", action="store_true")
    ap.add_argument("--include_open", action="store_true")
    args = ap.parse_args()

    project_root = THIS_DIR
    dataset_root = resolve_dataset_root(project_root, args.dataset_root)
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = resolve_image_root(project_root, dataset_root, args.image_root)

    split_jsons = {
        "train": resolve_split_json(dataset_root, "train", args.train_json),
        "valid": resolve_split_json(dataset_root, "valid", args.valid_json),
        "test": resolve_split_json(dataset_root, "test", args.test_json),
    }

    english_only = not args.include_non_english
    closed_only = not args.include_open
    summary = {
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "english_only": english_only,
        "closed_only": closed_only,
        "exclude_kg": True,
        "splits": {},
    }

    all_rows = []
    for split_name, json_path in split_jsons.items():
        records = load_json_records(json_path)
        kept_rows = []
        skipped = 0
        for record in records:
            if not filter_record(record, english_only=english_only, closed_only=closed_only):
                skipped += 1
                continue
            kept_rows.append(make_output_row(record, split_name, image_root, project_root))

        split_csv = out_dir / f"slake_{split_name}_closed.csv"
        write_csv(split_csv, kept_rows)
        all_rows.extend(kept_rows)
        summary["splits"][split_name] = {
            "json_path": str(json_path),
            "csv_path": str(split_csv),
            "n_input_records": len(records),
            "n_kept_records": len(kept_rows),
            "n_skipped_records": skipped,
        }

    all_csv = out_dir / "slake_closed_all.csv"
    write_csv(all_csv, all_rows)
    summary["all_csv"] = str(all_csv)
    summary_path = out_dir / "slake_prepared_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_train={out_dir / 'slake_train_closed.csv'}")
    print(f"saved_valid={out_dir / 'slake_valid_closed.csv'}")
    print(f"saved_test={out_dir / 'slake_test_closed.csv'}")
    print(f"saved_all={all_csv}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
