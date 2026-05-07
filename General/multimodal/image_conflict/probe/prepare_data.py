import argparse
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import (
    FOLLOW_MAP,
    default_preds_path,
    derive_follow_label,
    load_outcome_map,
    make_prompt_no_evidence,
    read_csv_rows,
    sample_key,
    write_json,
    write_jsonl,
)
from mask_utils import UNKNOWN_ANSWER
from model_registry import prefix_model_relative_path
from resume_utils import ResumeTracker, build_resume_dir, build_resume_scope


def build_rows(split_rows, split_name, position, outcome_map, tracker=None):
    rows = []
    matched = 0
    resume_file = f"{split_name}_samples.jsonl"
    existing_records = tracker.read_records(resume_file) if tracker else []
    existing_sample_keys = set()
    for record in existing_records:
        sample_key_value = str(record.get("sample_key", "")).strip()
        if sample_key_value:
            existing_sample_keys.add(sample_key_value)
        rows.extend(record.get("rows", []))
        if record.get("matched"):
            matched += 1

    for row in split_rows:
        img_id = row.get("img_id", "")
        question = row.get("question", "")
        gold = row.get("gold_norm") or row.get("gold") or row.get("answer") or ""
        wrong = row.get("unknown_target") or UNKNOWN_ANSWER
        key = sample_key(img_id, question)
        sample_resume_key = f"{split_name}:{key}"
        if tracker and (tracker.is_done(sample_resume_key) or key in existing_sample_keys):
            continue
        outcome_rec = outcome_map.get(key)
        if outcome_rec:
            matched += 1
        follow_label = derive_follow_label(outcome_rec, gold, wrong)
        prompt_text = make_prompt_no_evidence(question)

        shared = {
            "sample_id": str(row.get("id", key)),
            "sample_key": key,
            "img_id": img_id,
            "image_path": row.get("image_path", ""),
            "detection_path": row.get("detection_path", ""),
            "mask_path": row.get("mask_path", ""),
            "ic_target_labels": row.get("ic_target_labels", "[]"),
            "category": row.get("category", ""),
            "question": question,
            "split": split_name,
            "position": position,
            "gold_answer": gold,
            "wrong_answer": wrong,
            "baseline_pred": outcome_rec.get("nc_pred") if outcome_rec else None,
            "support_pred": outcome_rec.get("support_pred") if outcome_rec else None,
            "conflict_pred": outcome_rec.get("conflict_pred") if outcome_rec else None,
            "follow_label": follow_label,
            "follow_target": FOLLOW_MAP[follow_label],
            "hallucination_label": follow_label,
            "hallucination_target": FOLLOW_MAP[follow_label],
        }

        sample_rows = [
            {
                **shared,
                "record_id": f"{shared['sample_id']}:nc",
                "prompt_type": "nc",
                "image_variant": "nc",
                "prompt_text": prompt_text,
                "conflict_target": 0,
            },
            {
                **shared,
                "record_id": f"{shared['sample_id']}:ic",
                "prompt_type": "ic",
                "image_variant": "ic",
                "prompt_text": prompt_text,
                "conflict_target": 1,
            },
        ]
        rows.extend(sample_rows)
        if tracker:
            tracker.append_record(
                resume_file,
                {
                    "sample_key": key,
                    "matched": bool(outcome_rec),
                    "rows": sample_rows,
                },
            )
            existing_sample_keys.add(key)
            tracker.mark_done(sample_resume_key, {"split": split_name, "sample_key": key})
    return rows, matched


def summarize(rows, matched_samples, total_samples):
    summary = {
        "n_records": len(rows),
        "n_nc": 0,
        "n_ic": 0,
        "matched_outcome_samples": matched_samples,
        "total_samples": total_samples,
        "hallucination_counts": {"hallucination": 0, "unknown": 0},
    }
    for row in rows:
        if row["prompt_type"] == "nc":
            summary["n_nc"] += 1
        else:
            summary["n_ic"] += 1
            summary["hallucination_counts"][row["hallucination_label"]] += 1
    return summary


def main():
    ap = argparse.ArgumentParser(description="Prepare image-conflict probe manifests.")
    ap.add_argument("--train_csv", default="data/slake_nc_correct_ic_ready_train.csv")
    ap.add_argument("--val_csv", default="data/slake_nc_correct_ic_ready_val.csv")
    ap.add_argument("--model", default="hulumed-4b")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--preds_jsonl", default="")
    ap.add_argument("--position", default="image_conflict")
    ap.add_argument("--out_dir", default="probe/data")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_csv = root / args.train_csv
    val_csv = root / args.val_csv
    preds_jsonl = Path(args.preds_jsonl) if args.preds_jsonl else default_preds_path(
        root,
        args.position,
        model_name=args.model_name,
        model=args.model,
    )

    train_split_rows = read_csv_rows(train_csv)
    val_split_rows = read_csv_rows(val_csv)
    outcome_map = load_outcome_map(preds_jsonl)
    resume_scope = build_resume_scope(train_csv, val_csv, preds_jsonl)

    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            root,
            task_name="probe_prepare_data",
            model_name=args.model_name,
            model=args.model,
            position=args.position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        model=args.model,
        model_name=args.model_name,
        position=args.position,
        preds_jsonl=str(preds_jsonl),
        out_dir=str(args.out_dir),
    )

    train_rows, train_matched = build_rows(train_split_rows, "train", args.position, outcome_map, tracker=tracker)
    tracker.update(train_records=len(train_rows), train_matched=train_matched)
    val_rows, val_matched = build_rows(val_split_rows, "val", args.position, outcome_map, tracker=tracker)
    tracker.update(val_records=len(val_rows), val_matched=val_matched)

    out_dir = root / prefix_model_relative_path(args.out_dir, model_name=args.model_name, model=args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"train_{args.position}.jsonl"
    val_path = out_dir / f"val_{args.position}.jsonl"
    summary_path = out_dir / f"summary_{args.position}.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_json(
        summary_path,
        {
            "model": args.model,
            "model_name": args.model_name,
            "position": args.position,
            "preds_jsonl": str(preds_jsonl),
            "train": summarize(train_rows, train_matched, len(train_split_rows)),
            "val": summarize(val_rows, val_matched, len(val_split_rows)),
            "train_path": str(train_path),
            "val_path": str(val_path),
        },
    )

    print(f"saved_train={train_path}")
    print(f"saved_val={val_path}")
    print(f"saved_summary={summary_path}")
    tracker.finish(
        train_path=str(train_path),
        val_path=str(val_path),
        summary_path=str(summary_path),
        train_records=len(train_rows),
        val_records=len(val_rows),
    )


if __name__ == "__main__":
    main()
