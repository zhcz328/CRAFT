import argparse
import sys
from pathlib import Path

from tqdm import tqdm

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import (
    FOLLOW_MAP,
    default_preds_path,
    derive_follow_label,
    load_outcome_map,
    make_prompt_no_evidence,
    make_prompt_with_evidence,
    read_csv_rows,
    sample_key,
    write_json,
    write_jsonl,
)
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
    progress = tqdm(
        split_rows,
        total=len(split_rows),
        desc=f"build {split_name} manifests",
        unit="sample",
    )
    for row in progress:
        img_id = row.get("img_id", "")
        question = row.get("question", "")
        gold = row.get("gold_norm") or row.get("gold") or ""
        wrong = row.get("wrong_norm") or row.get("wrong") or ""
        key = sample_key(img_id, question)
        sample_resume_key = f"{split_name}:{key}"
        if tracker and (tracker.is_done(sample_resume_key) or key in existing_sample_keys):
            continue
        outcome_rec = outcome_map.get(key)
        if outcome_rec:
            matched += 1
        follow_label = derive_follow_label(outcome_rec, gold, wrong)

        shared = {
            "sample_id": str(row.get("id", key)),
            "sample_key": key,
            "img_id": img_id,
            "image_path": row.get("image_path", ""),
            "category": row.get("category", ""),
            "question": question,
            "split": split_name,
            "position": position,
            "gold_answer": gold,
            "wrong_answer": wrong,
            "follow_label": follow_label,
            "follow_target": FOLLOW_MAP[follow_label],
            "baseline_pred": outcome_rec.get("nc_pred") if outcome_rec else None,
            "support_pred": outcome_rec.get("support_pred") if outcome_rec else None,
            "conflict_pred": outcome_rec.get("conflict_pred") if outcome_rec else None,
        }

        prompts = {
            "base": make_prompt_no_evidence(question),
            "support": make_prompt_with_evidence(question, gold, position),
            "conflict": make_prompt_with_evidence(question, wrong, position),
        }
        sample_rows = []
        for prompt_type, prompt_text in prompts.items():
            sample_rows.append(
                {
                    **shared,
                    "record_id": f"{shared['sample_id']}:{prompt_type}",
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text,
                }
            )
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
        progress.set_postfix(records=len(rows), matched=matched)
    return rows, matched


def summarize(rows, matched_samples, total_samples):
    summary = {
        "n_records": len(rows),
        "prompt_type_counts": {"base": 0, "support": 0, "conflict": 0},
        "matched_outcome_samples": matched_samples,
        "total_samples": total_samples,
        "follow_counts_conflict_only": {"resist": 0, "follow_conflict": 0, "unknown": 0},
    }
    for row in rows:
        summary["prompt_type_counts"][row["prompt_type"]] += 1
        if row["prompt_type"] == "conflict":
            summary["follow_counts_conflict_only"][row["follow_label"]] += 1
    return summary


def main():
    ap = argparse.ArgumentParser(description="Prepare Hulu-med tuned-lens manifests.")
    ap.add_argument("--train_csv", default="")
    ap.add_argument("--val_csv", default="")
    ap.add_argument("--model", default="hulumed-4b")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--preds_jsonl", default="")
    ap.add_argument("--dataset_name", default="gqa")
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--out_dir", default="tuned_lens/data")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_name = str(args.dataset_name or "gqa").strip().lower()
    train_csv_arg = args.train_csv.strip() or f"data/{dataset_name}_nc_cc_both_correct_train.csv"
    val_csv_arg = args.val_csv.strip() or f"data/{dataset_name}_nc_cc_both_correct_val.csv"
    train_csv = root / train_csv_arg
    val_csv = root / val_csv_arg
    preds_jsonl = Path(args.preds_jsonl) if args.preds_jsonl else default_preds_path(
        root,
        args.position,
        model_name=args.model_name,
        model=args.model,
        dataset_name=dataset_name,
    )

    train_split_rows = read_csv_rows(train_csv)
    val_split_rows = read_csv_rows(val_csv)
    outcome_map = load_outcome_map(preds_jsonl)
    resume_scope = build_resume_scope(train_csv, val_csv, preds_jsonl)

    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            root,
            task_name="tuned_lens_prepare_data",
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
        dataset_name=dataset_name,
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
