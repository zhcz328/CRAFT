import argparse
from pathlib import Path

from tqdm import tqdm

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


def build_rows(split_rows, split_name, position, outcome_map):
    rows = []
    matched = 0
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
        for prompt_type, prompt_text in prompts.items():
            rows.append(
                {
                    **shared,
                    "record_id": f"{shared['sample_id']}:{prompt_type}",
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text,
                }
            )
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
    ap.add_argument("--train_csv", default="data/nc_cc_both_correct_rerun_tmp_train.csv")
    ap.add_argument("--val_csv", default="data/nc_cc_both_correct_rerun_tmp_val.csv")
    ap.add_argument("--preds_jsonl", default="")
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--out_dir", default="tuned_lens/data")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_csv = root / args.train_csv
    val_csv = root / args.val_csv
    preds_jsonl = Path(args.preds_jsonl) if args.preds_jsonl else default_preds_path(root, args.position)

    train_split_rows = read_csv_rows(train_csv)
    val_split_rows = read_csv_rows(val_csv)
    outcome_map = load_outcome_map(preds_jsonl)

    train_rows, train_matched = build_rows(train_split_rows, "train", args.position, outcome_map)
    val_rows, val_matched = build_rows(val_split_rows, "val", args.position, outcome_map)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"train_{args.position}.jsonl"
    val_path = out_dir / f"val_{args.position}.jsonl"
    summary_path = out_dir / f"summary_{args.position}.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_json(
        summary_path,
        {
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


if __name__ == "__main__":
    main()
