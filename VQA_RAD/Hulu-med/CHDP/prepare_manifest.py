import argparse
from pathlib import Path

from common import (
    LABEL_MAP,
    default_preds_path,
    derive_binary_label,
    load_outcome_map,
    make_prompt_no_evidence,
    make_prompt_with_evidence,
    read_csv_rows,
    sample_key,
    write_json,
    write_jsonl,
)


DEFAULT_TRAIN_CSV = "data/nc_cc_both_correct_rerun_tmp_train.csv"
DEFAULT_VAL_CSV = "data/nc_cc_both_correct_rerun_tmp_val.csv"


def normalize_split_rows(split_rows, split_name, position, outcome_map):
    kept_rows = []
    stats = {
        "total_rows": len(split_rows),
        "matched_outcome_rows": 0,
        "kept_rows": 0,
        "kept_resist": 0,
        "kept_hijack": 0,
        "dropped_no_outcome": 0,
        "dropped_nc_not_gold": 0,
        "dropped_ic_other": 0,
    }

    for row in split_rows:
        img_id = row.get("img_id", "")
        question = row.get("question", "")
        gold = row.get("gold_norm") or row.get("gold") or ""
        conflict = row.get("wrong_norm") or row.get("wrong") or ""
        key = sample_key(img_id, question)
        outcome = outcome_map.get(key)
        if outcome is None:
            stats["dropped_no_outcome"] += 1
            continue

        stats["matched_outcome_rows"] += 1
        label_name = derive_binary_label(outcome, gold, conflict)
        if label_name is None:
            if outcome.get("nc_pred", "") != str(gold).strip().lower():
                stats["dropped_nc_not_gold"] += 1
            else:
                stats["dropped_ic_other"] += 1
            continue

        sample_id = str(row.get("id", key))
        kept_rows.append(
            {
                "sample_id": sample_id,
                "sample_key": key,
                "split": split_name,
                "position": position,
                "img_id": img_id,
                "image_path": row.get("image_path", ""),
                "category": row.get("category", ""),
                "question": question,
                "gold_answer": gold,
                "conflict_answer": conflict,
                "label": LABEL_MAP[label_name],
                "label_name": label_name,
                "nc_pred": outcome.get("nc_pred"),
                "support_pred": outcome.get("support_pred"),
                "conflict_pred": outcome.get("conflict_pred"),
                "nc_prompt_text": make_prompt_no_evidence(question),
                "ic_prompt_text": make_prompt_with_evidence(question, conflict, position),
            }
        )
        stats["kept_rows"] += 1
        stats[f"kept_{label_name}"] += 1

    return kept_rows, stats


def process_split(root, split_name, csv_path, position, outcome_map, out_dir):
    if not csv_path:
        return None, None
    split_rows = read_csv_rows(root / csv_path)
    normalized_rows, summary = normalize_split_rows(split_rows, split_name, position, outcome_map)
    out_path = out_dir / f"{split_name}.jsonl"
    write_jsonl(out_path, normalized_rows)
    return out_path, summary


def main():
    ap = argparse.ArgumentParser(description="Prepare paired NC/IC manifests for CHDP.")
    ap.add_argument("--train_csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--val_csv", default=DEFAULT_VAL_CSV)
    ap.add_argument("--test_csv", default="")
    ap.add_argument("--preds_jsonl", default="")
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--out_dir", default="CHDP/data")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = Path(args.preds_jsonl) if args.preds_jsonl else default_preds_path(root, args.position)
    outcome_map = load_outcome_map(preds_path)

    split_summaries = {}
    written_files = {}
    for split_name, csv_path in (
        ("train", args.train_csv),
        ("val", args.val_csv),
        ("test", args.test_csv),
    ):
        out_path, summary = process_split(root, split_name, csv_path, args.position, outcome_map, out_dir)
        if out_path is not None:
            written_files[split_name] = str(out_path)
            split_summaries[split_name] = summary

    summary_path = out_dir / f"summary_{args.position}.json"
    write_json(
        summary_path,
        {
            "position": args.position,
            "preds_jsonl": str(preds_path),
            "written_files": written_files,
            "splits": split_summaries,
        },
    )

    for split_name, out_path in written_files.items():
        print(f"saved_{split_name}={out_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
