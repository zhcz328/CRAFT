import argparse
from pathlib import Path
import random

from common import (
    FOLLOW_MAP,
    YES,
    derive_follow_label,
    load_outcome_map,
    make_conflict_prompt,
    make_support_prompt,
    opposite_label,
    read_jsonl,
    write_json,
    write_jsonl,
)
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from yesno_utils import get_field, load_records, normalize_context  # noqa: E402


def build_rows(records, split_name, position, outcome_map, q_field, c_field, y_field, id_field):
    rows = []
    for idx, record in enumerate(records):
        pair_id = get_field(record, id_field) if id_field else record.get("id", idx)
        question = get_field(record, q_field)
        context = normalize_context(get_field(record, c_field))
        gold = str(get_field(record, y_field)).strip().lower()
        if question is None or gold not in ("yes", "no"):
            continue

        outcome_rec = outcome_map.get(pair_id)
        follow_label = derive_follow_label(outcome_rec["conflict"]["pred"], gold) if outcome_rec else "unknown"
        shared = {
            "sample_id": str(pair_id),
            "pair_id": pair_id,
            "split": split_name,
            "position": position,
            "gold_answer": gold,
            "conflict_answer": opposite_label(gold),
            "base_prompt": question,
            "baseline_pred": outcome_rec["base"]["pred"] if outcome_rec else None,
            "support_pred": outcome_rec["support"]["pred"] if outcome_rec else None,
            "conflict_pred": outcome_rec["conflict"]["pred"] if outcome_rec else None,
            "follow_label": follow_label,
            "follow_target": FOLLOW_MAP[follow_label],
        }

        rows.append(
            {
                **shared,
                "record_id": f"{pair_id}:support",
                "prompt_type": "support",
                "prompt_text": make_support_prompt(str(question), context, gold, position, allow_unknown=False),
                "conflict_target": 0,
            }
        )
        rows.append(
            {
                **shared,
                "record_id": f"{pair_id}:conflict",
                "prompt_type": "conflict",
                "prompt_text": make_conflict_prompt(str(question), context, gold, position, allow_unknown=False),
                "conflict_target": 1,
            }
        )
    return rows


def summarize(rows):
    summary = {
        "n_records": len(rows),
        "n_support": 0,
        "n_conflict": 0,
        "follow_counts": {"resist": 0, "follow_conflict": 0, "unknown": 0},
    }
    for row in rows:
        if row["prompt_type"] == "support":
            summary["n_support"] += 1
        else:
            summary["n_conflict"] += 1
            summary["follow_counts"][row["follow_label"]] += 1
    return summary


def stratified_split(records, outcome_map, y_field, id_field, val_ratio, seed):
    groups = {"resist": [], "follow_conflict": [], "unknown": []}
    for idx, record in enumerate(records):
        pair_id = get_field(record, id_field) if id_field else record.get("id", idx)
        gold = str(get_field(record, y_field)).strip().lower()
        if gold not in ("yes", "no"):
            continue
        outcome_rec = outcome_map.get(pair_id)
        follow_label = derive_follow_label(outcome_rec["conflict"]["pred"], gold) if outcome_rec else "unknown"
        groups.setdefault(follow_label, []).append(record)

    rng = random.Random(seed)
    train_records = []
    val_records = []
    for items in groups.values():
        bucket = items[:]
        rng.shuffle(bucket)
        n_val = int(round(len(bucket) * val_ratio))
        if len(bucket) > 1:
            n_val = max(1, min(len(bucket) - 1, n_val))
        else:
            n_val = 0
        val_records.extend(bucket[:n_val])
        train_records.extend(bucket[n_val:])
    return train_records, val_records


def main():
    ap = argparse.ArgumentParser(description="Prepare prompt-level manifests for PubMedQA probe training.")
    ap.add_argument("--train_records", default="data/train_yesno.train.json")
    ap.add_argument("--val_records", default="data/train_yesno.val.json")
    ap.add_argument("--conflict_positions", default="result_all_positions_yesno/conflict_positions.jsonl")
    ap.add_argument("--position", default="before_question")
    ap.add_argument("--out_dir", default="probe/data")
    ap.add_argument("--pair_val_ratio", type=float, default=0.1)
    ap.add_argument("--pair_split_seed", type=int, default=42)
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="pubid")
    args = ap.parse_args()

    train_records = load_records(str(ROOT_DIR / args.train_records), fmt="auto")
    val_records = load_records(str(ROOT_DIR / args.val_records), fmt="auto")
    outcome_map = load_outcome_map(ROOT_DIR / args.conflict_positions, args.position)

    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_rows(train_records, "train", args.position, outcome_map, args.q_field, args.c_field, args.y_field, args.id_field)
    val_rows = build_rows(val_records, "val", args.position, outcome_map, args.q_field, args.c_field, args.y_field, args.id_field)
    train_path = out_dir / f"train_{args.position}.jsonl"
    val_path = out_dir / f"val_{args.position}.jsonl"
    summary_path = out_dir / f"summary_{args.position}.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_json(summary_path, {"position": args.position, "train": summarize(train_rows), "val": summarize(val_rows)})

    all_records = list(train_records) + list(val_records)
    strat_train_records, strat_val_records = stratified_split(
        all_records,
        outcome_map,
        args.y_field,
        args.id_field,
        args.pair_val_ratio,
        args.pair_split_seed,
    )
    strat_train_rows = build_rows(strat_train_records, "train_pair_stratified", args.position, outcome_map, args.q_field, args.c_field, args.y_field, args.id_field)
    strat_val_rows = build_rows(strat_val_records, "val_pair_stratified", args.position, outcome_map, args.q_field, args.c_field, args.y_field, args.id_field)
    strat_train_path = out_dir / f"train_pair_stratified_{args.position}.jsonl"
    strat_val_path = out_dir / f"val_pair_stratified_{args.position}.jsonl"
    strat_summary_path = out_dir / f"summary_pair_stratified_{args.position}.json"
    write_jsonl(strat_train_path, strat_train_rows)
    write_jsonl(strat_val_path, strat_val_rows)
    write_json(strat_summary_path, {"position": args.position, "train": summarize(strat_train_rows), "val": summarize(strat_val_rows)})

    print(f"saved_train={train_path}")
    print(f"saved_val={val_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_pair_stratified_train={strat_train_path}")
    print(f"saved_pair_stratified_val={strat_val_path}")
    print(f"saved_pair_stratified_summary={strat_summary_path}")


if __name__ == "__main__":
    main()
