import argparse
from pathlib import Path
import random
from collections import defaultdict

from common import (
    FOLLOW_MAP,
    YES,
    derive_follow_label,
    gold_label_for_side,
    inject_evidence,
    load_outcome_map,
    make_evidence,
    opposite_label,
    read_jsonl,
    write_json,
    write_jsonl,
)


def build_rows(pairs, split_name, position, outcome_map):
    rows = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        for side in ("correct", "wrong"):
            side_rec = pair[side]
            base_prompt = side_rec["prompt"]
            gold = gold_label_for_side(side)
            conflict = opposite_label(gold)

            outcome_rec = outcome_map.get((pair_id, side))
            follow_label = derive_follow_label(outcome_rec)

            shared = {
                "sample_id": f"{pair_id}:{side}",
                "pair_id": pair_id,
                "side": side,
                "split": split_name,
                "position": position,
                "gold_answer": gold,
                "conflict_answer": conflict,
                "base_prompt": base_prompt,
                "baseline_pred": outcome_rec["base"]["pred"] if outcome_rec else None,
                "support_pred": outcome_rec["support"]["pred"] if outcome_rec else None,
                "conflict_pred": outcome_rec["conflict"]["pred"] if outcome_rec else None,
                "follow_label": follow_label,
                "follow_target": FOLLOW_MAP[follow_label],
            }

            support_prompt = inject_evidence(base_prompt, make_evidence(gold == YES), position)
            conflict_prompt = inject_evidence(base_prompt, make_evidence(conflict == YES), position)

            rows.append(
                {
                    **shared,
                    "record_id": f"{pair_id}:{side}:support",
                    "prompt_type": "support",
                    "prompt_text": support_prompt,
                    "conflict_target": 0,
                }
            )
            rows.append(
                {
                    **shared,
                    "record_id": f"{pair_id}:{side}:conflict",
                    "prompt_type": "conflict",
                    "prompt_text": conflict_prompt,
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
        if row["prompt_type"] == "conflict":
            summary["follow_counts"][row["follow_label"]] += 1
    return summary


def pair_category(pair, outcome_map):
    labels = set()
    for side in ("correct", "wrong"):
        labels.add(derive_follow_label(outcome_map.get((pair["pair_id"], side))))
    return "+".join(sorted(labels))


def stratified_pair_split(pairs, outcome_map, val_ratio, seed):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair_category(pair, outcome_map)].append(pair)

    rng = random.Random(seed)
    train_pairs = []
    val_pairs = []
    for _, group_pairs in groups.items():
        group_pairs = group_pairs[:]
        rng.shuffle(group_pairs)
        n_val = int(round(len(group_pairs) * val_ratio))
        if len(group_pairs) > 1:
            n_val = max(1, min(len(group_pairs) - 1, n_val))
        else:
            n_val = 0
        val_pairs.extend(group_pairs[:n_val])
        train_pairs.extend(group_pairs[n_val:])
    return train_pairs, val_pairs


def main():
    ap = argparse.ArgumentParser(description="Prepare prompt-level manifests for probe training.")
    ap.add_argument("--train_pairs", default="data/kept_pairs_a12_b10_all_train.jsonl")
    ap.add_argument("--val_pairs", default="data/kept_pairs_a12_b10_all_val.jsonl")
    ap.add_argument("--conflict_positions", default="result_all_positions/conflict_positions.jsonl")
    ap.add_argument("--position", default="before_question")
    ap.add_argument("--out_dir", default="probe/data")
    ap.add_argument("--pair_val_ratio", type=float, default=0.1)
    ap.add_argument("--pair_split_seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_pairs = read_jsonl(root / args.train_pairs)
    val_pairs = read_jsonl(root / args.val_pairs)
    outcome_map = load_outcome_map(root / args.conflict_positions, args.position)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_rows(train_pairs, "train", args.position, outcome_map)
    val_rows = build_rows(val_pairs, "val", args.position, outcome_map)

    train_path = out_dir / f"train_{args.position}.jsonl"
    val_path = out_dir / f"val_{args.position}.jsonl"
    summary_path = out_dir / f"summary_{args.position}.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_json(
        summary_path,
        {
            "position": args.position,
            "train": summarize(train_rows),
            "val": summarize(val_rows),
            "train_path": str(train_path),
            "val_path": str(val_path),
        },
    )

    all_pairs = train_pairs + val_pairs
    strat_train_pairs, strat_val_pairs = stratified_pair_split(
        all_pairs,
        outcome_map,
        val_ratio=args.pair_val_ratio,
        seed=args.pair_split_seed,
    )
    strat_train_rows = build_rows(strat_train_pairs, "train_pair_stratified", args.position, outcome_map)
    strat_val_rows = build_rows(strat_val_pairs, "val_pair_stratified", args.position, outcome_map)
    strat_train_path = out_dir / f"train_pair_stratified_{args.position}.jsonl"
    strat_val_path = out_dir / f"val_pair_stratified_{args.position}.jsonl"
    strat_summary_path = out_dir / f"summary_pair_stratified_{args.position}.json"
    write_jsonl(strat_train_path, strat_train_rows)
    write_jsonl(strat_val_path, strat_val_rows)
    write_json(
        strat_summary_path,
        {
            "position": args.position,
            "pair_val_ratio": args.pair_val_ratio,
            "pair_split_seed": args.pair_split_seed,
            "train": summarize(strat_train_rows),
            "val": summarize(strat_val_rows),
            "train_path": str(strat_train_path),
            "val_path": str(strat_val_path),
        },
    )

    print(f"saved_train={train_path}")
    print(f"saved_val={val_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_pair_stratified_train={strat_train_path}")
    print(f"saved_pair_stratified_val={strat_val_path}")
    print(f"saved_pair_stratified_summary={strat_summary_path}")


if __name__ == "__main__":
    main()
