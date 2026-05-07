#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Split kept_pairs jsonl into train/val sets.")
    ap.add_argument(
        "--in_path",
        default="data/kept_pairs_a12_b10_all.jsonl",
        help="Input jsonl file path.",
    )
    ap.add_argument(
        "--train_out",
        default="data/kept_pairs_a12_b10_all_train.jsonl",
        help="Output train jsonl path.",
    )
    ap.add_argument(
        "--val_out",
        default="data/kept_pairs_a12_b10_all_val.jsonl",
        help="Output validation jsonl path.",
    )
    ap.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio in (0, 1). Default: 0.9",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable shuffling before split.",
    )
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")

    base_dir = Path(__file__).resolve().parent

    in_path = Path(args.in_path)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)

    if not in_path.is_absolute():
        in_path = base_dir / in_path
    if not train_out.is_absolute():
        train_out = base_dir / train_out
    if not val_out.is_absolute():
        val_out = base_dir / val_out

    rows = read_jsonl(in_path)
    n = len(rows)
    if n == 0:
        raise ValueError(f"Input is empty: {in_path}")

    if args.no_shuffle:
        data = rows
    else:
        data = rows[:]
        random.Random(args.seed).shuffle(data)

    n_train = int(n * args.train_ratio)
    n_train = max(1, min(n - 1, n_train))

    train_rows = data[:n_train]
    val_rows = data[n_train:]

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)

    print(
        f"total={n} train={len(train_rows)} val={len(val_rows)} "
        f"ratio={args.train_ratio} seed={args.seed} shuffle={not args.no_shuffle}"
    )
    print(f"train_out={train_out}")
    print(f"val_out={val_out}")


if __name__ == "__main__":
    main()
