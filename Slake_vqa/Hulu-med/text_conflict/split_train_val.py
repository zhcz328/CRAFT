#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import pandas as pd


def build_default_paths(in_csv: Path):
    stem = in_csv.stem
    parent = in_csv.parent
    train_path = parent / f"{stem}_train.csv"
    val_path = parent / f"{stem}_val.csv"
    return train_path, val_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV to split")
    ap.add_argument("--out_train", default="", help="Output train CSV path")
    ap.add_argument("--out_val", default="", help="Output val CSV path")
    ap.add_argument("--val_percent", type=float, default=20.0, help="Validation set percentage, e.g. 20 means 20%")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    if args.val_percent <= 0 or args.val_percent >= 100:
        raise ValueError("--val_percent must be in (0, 100)")

    df = pd.read_csv(in_csv)
    n = len(df)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split, got {n}")

    val_count = int(round(n * args.val_percent / 100.0))
    val_count = max(1, min(val_count, n - 1))

    val_idx = df.sample(n=val_count, random_state=args.seed).index
    val_df = df.loc[val_idx].reset_index(drop=True)
    train_df = df.drop(val_idx).reset_index(drop=True)

    default_train, default_val = build_default_paths(in_csv)
    out_train = Path(args.out_train) if args.out_train else default_train
    out_val = Path(args.out_val) if args.out_val else default_val

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)

    print("Done.")
    print(f"Input: {in_csv}")
    print(f"Total rows: {n}")
    print(f"Val percent: {args.val_percent}%")
    print(f"Train rows: {len(train_df)} -> {out_train}")
    print(f"Val rows: {len(val_df)} -> {out_val}")


if __name__ == "__main__":
    main()

