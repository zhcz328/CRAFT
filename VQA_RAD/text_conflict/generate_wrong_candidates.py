#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd


def norm_text(x: str) -> str:
    return str(x).strip().lower()


def ensure_category(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "category" not in out.columns:
        if "content_type" in out.columns:
            out["category"] = out["content_type"]
        else:
            out["category"] = "other"
    out["category"] = out["category"].fillna("other")
    return out


def build_by_cat_sorted(df: pd.DataFrame, answer_col: str) -> Dict[str, List[str]]:
    pool = defaultdict(list)
    for _, row in df.iterrows():
        cat = norm_text(row["category"])
        ans = norm_text(row[answer_col])
        if ans:
            pool[cat].append(ans)

    by_cat_sorted: Dict[str, List[str]] = {}
    for cat, answers in pool.items():
        counts = Counter(answers)
        by_cat_sorted[cat] = [answer for answer, _ in counts.most_common()]
    return by_cat_sorted


def choose_wrong_answer(gold: str, cat: str, by_cat_sorted: Dict[str, List[str]]) -> str:
    gold_norm = norm_text(gold)
    cat_norm = norm_text(cat)

    if gold_norm == "yes":
        return "no"
    if gold_norm == "no":
        return "yes"

    for cand in by_cat_sorted.get(cat_norm, []):
        cand_norm = norm_text(cand)
        if cand_norm and cand_norm != gold_norm and cand_norm != "unknown":
            return cand_norm

    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Prepared VQA-RAD CSV without wrong column.")
    ap.add_argument("--out_csv", required=True, help="Output CSV with generated wrong column.")
    ap.add_argument(
        "--out_xlsx",
        default="",
        help="Optional Excel export for manual wrong-answer edits.",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    df = pd.read_csv(in_csv)
    df = ensure_category(df)

    if "gold" in df.columns:
        answer_col = "gold"
    elif "answer" in df.columns:
        answer_col = "answer"
    else:
        raise ValueError("Input CSV must contain either 'gold' or 'answer'.")

    by_cat_sorted = build_by_cat_sorted(df, answer_col=answer_col)

    out_df = df.copy()
    out_df["gold"] = out_df[answer_col].astype(str).map(norm_text)
    out_df["wrong"] = out_df.apply(
        lambda row: choose_wrong_answer(row["gold"], row["category"], by_cat_sorted),
        axis=1,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else out_csv.with_suffix(".xlsx")
    out_df.to_excel(out_xlsx, index=False)

    yes_no_count = int(out_df["gold"].isin(["yes", "no"]).sum())
    unknown_wrong_count = int((out_df["wrong"] == "unknown").sum())

    print("Done.")
    print(f"Input rows: {len(out_df)}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved XLSX: {out_xlsx}")
    print(f"Yes/No rows: {yes_no_count}")
    print(f"Rows with wrong=unknown: {unknown_wrong_count}")


if __name__ == "__main__":
    main()

