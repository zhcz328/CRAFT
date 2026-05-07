#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keep only DriftMed pairs that satisfy:
1) pair type is (Yes, No): correct_pred == "yes" and wrong_pred == "no"
2) margin is large: delta_correct > a and delta_wrong < -b

Input: logit results jsonl (one record per line, 1000 lines for 500 pairs)
Output: jsonl where each line is one kept pair (correct + wrong packed together)

Example:
  python filter_pairs_margin.py \
    --in 36a99e5b-147d-444e-a337-9107be92c3b6.jsonl \
    --out kept_pairs.jsonl \
    --a 8 --b 8
"""

import argparse
import json

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_pairs(rows, offset):
    by_id = {r["id"]: r for r in rows if isinstance(r.get("id"), int)}
    pairs = []
    for r in rows:
        if r.get("label") != "correct":
            continue
        cid = r["id"]
        wid = cid + offset
        w = by_id.get(wid)
        if w and w.get("label") == "wrong":
            pairs.append((r, w))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default="kept_pairs.jsonl")
    ap.add_argument("--offset", type=int, default=2145)
    ap.add_argument("--a", type=float, required=True, help="keep if delta_correct > a")
    ap.add_argument("--b", type=float, required=True, help="keep if delta_wrong < -b")
    ap.add_argument("--topk", type=int, default=0, help="keep only top-k most confident after sorting (0 = keep all)")
    args = ap.parse_args()

    rows = read_jsonl(args.in_path)
    pairs = build_pairs(rows, args.offset)

    kept = []
    for c, w in pairs:
        c_pred = (c.get("pred") or "").strip().lower()
        w_pred = (w.get("pred") or "").strip().lower()
        c_delta = float(c.get("delta", 0.0))
        w_delta = float(w.get("delta", 0.0))

        if c_pred == "yes" and w_pred == "no" and (c_delta > args.a) and (w_delta < -args.b):
            kept.append((c, w))

    # Sort by "confidence": bigger correct delta and more negative wrong delta first
    kept.sort(key=lambda cw: (cw[0].get("delta", 0.0), -cw[1].get("delta", 0.0)), reverse=True)

    if args.topk and args.topk > 0:
        kept = kept[:args.topk]

    with open(args.out_path, "w", encoding="utf-8") as f:
        for c, w in kept:
            rec = {
                "pair_id": c["id"],
                "offset": w["id"] - c["id"],
                "a": args.a,
                "b": args.b,
                "correct": {
                    "id": c["id"],
                    "label": c.get("label"),
                    "gold": c.get("gold"),
                    "pred": c.get("pred"),
                    "delta": c.get("delta"),
                    "yes_lp": c.get("yes_lp"),
                    "no_lp": c.get("no_lp"),
                    "bias_type": c.get("bias_type"),
                    "change_category": c.get("change_category"),
                    "disease": c.get("disease"),
                    "prompt": c.get("prompt"),
                    "raw": c.get("raw"),
                },
                "wrong": {
                    "id": w["id"],
                    "label": w.get("label"),
                    "gold": w.get("gold"),
                    "pred": w.get("pred"),
                    "delta": w.get("delta"),
                    "yes_lp": w.get("yes_lp"),
                    "no_lp": w.get("no_lp"),
                    "bias_type": w.get("bias_type"),
                    "change_category": w.get("change_category"),
                    "disease": w.get("disease"),
                    "prompt": w.get("prompt"),
                    "raw": w.get("raw"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"loaded rows={len(rows)} pairs={len(pairs)} kept={len(kept)} -> {args.out_path}")

if __name__ == "__main__":
    main()
