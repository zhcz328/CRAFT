import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_data(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        rows: List[Dict[str, Any]] = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rows.append({"id": k, **v})
            else:
                rows.append({"id": k, "value": v})
        return rows
    raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def count_by_field(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    cnt: Dict[str, int] = {}
    for r in rows:
        key = str(r.get(field, "MISSING"))
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = len(rows)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    train_n = int(n * train_ratio)
    if n >= 2 and 0.0 < train_ratio < 1.0:
        train_n = max(1, min(n - 1, train_n))

    train_idx = set(idx[:train_n])
    train = [rows[i] for i in range(n) if i in train_idx]
    val = [rows[i] for i in range(n) if i not in train_idx]
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser(description="Randomly split PubMedQA filtered data into train/val by ratio.")
    ap.add_argument(
        "--input",
        default="pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json",
        help="Input dataset file (.json/.jsonl).",
    )
    ap.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio in [0,1].")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--label_field", default="final_decision", help="Field used for split statistics only.")
    ap.add_argument("--out_train", default="", help="Output train path (default: <input>.train.json).")
    ap.add_argument("--out_val", default="", help="Output val path (default: <input>.val.json).")
    ap.add_argument("--out_meta", default="", help="Output meta path (default: <input>.split_meta.json).")
    args = ap.parse_args()

    if not (0.0 <= args.train_ratio <= 1.0):
        raise SystemExit("--train_ratio must be within [0, 1].")

    in_path = Path(args.input)
    rows = load_data(in_path)
    if not rows:
        raise SystemExit(f"Input is empty: {in_path}")

    train, val = split_rows(rows, args.train_ratio, args.seed)

    stem = in_path.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]

    out_train = Path(args.out_train) if args.out_train else in_path.with_name(f"{stem}.train.json")
    out_val = Path(args.out_val) if args.out_val else in_path.with_name(f"{stem}.val.json")
    out_meta = Path(args.out_meta) if args.out_meta else in_path.with_name(f"{stem}.split_meta.json")

    dump_json(out_train, train)
    dump_json(out_val, val)

    meta = {
        "input": str(in_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "total": len(rows),
        "train_count": len(train),
        "val_count": len(val),
        "label_field": args.label_field,
        "label_dist_total": count_by_field(rows, args.label_field),
        "label_dist_train": count_by_field(train, args.label_field),
        "label_dist_val": count_by_field(val, args.label_field),
        "out_train": str(out_train),
        "out_val": str(out_val),
    }
    dump_json(out_meta, meta)

    print(f"[OK] total={len(rows)} train={len(train)} val={len(val)} seed={args.seed}")
    print(f"[OK] train -> {out_train}")
    print(f"[OK] val   -> {out_val}")
    print(f"[OK] meta  -> {out_meta}")


if __name__ == "__main__":
    main()
