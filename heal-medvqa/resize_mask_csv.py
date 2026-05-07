#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def decode_rle(rle, height: int, width: int) -> np.ndarray:
    mask = np.zeros(height * width, dtype=np.uint8)
    if not rle:
        return mask.reshape((height, width), order="F")
    if len(rle) % 2 != 0:
        raise ValueError(f"Invalid RLE length: {len(rle)}")
    for start, length in zip(rle[0::2], rle[1::2]):
        if length <= 0:
            continue
        s = int(start)
        e = s + int(length)
        mask[s:e] = 1
    return mask.reshape((height, width), order="F")


def encode_rle(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if flat.size == 0:
        return []
    padded = np.concatenate(([0], flat, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    out: list[int] = []
    for start, length in zip(starts.tolist(), lengths.tolist()):
        out.extend([int(start), int(length)])
    return out


def resize_dims(height: int, width: int, max_side: int) -> tuple[int, int]:
    if max(height, width) <= max_side:
        return height, width
    scale = max_side / float(max(height, width))
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    return new_h, new_w


def fixed_dims(side: int) -> tuple[int, int]:
    return side, side


def resize_mask(mask: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    pil = Image.fromarray(mask * 255, mode="L")
    resized = pil.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resize mask RLE fields in a CSV to a new max-side resolution.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_side", type=int, default=672)
    ap.add_argument(
        "--fixed_square",
        action="store_true",
        help="Resize every mask to max_side x max_side instead of preserving aspect ratio.",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    resized = 0
    unchanged = 0
    for row in rows:
        raw_h = row.get("mask_h", "").strip()
        raw_w = row.get("mask_w", "").strip()
        raw_rle = row.get("mask_rle", "").strip()
        if not raw_h or not raw_w:
            unchanged += 1
            continue

        mask_h = int(raw_h)
        mask_w = int(raw_w)
        if args.fixed_square:
            new_h, new_w = fixed_dims(args.max_side)
        else:
            new_h, new_w = resize_dims(mask_h, mask_w, args.max_side)
        if (new_h, new_w) == (mask_h, mask_w):
            unchanged += 1
            continue

        rle = json.loads(raw_rle) if raw_rle else []
        mask = decode_rle(rle, mask_h, mask_w)
        mask_small = resize_mask(mask, new_h, new_w)
        row["mask_rle"] = json.dumps(encode_rle(mask_small), ensure_ascii=False)
        row["mask_h"] = str(new_h)
        row["mask_w"] = str(new_w)
        resized += 1

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "max_side": args.max_side,
        "fixed_square": bool(args.fixed_square),
        "rows": len(rows),
        "resized_rows": resized,
        "unchanged_rows": unchanged,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
