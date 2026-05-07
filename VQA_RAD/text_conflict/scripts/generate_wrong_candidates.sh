#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IN_CSV="${IN_CSV:-../data/vqa_rad_closed_all.csv}"
OUT_CSV="${OUT_CSV:-./data/vqa_rad_closed_with_wrong.csv}"
OUT_XLSX="${OUT_XLSX:-}"

OUT_XLSX_ARGS=()
if [[ -n "$OUT_XLSX" ]]; then
  OUT_XLSX_ARGS+=(--out_xlsx "$OUT_XLSX")
fi

python generate_wrong_candidates.py \
  --in_csv "$IN_CSV" \
  --out_csv "$OUT_CSV" \
  "${OUT_XLSX_ARGS[@]}"
