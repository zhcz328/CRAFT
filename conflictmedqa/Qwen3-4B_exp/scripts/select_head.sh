#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"
POSITION="${POSITION:-before_question}"
SUMMARY_PATH="${SUMMARY_PATH:-$(join_output_path "result/${POSITION}/headscan_rounds_top30_inf/head_scan_summary.json")}"
SELECT_OUT_DIR="${SELECT_OUT_DIR:-$(join_output_path "result/${POSITION}/headscan_rounds_top30_inf")}"
SELECT_MODE="${SELECT_MODE:-B}"
ER_QUANTILE="${ER_QUANTILE:-0.75}"
BC_QUANTILE="${BC_QUANTILE:-0.25}"
TOPK_FALLBACK="${TOPK_FALLBACK:-50}"
ER_MIN="${ER_MIN:-3.5}"
BC_MAX="${BC_MAX:-1.5}"
PRINT_TOPN="${PRINT_TOPN:-30}"

mkdir -p "$SELECT_OUT_DIR"

CMD=(
  "$PYTHON_BIN" select_heads.py
  --summary_path "$SUMMARY_PATH"
  --out_dir "$SELECT_OUT_DIR"
  --mode "$SELECT_MODE"
  --er_quantile "$ER_QUANTILE"
  --bc_quantile "$BC_QUANTILE"
  --topk_fallback "$TOPK_FALLBACK"
  --er_min "$ER_MIN"
  --bc_max "$BC_MAX"
  --print_topn "$PRINT_TOPN"
)

echo "==> ${CMD[*]}"
"${CMD[@]}"
