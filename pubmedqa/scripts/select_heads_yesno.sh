#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITION="${POSITION:-before_question}"
HEADSCAN_DIR="${HEADSCAN_DIR:-$(join_output_path "result_yesno/${POSITION}/headscan_pubmed")}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_yesno/${POSITION}/selected_heads")}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" select_heads.py \
  --round_files "${HEADSCAN_DIR}/head_scan_round1_preplateau.json,${HEADSCAN_DIR}/head_scan_round1_topk_fallback.json,${HEADSCAN_DIR}/head_scan_round2_tail.json" \
  --round_names "round1_preplateau,round1_topk_fallback,round2_tail" \
  --topk_per_round 20 \
  --mode conflict_specific \
  --combine union \
  --out_json "${OUT_DIR}/selected_conflict_heads.json" \
  --out_csv "${OUT_DIR}/selected_conflict_heads.csv"
