#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DATA_PATH="${DATA_PATH:-$(join_output_path "data/filter_yesno/${MODEL_SLUG}.filtered.json")}"
DATA_FMT="${DATA_FMT:-json}"
POSITION="${POSITION:-before_question}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1000}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_yesno/${POSITION}/headscan_pubmed")}"
PLAN_PATH="${PLAN_PATH:-$(join_output_path "result_yesno/${POSITION}/layer_trace/scan_plan.json")}"
ROUND_NAME="${ROUND_NAME:-}"

ROUND_ARGS=()
if [[ -n "$ROUND_NAME" ]]; then
  ROUND_ARGS+=(--round "$ROUND_NAME")
fi

"$PYTHON_BIN" head_scan_on_pubmed_yesno.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --model "$MODEL_PATH" \
  --position "$POSITION" \
  --metrics "follow_conflict,yesno_margin" \
  --weights "0.7,0.3" \
  --plan "$PLAN_PATH" \
  --plan_out_dir "$OUT_DIR" \
  --max_examples "$MAX_EXAMPLES" \
  "${ROUND_ARGS[@]}"
