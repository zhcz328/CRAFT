#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DATA_PATH="${DATA_PATH:-$(join_output_path "data/filter_yesno/${MODEL_SLUG}.filtered.json")}"
DATA_FMT="${DATA_FMT:-json}"
POSITIONS="${POSITIONS:-prefix before_question before_answer}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_all_positions_yesno")}"
LIMIT="${LIMIT:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

"$PYTHON_BIN" eval_nc_ic_cc_conflict_yesno.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --models "$MODEL_PATH" \
  --model_labels "$MODEL_SLUG" \
  --positions $POSITIONS \
  --out_dir "$OUT_DIR" \
  --limit "$LIMIT" \
  --dtype bf16 \
  --device_map auto \
  "${THINK_ARGS[@]}"
