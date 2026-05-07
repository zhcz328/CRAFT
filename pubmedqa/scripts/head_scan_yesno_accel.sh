#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DATA_PATH="${DATA_PATH:-$(join_output_path "data/train_yesno.train.json")}"
DATA_FMT="${DATA_FMT:-json}"
POSITION="${POSITION:-before_question}"
MAX_EXAMPLES="${MAX_EXAMPLES:-${LIMIT:-1000}}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_yesno/${POSITION}/headscan_pubmed")}"
PLAN_PATH="${PLAN_PATH:-$(join_output_path "result_yesno/${POSITION}/layer_trace/scan_plan.json")}"
ROUND_NAME="${ROUND_NAME:-}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
COARSE_LIMIT="${COARSE_LIMIT:-128}"
COARSE_TOPK="${COARSE_TOPK:-36}"
COARSE_SEED="${COARSE_SEED:-42}"
COARSE_WITH_NC="${COARSE_WITH_NC:-0}"
SAVE_RECORDS="${SAVE_RECORDS:-0}"

ROUND_ARGS=()
if [[ -n "$ROUND_NAME" ]]; then
  ROUND_ARGS+=(--round "$ROUND_NAME")
fi

COARSE_NC_ARGS=()
if [[ "$COARSE_WITH_NC" == "1" ]]; then
  COARSE_NC_ARGS+=(--coarse_with_nc)
fi

SAVE_RECORD_ARGS=()
if [[ "$SAVE_RECORDS" == "1" ]]; then
  SAVE_RECORD_ARGS+=(--save_records)
fi

"$PYTHON_BIN" head_scan_on_pubmed_yesno_accel.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --model "$MODEL_PATH" \
  --position "$POSITION" \
  --metrics "follow_conflict,yesno_margin" \
  --weights "0.7,0.3" \
  --plan "$PLAN_PATH" \
  --plan_out_dir "$OUT_DIR" \
  --max_examples "$MAX_EXAMPLES" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --coarse_limit "$COARSE_LIMIT" \
  --coarse_topk "$COARSE_TOPK" \
  --coarse_seed "$COARSE_SEED" \
  "${COARSE_NC_ARGS[@]}" \
  "${SAVE_RECORD_ARGS[@]}" \
  "${ROUND_ARGS[@]}"
