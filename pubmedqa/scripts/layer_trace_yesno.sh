#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DATA_PATH="${DATA_PATH:-$(join_output_path "data/train_yesno.train.json")}"
DATA_FMT="${DATA_FMT:-json}"
POSITION="${POSITION:-before_question}"
LIMIT="${LIMIT:-500}"
PATCH_ALL_TOKENS="${PATCH_ALL_TOKENS:-0}"
METRICS="${METRICS:-yesno_margin,follow_conflict}"
SCAN_PLAN_METRICS="${SCAN_PLAN_METRICS:-follow_conflict,yesno_margin}"
SCAN_PLAN_WEIGHTS="${SCAN_PLAN_WEIGHTS:-1.0,0.0}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_yesno/${POSITION}/layer_trace")}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
PATCH_ARGS=()
if [[ "$PATCH_ALL_TOKENS" == "1" ]]; then
  PATCH_ARGS+=(--patch_all_tokens)
fi

"$PYTHON_BIN" layer_trace_on_pubmed_yesno.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --model "$MODEL_PATH" \
  --position "$POSITION" \
  --limit "$LIMIT" \
  --metrics "$METRICS" \
  --scan_plan_out "${OUT_DIR}/scan_plan.json" \
  --scan_plan_metrics "$SCAN_PLAN_METRICS" \
  --scan_plan_weights "$SCAN_PLAN_WEIGHTS" \
  --device_map "$DEVICE_MAP" \
  --out "${OUT_DIR}/trace.json" \
  --plot_prefix "${OUT_DIR}/trace" \
  "${PATCH_ARGS[@]}"
