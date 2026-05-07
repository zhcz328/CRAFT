#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITION="${POSITION:-before_question}"
TRACE_DIR="${TRACE_DIR:-$(join_output_path "result/${POSITION}/layer_trace_rise")}"
TRACE_OUT="${TRACE_OUT:-$TRACE_DIR/layer_trace.json}"
TRACE_PLOT_OUT="${TRACE_PLOT_OUT:-$TRACE_DIR/layer_trace_scores.png}"
TRACE_PLAN_OUT="${TRACE_PLAN_OUT:-$TRACE_DIR/scan_plan.json}"
PATCH_K="${PATCH_K:-8}"
MAX_PAIRS="${MAX_PAIRS:-120}"
DTYPE="${DTYPE:-bfloat16}"
PATCH_ALL_TOKEN="${PATCH_ALL_TOKEN:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

if [[ -z "${PAIRS_PATH:-}" ]]; then
  PAIRS_PATH="$(join_output_path "data/kept_pairs_a12_b10_all_train.jsonl")"
fi

mkdir -p "$TRACE_DIR"

CMD=(
  "$PYTHON_BIN" layer_trace.py
  --pairs "$PAIRS_PATH"
  --model "$MODEL_PATH"
  --position "$POSITION"
  --patch_k "$PATCH_K"
  --max_pairs "$MAX_PAIRS"
  --dtype "$DTYPE"
  --out "$TRACE_OUT"
  --plot_out "$TRACE_PLOT_OUT"
  --plan_out "$TRACE_PLAN_OUT"
)

if [[ "$PATCH_ALL_TOKEN" == "1" ]]; then
  CMD+=(--patch_all_token)
fi
if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable_thinking)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
