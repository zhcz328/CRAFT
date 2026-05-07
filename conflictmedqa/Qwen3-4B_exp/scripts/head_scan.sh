#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

POSITION="${POSITION:-before_question}"
PAIRS_PATH="${PAIRS_PATH:-$(join_output_path "data/kept_pairs_a12_b10_all_train.jsonl")}"
PLAN_PATH="${PLAN_PATH:-$(join_output_path "result/${POSITION}/layer_trace_rise/scan_plan.json")}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-$(join_output_path "result/${POSITION}/headscan_rounds_top30_inf")}"
MAX_PAIRS="${MAX_PAIRS:-120}"
DTYPE="${DTYPE:-bfloat16}"
GROUPS_TOPK="${GROUPS_TOPK:-30}"
CS_EFF_MIN="${CS_EFF_MIN:-3.5}"
CS_BASE_MAX="${CS_BASE_MAX:-4.5}"
BB_EFF_MIN="${BB_EFF_MIN:-3.44}"
BB_BASE_MIN="${BB_BASE_MIN:-2.58}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

mkdir -p "$PLAN_OUT_DIR"

CMD=(
  "$PYTHON_BIN" head_scan_inf.py
  --pairs "$PAIRS_PATH"
  --model "$MODEL_PATH"
  --plan "$PLAN_PATH"
  --plan_out_dir "$PLAN_OUT_DIR"
  --position "$POSITION"
  --max_pairs "$MAX_PAIRS"
  --groups_topk "$GROUPS_TOPK"
  --cs_eff_min "$CS_EFF_MIN"
  --cs_base_max "$CS_BASE_MAX"
  --bb_eff_min "$BB_EFF_MIN"
  --bb_base_min "$BB_BASE_MIN"
  --dtype "$DTYPE"
)

if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable_thinking)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
