#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/slake_nc_cc_both_correct_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-fp16}"
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-before_question}"
TRACE_ROOT="${MODEL_SLUG}/result_${POSITION}_slake"
PATCH_K="${PATCH_K:-16}"
LIMIT="${LIMIT:-300}"
METRICS="${METRICS:-follow_conflict}"
OUT_PATH="${OUT_PATH:-${TRACE_ROOT}/trace_conflict.json}"
PLOT_PREFIX="${PLOT_PREFIX:-${TRACE_ROOT}/trace_conflict}"
SCAN_PLAN_OUT="${SCAN_PLAN_OUT:-${TRACE_ROOT}/trace_conflict_scan_plan.json}"
SCAN_PLAN_METRIC="${SCAN_PLAN_METRIC:-$METRICS}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
MERGE_SHARDS="${MERGE_SHARDS:-0}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
MERGE_ARGS=()
if [[ "$MERGE_SHARDS" == "1" ]]; then
  MERGE_ARGS+=(--merge_shards)
fi

python layer_trace_slake_mm_current_fixed_fastcache.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --patch_k "$PATCH_K" \
  --limit "$LIMIT" \
  --metrics "$METRICS" \
  --out "$OUT_PATH" \
  --plot_prefix "$PLOT_PREFIX" \
  --scan_plan_out "$SCAN_PLAN_OUT" \
  --scan_plan_metric "$SCAN_PLAN_METRIC" \
  --num_shards "$NUM_SHARDS" \
  --shard_idx "$SHARD_IDX" \
  "${MERGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
