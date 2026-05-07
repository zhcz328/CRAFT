#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}/${DATASET_NAME}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_correct_ic_ready_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bf16}"
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-image_conflict}"
METRIC="${METRIC:-follow_context}"
MASK_SCALE="${MASK_SCALE:-1.0}"
SAMPLE_FILTER="${SAMPLE_FILTER:-none}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/${DATASET_NAME}/result_${POSITION}_slake}"
PLAN_PATH="${PLAN_PATH:-${RESULT_ROOT}/trace_image_conflict_scan_plan.json}"
PLAN_LAYER_KEY="${PLAN_LAYER_KEY:-core_layers}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-${RESULT_ROOT}/headscan_slake_mm}"
LIMIT="${LIMIT:-300}"
RESUME="${RESUME:-0}"
DEVICE="${DEVICE:-auto}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python head_scan_slake_mm_fastcache.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --mask_scale "$MASK_SCALE" \
  --sample_filter "$SAMPLE_FILTER" \
  --metrics "$METRIC" \
  --device "$DEVICE" \
  --plan "$PLAN_PATH" \
  --plan_layer_key "$PLAN_LAYER_KEY" \
  --plan_out_dir "$PLAN_OUT_DIR" \
  --limit "$LIMIT" \
  "${RESUME_ARGS[@]}"

