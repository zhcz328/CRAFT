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
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-image_conflict}"
TRACE_ROOT="${MODEL_SLUG}/${DATASET_NAME}/result_${POSITION}_slake"
MASK_SCALE="${MASK_SCALE:-1.0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-672}"
SAMPLE_FILTER="${SAMPLE_FILTER:-none}"
LAYER_SCALE="${LAYER_SCALE:-0.0}"
LIMIT="${LIMIT:-300}"
METRICS="${METRICS:-follow_context}"
OUT_PATH="${OUT_PATH:-${TRACE_ROOT}/trace_image_conflict.json}"
PLOT_PREFIX="${PLOT_PREFIX:-${TRACE_ROOT}/trace_image_conflict}"
SCAN_PLAN_OUT="${SCAN_PLAN_OUT:-${TRACE_ROOT}/trace_image_conflict_scan_plan.json}"
SCAN_PLAN_METRIC="${SCAN_PLAN_METRIC:-hallucination_relief}"
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

python layer_trace_slake_mm_current_fixed_fastcache.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --layer_scale "$LAYER_SCALE" \
  --mask_scale "$MASK_SCALE" \
  --max_image_side "$MAX_IMAGE_SIDE" \
  --sample_filter "$SAMPLE_FILTER" \
  --limit "$LIMIT" \
  --metrics "$METRICS" \
  --out "$OUT_PATH" \
  --device "$DEVICE" \
  --plot_prefix "$PLOT_PREFIX" \
  --scan_plan_out "$SCAN_PLAN_OUT" \
  --scan_plan_metric "$SCAN_PLAN_METRIC" \
  "${RESUME_ARGS[@]}"
