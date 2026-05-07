#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/vqa_rad_nc_cc_both_correct_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bf16}"
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-before_question}"
METRIC="${METRIC:-follow_conflict}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/result_${POSITION}_vqarad}"
PLAN_PATH="${PLAN_PATH:-${RESULT_ROOT}/trace_conflict_scan_plan.json}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-${RESULT_ROOT}/headscan_vqarad_mm}"
LIMIT="${LIMIT:-300}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python head_scan_vqarad_mm_fastcache.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --metrics "$METRIC" \
  --plan "$PLAN_PATH" \
  --plan_out_dir "$PLAN_OUT_DIR" \
  --limit "$LIMIT" \
  "${RESUME_ARGS[@]}"

