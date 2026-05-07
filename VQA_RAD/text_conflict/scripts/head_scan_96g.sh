#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
POSITION="${POSITION:-before_question}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/result_${POSITION}_vqarad}"
PLAN_PATH="${PLAN_PATH:-${RESULT_ROOT}/trace_conflict_scan_plan.json}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-${RESULT_ROOT}/headscan_vqarad_mm_accel_96g}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

python head_scan_vqarad_mm_fastcache_accel.py \
  --data_csv "${DATA_ROOT}/vqa_rad_nc_cc_both_correct_train.csv" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device auto \
  --dtype bf16 \
  --trace_mode conflict \
  --position "$POSITION" \
  --metrics follow_conflict \
  --plan "$PLAN_PATH" \
  --plan_out_dir "$PLAN_OUT_DIR" \
  --limit 0 \
  --coarse_limit 256 \
  --coarse_topk 96
