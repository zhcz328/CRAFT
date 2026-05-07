#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET="${DATASET:-slake_vqa}"
DATASET_KEY="$(resolve_dataset_key "$DATASET")"
SAVE_ROOT="${SAVE_ROOT:-$(resolve_dataset_save_root "$ROOT_DIR" "$DATASET_KEY")}"
DATASET_TAG="$(resolve_dataset_tag "$DATASET_KEY")"
DATASET_STEM="$(resolve_dataset_stem "$DATASET_KEY")"
DATA_ROOT="${DATA_ROOT:-${SAVE_ROOT}/data/${MODEL_SLUG}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_STEM}_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$(resolve_dataset_image_root "$DATASET_KEY")}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bf16}"
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-image_conflict}"
METRICS="${METRICS:-follow_context}"
MASK_SCALE="${MASK_SCALE:-1.0}"
RESULT_ROOT="${RESULT_ROOT:-${SAVE_ROOT}/${MODEL_SLUG}/result_${POSITION}_${DATASET_TAG}}"
PLAN_PATH="${PLAN_PATH:-${RESULT_ROOT}/trace_image_conflict_scan_plan.json}"
PLAN_LAYER_KEY="${PLAN_LAYER_KEY:-core_layers}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-${RESULT_ROOT}/headscan_slake_mm_accel}"
LIMIT="${LIMIT:-0}"
COARSE_LIMIT="${COARSE_LIMIT:-128}"
COARSE_TOPK="${COARSE_TOPK:-36}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python head_scan_slake_mm_fastcache_accel.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --mask_scale "$MASK_SCALE" \
  --metrics "$METRICS" \
  --plan "$PLAN_PATH" \
  --plan_layer_key "$PLAN_LAYER_KEY" \
  --plan_out_dir "$PLAN_OUT_DIR" \
  --limit "$LIMIT" \
  --coarse_limit "$COARSE_LIMIT" \
  --coarse_topk "$COARSE_TOPK" \
  "${RESUME_ARGS[@]}"
