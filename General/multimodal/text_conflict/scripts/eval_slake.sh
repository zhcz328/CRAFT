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
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
POSITION="${POSITION:-before_question}"
OUT_DIR="${OUT_DIR:-${MODEL_SLUG}/${DATASET_NAME}/eval-results_slake_all}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python eval_slake_mm_support_conflict_use_wrong_csv.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --position "$POSITION" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --limit "$LIMIT" \
  "${RESUME_ARGS[@]}"