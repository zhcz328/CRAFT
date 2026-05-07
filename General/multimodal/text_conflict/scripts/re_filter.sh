#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
IN_CSV="${IN_CSV:-./data/${DATASET_NAME}_closed_with_wrong.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
OUT_CSV="${OUT_CSV:-./data/${MODEL_SLUG}/${DATASET_NAME}_nc_cc_both_correct.csv}"
DTYPE="${DTYPE:-fp16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESIZE_MAX_SIDE="${RESIZE_MAX_SIDE:-672}"
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

python filter_fine_grained.py \
  --in_csv "$IN_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --out_csv "$OUT_CSV" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --resize_max_side "$RESIZE_MAX_SIDE" \
  --limit "$LIMIT" \
  "${RESUME_ARGS[@]}"