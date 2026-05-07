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
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_STEM}.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-$(resolve_dataset_image_root "$DATASET_KEY")}"
OUT_DIR="${OUT_DIR:-${SAVE_ROOT}/${MODEL_SLUG}/eval-results_${DATASET_TAG}_image_conflict}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESIZE_MAX_SIDE="${RESIZE_MAX_SIDE:-672}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
LIMIT="${LIMIT:-0}"
DEBUG_MASK_DIR="${DEBUG_MASK_DIR:-}"
SAVE_PROCESSED_IMAGES="${SAVE_PROCESSED_IMAGES:-0}"
MASK_ALL="${MASK_ALL:-0}"
ADD_NOISE="${ADD_NOISE:-0}"
ADD_NOISE_ALL="${ADD_NOISE_ALL:-0}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
MASK_SCALE="${MASK_SCALE:-1.0}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
DEBUG_ARGS=()
if [[ -n "$DEBUG_MASK_DIR" ]]; then
  DEBUG_ARGS+=(--debug_mask_dir "$DEBUG_MASK_DIR")
fi
SAVE_PROCESSED_IMAGES_ARGS=()
if [[ "$SAVE_PROCESSED_IMAGES" == "1" ]]; then
  SAVE_PROCESSED_IMAGES_ARGS+=(--save_processed_images)
fi
MASK_ALL_ARGS=()
if [[ "$MASK_ALL" == "1" ]]; then
  MASK_ALL_ARGS+=(--mask_all)
fi
ADD_NOISE_ARGS=()
if [[ "$ADD_NOISE" == "1" ]]; then
  ADD_NOISE_ARGS+=(--add_noise)
fi
ADD_NOISE_ALL_ARGS=()
if [[ "$ADD_NOISE_ALL" == "1" ]]; then
  ADD_NOISE_ALL_ARGS+=(--add_noise_all)
fi

python eval_slake_mm_image_conflict.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --dataset "$DATASET_KEY" \
  --out_dir "$OUT_DIR" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --resize_max_side "$RESIZE_MAX_SIDE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --noise_scale "$NOISE_SCALE" \
  --mask_scale "$MASK_SCALE" \
  "${DEBUG_ARGS[@]}" \
  "${SAVE_PROCESSED_IMAGES_ARGS[@]}" \
  "${MASK_ALL_ARGS[@]}" \
  "${ADD_NOISE_ARGS[@]}" \
  "${ADD_NOISE_ALL_ARGS[@]}" \
  "${RESUME_ARGS[@]}"


