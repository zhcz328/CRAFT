#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/logit_lens/exp/noise"
NOISE_STD="${NOISE_STD:-24.0}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Hulu-Med-4B}"
MODEL_NAME="${MODEL_NAME:-}"
DATA_CSV="${DATA_CSV:-/root/logit_lens/Slake_vqa/image_conflict/data/hulumed4b/slake_nc_correct_ic_ready.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/logit_lens/Slake_vqa}"
SELECTED_HEADS="${SELECTED_HEADS:-/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/selected_heads_core_layers.json}"
OUT_JSON="${OUT_JSON:-/root/logit_lens/exp/noise/hulumed4b_noise_ablation_selected_heads_${NOISE_STD}.json}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:--1}"
NOISE_MODE="${NOISE_MODE:-local}"
MASK_SCOPE="${MASK_SCOPE:-ctx_only}"
KEEP_MODE="${KEEP_MODE:-self}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
MASK_SCALE="${MASK_SCALE:-1.0}"

RESUME="${RESUME:-0}"
SAVE_NOISE_IMAGES="${SAVE_NOISE_IMAGES:-1}"
NOISE_IMAGE_DIR="${NOISE_IMAGE_DIR:-/root/logit_lens/exp/noise/hulumed4b_noise_images_${NOISE_STD}}"

MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

SAVE_NOISE_IMAGE_ARGS=()
if [[ "$SAVE_NOISE_IMAGES" == "1" ]]; then
  SAVE_NOISE_IMAGE_ARGS+=(--save_noise_images --noise_image_dir "$NOISE_IMAGE_DIR")
fi

python3 "$ROOT_DIR/ablate_selected_heads_noise.py" \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --selected_heads "$SELECTED_HEADS" \
  --out_json "$OUT_JSON" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --max_examples "$MAX_EXAMPLES" \
  --noise_mode "$NOISE_MODE" \
  --mask_scope "$MASK_SCOPE" \
  --keep_mode "$KEEP_MODE" \
  --noise_scale "$NOISE_SCALE" \
  --mask_scale "$MASK_SCALE" \
  --noise_std "$NOISE_STD" \
  "${SAVE_NOISE_IMAGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
