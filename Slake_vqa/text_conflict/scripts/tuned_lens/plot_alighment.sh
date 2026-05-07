#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
POSITION="${POSITION:-before_question}"
TL_ROOT="${TL_ROOT:-/root/autodl-tmp/tuned_lens/${MODEL_SLUG}}"
DATA_DIR="${DATA_DIR:-${TL_ROOT}/data}"
RESULTS_ROOT="${RESULTS_ROOT:-${TL_ROOT}/results/${POSITION}}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-train_idreg}"
MANIFEST="${MANIFEST:-${DATA_DIR}/val_${POSITION}.jsonl}"
LENS_CKPT="${LENS_CKPT:-${RESULTS_ROOT}/${TRAIN_RUN_NAME}/tuned_lens.pt}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/alignment_${TRAIN_RUN_NAME}}"
PROMPT_TYPES="${PROMPT_TYPES:-base,support,conflict}"
MODEL_LABEL="${MODEL_LABEL:-${MODEL_SLUG}}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LENS_DEVICE="${LENS_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python tuned_lens/plot_alignment_figure.py \
  --manifest "$MANIFEST" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --lens_ckpt "$LENS_CKPT" \
  --out_dir "$OUT_DIR" \
  --prompt_types "$PROMPT_TYPES" \
  --model_label "$MODEL_LABEL" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --lens_device "$LENS_DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --temperature "$TEMPERATURE" \
  --max_samples "$MAX_SAMPLES" \
  "${RESUME_ARGS[@]}"
