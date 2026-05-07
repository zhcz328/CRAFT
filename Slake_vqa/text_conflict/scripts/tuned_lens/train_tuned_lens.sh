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
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_DIR}/train_${POSITION}.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATA_DIR}/val_${POSITION}.jsonl}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/${TRAIN_RUN_NAME}}"
PROMPT_TYPES="${PROMPT_TYPES:-base,support,conflict}"
LAYER_SPEC="${LAYER_SPEC:-}"
TRANSLATOR_RANK="${TRANSLATOR_RANK:-0}"
DTYPE="${DTYPE:-bfloat16}"
TRANSLATOR_DTYPE="${TRANSLATOR_DTYPE:-float32}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LENS_DEVICE="${LENS_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-4e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
IDENTITY_REG_WEIGHT="${IDENTITY_REG_WEIGHT:-3e-3}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

layer_args=()
if [[ -n "$LAYER_SPEC" ]]; then
  layer_args+=(--layer_spec "$LAYER_SPEC")
fi

python tuned_lens/train_tuned_lens.py \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --out_dir "$OUT_DIR" \
  --prompt_types "$PROMPT_TYPES" \
  "${layer_args[@]}" \
  --translator_rank "$TRANSLATOR_RANK" \
  --dtype "$DTYPE" \
  --translator_dtype "$TRANSLATOR_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --lens_device "$LENS_DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --image_size "$IMAGE_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --kl_temperature "$KL_TEMPERATURE" \
  --identity_reg_weight "$IDENTITY_REG_WEIGHT" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --log_every "$LOG_EVERY" \
  "${RESUME_ARGS[@]}"
