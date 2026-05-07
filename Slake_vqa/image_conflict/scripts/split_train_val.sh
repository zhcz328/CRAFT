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
DATASET_STEM="$(resolve_dataset_stem "$DATASET_KEY")"
DATA_ROOT="${DATA_ROOT:-${SAVE_ROOT}/data/${MODEL_SLUG}}"
IN_CSV="${IN_CSV:-${DATA_ROOT}/${DATASET_STEM}.csv}"
OUT_TRAIN="${OUT_TRAIN:-${DATA_ROOT}/${DATASET_STEM}_train.csv}"
OUT_VAL="${OUT_VAL:-${DATA_ROOT}/${DATASET_STEM}_val.csv}"
VAL_PERCENT="${VAL_PERCENT:-30}"
SEED="${SEED:-42}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python split_train_val.py \
  --in_csv "$IN_CSV" \
  --out_train "$OUT_TRAIN" \
  --out_val "$OUT_VAL" \
  --val_percent "$VAL_PERCENT" \
  --seed "$SEED" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
