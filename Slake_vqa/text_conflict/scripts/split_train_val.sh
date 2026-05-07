#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}}"
IN_CSV="${IN_CSV:-${DATA_ROOT}/slake_nc_cc_both_correct.csv}"
VAL_PERCENT="${VAL_PERCENT:-30}"
SEED="${SEED:-42}"
OUT_TRAIN="${OUT_TRAIN:-${DATA_ROOT}/slake_nc_cc_both_correct_train.csv}"
OUT_VAL="${OUT_VAL:-${DATA_ROOT}/slake_nc_cc_both_correct_val.csv}"
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
  --val_percent "$VAL_PERCENT" \
  --seed "$SEED" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --out_train "$OUT_TRAIN" \
  --out_val "$OUT_VAL" \
  "${RESUME_ARGS[@]}"
