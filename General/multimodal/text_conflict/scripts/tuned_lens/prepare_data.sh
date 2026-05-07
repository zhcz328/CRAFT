#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
POSITION="${POSITION:-before_question}"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}/${DATASET_NAME}}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct_val.csv}"
TL_ROOT="${TL_ROOT:-/root/autodl-tmp/tuned_lens/${MODEL_SLUG}}"
OUT_DIR="${OUT_DIR:-${TL_ROOT}/data}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

PREDS_ARGS=()
if [[ -n "${PREDS_JSONL:-}" ]]; then
  PREDS_ARGS+=(--preds_jsonl "$PREDS_JSONL")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python tuned_lens/prepare_data.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  "${PREDS_ARGS[@]}" \
  --dataset_name "$DATASET_NAME" \
  --position "$POSITION" \
  --out_dir "$OUT_DIR" \
  "${RESUME_ARGS[@]}"
