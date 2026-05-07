#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
POSITION="${POSITION:-image_conflict}"
DATA_ROOT="${DATA_ROOT:-${SAVE_ROOT}/data/${MODEL_SLUG}}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/${DATASET_STEM}_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/${DATASET_STEM}_val.csv}"
TL_ROOT="${TL_ROOT:-${SAVE_ROOT}/tuned_lens/${MODEL_SLUG}}"
OUT_DIR="${OUT_DIR:-${TL_ROOT}/data}"
PREDS_JSONL="${PREDS_JSONL:-${SAVE_ROOT}/${MODEL_SLUG}/eval-results_${DATASET_TAG}_image_conflict/preds.jsonl}"
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
  --position "$POSITION" \
  --out_dir "$OUT_DIR" \
  "${RESUME_ARGS[@]}"
