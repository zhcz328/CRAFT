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
POSITION="${POSITION:-before_question}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/${DATASET_NAME}/result_${POSITION}_slake}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct_val.csv}"
ABLATE_SPLITS="${ABLATE_SPLITS:-train val}"
SELECTED_HEADS="${SELECTED_HEADS:-${RESULT_ROOT}/selected_heads_merged_unique_layers.json}"
TRACE_MODE="${TRACE_MODE:-conflict}"
METRICS="${METRICS:-follow_conflict}"
MASK_SCOPE="${MASK_SCOPE:-all}"
KEEP_MODE="${KEEP_MODE:-self}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-${LIMIT:-3}}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-672}"
RANDOM_SEED="${RANDOM_SEED:-0}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/ablate_random}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

for split in $ABLATE_SPLITS; do
  case "$split" in
    train)
      DATA_CSV="$TRAIN_CSV"
      ;;
    val)
      DATA_CSV="$VAL_CSV"
      ;;
    *)
      echo "Unknown split: $split" >&2
      exit 1
      ;;
  esac

  python ablate_head.py \
    --data_csv "$DATA_CSV" \
    --image_root . \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --selected_heads "$SELECTED_HEADS" \
    --trace_mode "$TRACE_MODE" \
    --position "$POSITION" \
    --metrics "$METRICS" \
    --mask_scope "$MASK_SCOPE" \
    --keep_mode "$KEEP_MODE" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --max_image_side "$MAX_IMAGE_SIDE" \
    --out_json "${OUT_DIR}/ablate_random_${MASK_SCOPE}_${split}.json" \
    --max_examples "$MAX_EXAMPLES" \
    --random_ablate \
    --random_seed "$RANDOM_SEED" \
    "${RESUME_ARGS[@]}"
done

