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
POSITION="${POSITION:-image_conflict}"
RESULT_ROOT="${RESULT_ROOT:-${SAVE_ROOT}/${MODEL_SLUG}/result_${POSITION}_${DATASET_TAG}}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/${DATASET_STEM}_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/${DATASET_STEM}_val.csv}"
ABLATE_SPLITS="${ABLATE_SPLITS:-train val}"
SELECTED_HEADS="${SELECTED_HEADS:-${RESULT_ROOT}/selected_heads_core_layers.json}"
TRACE_MODE="${TRACE_MODE:-conflict}"
METRICS="${METRICS:-follow_context}"
MASK_SCALE="${MASK_SCALE:-1.0}"
MASK_SCOPE="${MASK_SCOPE:-all}"
KEEP_MODE="${KEEP_MODE:-self}"
DTYPE="${DTYPE:-bf16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/ablate}"
RESUME="${RESUME:-0}"
DEVICE="${DEVICE:-auto}"
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
    --image_root "${IMAGE_ROOT:-$(resolve_dataset_image_root "$DATASET_KEY")}" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --selected_heads "$SELECTED_HEADS" \
    --trace_mode "$TRACE_MODE" \
    --position "$POSITION" \
    --mask_scale "$MASK_SCALE" \
    --metrics "$METRICS" \
    --mask_scope "$MASK_SCOPE" \
    --keep_mode "$KEEP_MODE" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --out_json "${OUT_DIR}/ablate_selected_heads_${MASK_SCOPE}_${split}.json" \
    --max_examples "$MAX_EXAMPLES" \
    "${RESUME_ARGS[@]}"
done
