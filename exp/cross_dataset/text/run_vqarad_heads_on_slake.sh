#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/logit_lens"
SRC_SELECTED_HEADS="${SRC_SELECTED_HEADS:-$ROOT/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json}"
TARGET_PROJECT="${TARGET_PROJECT:-$ROOT/Slake_vqa/Hulu-med/text_conflict}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Hulu-Med-4B}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/data/SLAKE/imgs}"
POSITION="${POSITION:-before_question}"
TRACE_MODE="${TRACE_MODE:-conflict}"
METRICS="${METRICS:-follow_conflict}"
MASK_SCOPE="${MASK_SCOPE:-all}"
KEEP_MODE="${KEEP_MODE:-self}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-auto}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SPLITS="${SPLITS:-train val}"
OUT_ROOT="${OUT_ROOT:-$ROOT/exp/cross_dataset/text/results/vqarad_heads_on_slake_before_question}"

mkdir -p "$OUT_ROOT"

for split in $SPLITS; do
  case "$split" in
    train)
      DATA_CSV="$TARGET_PROJECT/data/slake_nc_cc_both_correct_train.csv"
      ;;
    val)
      DATA_CSV="$TARGET_PROJECT/data/slake_nc_cc_both_correct_val.csv"
      ;;
    *)
      echo "Unknown split: $split" >&2
      exit 1
      ;;
  esac

  python "$TARGET_PROJECT/ablate_head.py" \
    --data_csv "$DATA_CSV" \
    --image_root "$IMAGE_ROOT" \
    --model "$MODEL_PATH" \
    --selected_heads "$SRC_SELECTED_HEADS" \
    --trace_mode "$TRACE_MODE" \
    --position "$POSITION" \
    --metrics "$METRICS" \
    --mask_scope "$MASK_SCOPE" \
    --keep_mode "$KEEP_MODE" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --max_examples "$MAX_EXAMPLES" \
    --out_json "$OUT_ROOT/ablate_selected_heads_${MASK_SCOPE}_${split}.json"
done
