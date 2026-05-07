#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/logit_lens"
TARGET_PROJECT="$ROOT/Slake_vqa/image_conflict"
SELECTED_HEADS="${SELECTED_HEADS:-$ROOT/heal-medvqa/hulumed4b/result_image_conflict_heal_medvqa/selected_heads_core_layers.json}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Hulu-Med-4B}"
MODEL_NAME="${MODEL_NAME:-hulumed-4b}"
DATASET="${DATASET:-slake_vqa}"
POSITION="${POSITION:-image_conflict}"
MASK_SCOPE="${MASK_SCOPE:-all}"
SPLITS="${SPLITS:-train val}"
OUT_DIR="${OUT_DIR:-$ROOT/exp/cross_dataset/image/results/heal_heads_on_slake}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
MASK_SCALE="${MASK_SCALE:-1.0}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-auto}"
TRACE_MODE="${TRACE_MODE:-conflict}"
KEEP_MODE="${KEEP_MODE:-self}"
METRICS="${METRICS:-follow_context}"
RESUME="${RESUME:-0}"

mkdir -p "$OUT_DIR"

cd "$TARGET_PROJECT"
DATASET="$DATASET" \
MODEL_PATH="$MODEL_PATH" \
MODEL_NAME="$MODEL_NAME" \
POSITION="$POSITION" \
SELECTED_HEADS="$SELECTED_HEADS" \
MASK_SCOPE="$MASK_SCOPE" \
ABLATE_SPLITS="$SPLITS" \
OUT_DIR="$OUT_DIR" \
MAX_EXAMPLES="$MAX_EXAMPLES" \
MASK_SCALE="$MASK_SCALE" \
DTYPE="$DTYPE" \
DEVICE="$DEVICE" \
TRACE_MODE="$TRACE_MODE" \
KEEP_MODE="$KEEP_MODE" \
METRICS="$METRICS" \
RESUME="$RESUME" \
bash "$TARGET_PROJECT/scripts/ablate_head.sh"
