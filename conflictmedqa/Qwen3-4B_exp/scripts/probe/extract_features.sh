#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
MANIFEST_DIR="${MANIFEST_DIR:-$(probe_save_path "$POSITION")/data}"
FEATURE_DIR="${FEATURE_DIR:-$(probe_save_path "$POSITION")/features}"
SPLITS="${SPLITS:-train_pair_stratified val_pair_stratified}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
INCLUDE_EMBEDDINGS="${INCLUDE_EMBEDDINGS:-0}"

mkdir -p "$FEATURE_DIR"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

EMBED_ARGS=()
if [[ "$INCLUDE_EMBEDDINGS" == "1" ]]; then
  EMBED_ARGS+=(--include_embeddings)
fi

for split in $SPLITS; do
  "$PYTHON_BIN" probe/extract_features.py \
    --manifest "${MANIFEST_DIR}/${split}_${POSITION}.jsonl" \
    --model "$MODEL_PATH" \
    --out "${FEATURE_DIR}/${split}_${POSITION}.pt" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --save_dtype "$SAVE_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_samples "$MAX_SAMPLES" \
    "${THINK_ARGS[@]}" \
    "${EMBED_ARGS[@]}"
done
