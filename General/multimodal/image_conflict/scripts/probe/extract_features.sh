#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
POSITION="${POSITION:-image_conflict}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/image_conflict/probe/${MODEL_SLUG}}"
MANIFEST_DIR="${MANIFEST_DIR:-${PROBE_ROOT}/data}"
FEATURE_DIR="${FEATURE_DIR:-${PROBE_ROOT}/features}"
SPLITS="${SPLITS:-train val}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
MASK_SCALE="${MASK_SCALE:-1.0}"
DTYPE="${DTYPE:-bfloat16}"
SAVE_DTYPE="${SAVE_DTYPE:-float16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

for split in $SPLITS; do
  python probe/extract_features.py \
    --manifest "${MANIFEST_DIR}/${split}_${POSITION}.jsonl" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --out "${FEATURE_DIR}/${split}_${POSITION}.pt" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --mask_scale "$MASK_SCALE" \
    --dtype "$DTYPE" \
    --save_dtype "$SAVE_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_samples "$MAX_SAMPLES" \
    "${RESUME_ARGS[@]}"
done


