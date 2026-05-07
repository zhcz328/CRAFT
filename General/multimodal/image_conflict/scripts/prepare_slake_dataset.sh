#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_ROOT="${DATASET_ROOT:-../data}"
IMAGE_ROOT="${IMAGE_ROOT:-}"
OUT_DIR="${OUT_DIR:-data}"
INCLUDE_NON_ENGLISH="${INCLUDE_NON_ENGLISH:-0}"
INCLUDE_OPEN="${INCLUDE_OPEN:-0}"
PROXY_STRATEGY="${PROXY_STRATEGY:-strict}"
TARGET_MODE="${TARGET_MODE:-resolved_targets}"

extra_args=()
if [[ -n "$IMAGE_ROOT" ]]; then
  extra_args+=(--image_root "$IMAGE_ROOT")
fi
if [[ "$INCLUDE_NON_ENGLISH" == "1" ]]; then
  extra_args+=(--include_non_english)
fi
if [[ "$INCLUDE_OPEN" == "1" ]]; then
  extra_args+=(--include_open)
fi

python prepare_slake_dataset.py \
  --dataset_root "$DATASET_ROOT" \
  --out_dir "$OUT_DIR" \
  --proxy_strategy "$PROXY_STRATEGY" \
  --target_mode "$TARGET_MODE" \
  "${extra_args[@]}"

