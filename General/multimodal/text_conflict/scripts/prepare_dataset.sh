#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_NAME="${DATASET_NAME:-gqa}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUT_CSV="${OUT_CSV:-}"
IMAGE_DIR_NAME="${IMAGE_DIR_NAME:-images}"

ARGS=(--dataset_name "$DATASET_NAME" --image_dir_name "$IMAGE_DIR_NAME")
if [[ -n "$DATASET_ROOT" ]]; then
  ARGS+=(--dataset_root "$DATASET_ROOT")
fi
if [[ -n "$OUT_CSV" ]]; then
  ARGS+=(--out_csv "$OUT_CSV")
fi

python prepare_dataset.py "${ARGS[@]}"
