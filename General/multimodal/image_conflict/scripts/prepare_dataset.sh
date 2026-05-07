#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_ROOT="${DATASET_ROOT:-}"
OUT_CSV="${OUT_CSV:-}"
IMAGE_DIR_NAME="${IMAGE_DIR_NAME:-images}"
DISABLE_SYNTHETIC="${DISABLE_SYNTHETIC:-0}"

ARGS=(--image_dir_name "$IMAGE_DIR_NAME")
if [[ -n "$DATASET_ROOT" ]]; then
  ARGS+=(--dataset_root "$DATASET_ROOT")
fi
if [[ -n "$OUT_CSV" ]]; then
  ARGS+=(--out_csv "$OUT_CSV")
fi
if [[ "$DISABLE_SYNTHETIC" == "1" ]]; then
  ARGS+=(--disable_scene_graph_synthetic)
fi

python prepare_dataset.py "${ARGS[@]}"
