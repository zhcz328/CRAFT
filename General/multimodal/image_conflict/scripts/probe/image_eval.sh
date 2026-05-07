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
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}/${DATASET_NAME}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_correct_ic_ready_val.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/image_conflict/probe/${MODEL_SLUG}}"
OUT_DIR="${OUT_DIR:-${PROBE_ROOT}/image_eval/${POSITION}}"
ABLATION="${ABLATION:-object_mask}"
SAMPLE_FILTER="${SAMPLE_FILTER:-base_gold}"
MASK_SCALE="${MASK_SCALE:-1.0}"
PATCH_SIZE="${PATCH_SIZE:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-672}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

python probe/image_eval.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --ablation "$ABLATION" \
  --sample_filter "$SAMPLE_FILTER" \
  --mask_scale "$MASK_SCALE" \
  --patch_size "$PATCH_SIZE" \
  --max_samples "$MAX_SAMPLES" \
  --max_image_side "$MAX_IMAGE_SIDE" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --out_jsonl "${OUT_DIR}/${ABLATION}_${SAMPLE_FILTER}.jsonl" \
  --out_summary "${OUT_DIR}/${ABLATION}_${SAMPLE_FILTER}_summary.json"


