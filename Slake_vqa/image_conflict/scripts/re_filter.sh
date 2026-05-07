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
DATASET_STEM="$(resolve_dataset_stem "$DATASET_KEY")"
IN_CSV="${IN_CSV:-$(resolve_dataset_source_csv "$ROOT_DIR" "$DATASET_KEY")}"
IMAGE_ROOT="${IMAGE_ROOT:-$(resolve_dataset_image_root "$DATASET_KEY")}"
OUT_CSV="${OUT_CSV:-${SAVE_ROOT}/data/${MODEL_SLUG}/${DATASET_STEM}.csv}"
DEFAULT_WRONG_CSV=""
if [[ "$DATASET_KEY" == "slake_vqa" ]]; then
  for candidate in "./data/slake_closed_all_wrong_backup.csv" "../text_conflict/data/slake_closed_all.csv"; do
    if [[ -f "$candidate" ]]; then
      DEFAULT_WRONG_CSV="$candidate"
      break
    fi
  done
fi
WRONG_CSV="${WRONG_CSV:-$DEFAULT_WRONG_CSV}"
DTYPE="${DTYPE:-fp16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESIZE_MAX_SIDE="${RESIZE_MAX_SIDE:-672}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
LIMIT="${LIMIT:-0}"
RESUME="${RESUME:-0}"
INTERNVL_CLOSED_FULL_FORWARD="${INTERNVL_CLOSED_FULL_FORWARD:-0}"
DEBUG_ROWS="${DEBUG_ROWS:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
INTERNVL_CLOSED_FULL_FORWARD_ARGS=()
if [[ "$INTERNVL_CLOSED_FULL_FORWARD" == "1" ]]; then
  INTERNVL_CLOSED_FULL_FORWARD_ARGS+=(--internvl_closed_full_forward)
fi
WRONG_CSV_ARGS=()
if [[ -n "$WRONG_CSV" ]]; then
  WRONG_CSV_ARGS+=(--wrong_csv "$WRONG_CSV")
fi

python filter_fine_grained.py \
  --in_csv "$IN_CSV" \
  --dataset "$DATASET_KEY" \
  "${WRONG_CSV_ARGS[@]}" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --out_csv "$OUT_CSV" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --resize_max_side "$RESIZE_MAX_SIDE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --limit "$LIMIT" \
  --debug_rows "$DEBUG_ROWS" \
  "${RESUME_ARGS[@]}" \
  "${INTERNVL_CLOSED_FULL_FORWARD_ARGS[@]}"
