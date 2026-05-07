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
POSITION="${POSITION:-image_conflict}"
RESULT_ROOT="${RESULT_ROOT:-${SAVE_ROOT}/${MODEL_SLUG}/result_${POSITION}_${DATASET_TAG}}"
HEADSCAN_DIR="${HEADSCAN_DIR:-${RESULT_ROOT}/headscan_slake_mm_accel}"
SOURCE="${SOURCE:-results}"
MODE="${MODE:-conflict_specific}"
EFF_MIN="${EFF_MIN:-0.28}"
BASE_MAX="${BASE_MAX:-0.35}"
BASE_MIN="${BASE_MIN:-}"
TOPK="${TOPK:-50}"
OUT_JSON="${OUT_JSON:-${RESULT_ROOT}/selected_heads_core_layers.json}"
OUT_CSV="${OUT_CSV:-${RESULT_ROOT}/selected_heads_core_layers.csv}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
BASE_MAX_ARGS=()
if [[ -n "$BASE_MAX" ]]; then
  BASE_MAX_ARGS+=(--base_max "$BASE_MAX")
fi
BASE_MIN_ARGS=()
if [[ -n "$BASE_MIN" ]]; then
  BASE_MIN_ARGS+=(--base_min "$BASE_MIN")
fi

python select_heads.py \
  --input_json "${HEADSCAN_DIR}/head_scan_core_layers.json" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --position "$POSITION" \
  --source "$SOURCE" \
  --mode "$MODE" \
  --eff_min "$EFF_MIN" \
  "${BASE_MAX_ARGS[@]}" \
  "${BASE_MIN_ARGS[@]}" \
  --topk "$TOPK" \
  --out_json "$OUT_JSON" \
  --out_csv "$OUT_CSV" \
  "${RESUME_ARGS[@]}"
