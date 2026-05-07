#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
POSITION="${POSITION:-before_question}"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/slake_nc_cc_both_correct_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/slake_nc_cc_both_correct_val.csv}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/probe/${MODEL_SLUG}}"
OUT_DIR="${OUT_DIR:-${PROBE_ROOT}/data}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

PREDS_ARGS=()
if [[ -n "${PREDS_JSONL:-}" ]]; then
  PREDS_ARGS+=(--preds_jsonl "$PREDS_JSONL")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

python probe/prepare_data.py \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  "${PREDS_ARGS[@]}" \
  --position "$POSITION" \
  --out_dir "$OUT_DIR" \
  "${RESUME_ARGS[@]}"
