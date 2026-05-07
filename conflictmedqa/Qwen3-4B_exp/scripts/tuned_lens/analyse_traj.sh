#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
DATA_DIR="${DATA_DIR:-$(tuned_lens_save_path "$POSITION")/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$(tuned_lens_save_path "$POSITION")}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-train_${POSITION}}"
MANIFEST="${MANIFEST:-${DATA_DIR}/val_pair_stratified_${POSITION}.jsonl}"
LENS_CKPT="${LENS_CKPT:-${RESULTS_ROOT}/tuned_lens.pt}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/trajectory}"
PROMPT_TYPES="${PROMPT_TYPES:-conflict}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LENS_DEVICE="${LENS_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MARGIN_EPS="${MARGIN_EPS:-0.1}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PLOT_MAX_RECORDS="${PLOT_MAX_RECORDS:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

mkdir -p "$OUT_DIR"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

"$PYTHON_BIN" tuned_lens/analyze_trajectories.py \
  --manifest "$MANIFEST" \
  --model "$MODEL_PATH" \
  --lens_ckpt "$LENS_CKPT" \
  --out_dir "$OUT_DIR" \
  --prompt_types "$PROMPT_TYPES" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --lens_device "$LENS_DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --margin_eps "$MARGIN_EPS" \
  --max_samples "$MAX_SAMPLES" \
  --plot_max_records "$PLOT_MAX_RECORDS" \
  "${THINK_ARGS[@]}"
