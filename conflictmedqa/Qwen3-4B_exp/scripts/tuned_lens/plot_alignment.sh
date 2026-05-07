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
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/alignment}"
PROMPT_TYPES="${PROMPT_TYPES:-base,support,conflict}"
MODEL_LABEL="${MODEL_LABEL:-${OUTPUT_ROOT:-default}}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LENS_DEVICE="${LENS_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

"$PYTHON_BIN" tuned_lens/plot_alignment_figure.py \
  --manifest "$MANIFEST" \
  --model "$MODEL_PATH" \
  --lens_ckpt "$LENS_CKPT" \
  --out_dir "$OUT_DIR" \
  --prompt_types "$PROMPT_TYPES" \
  --model_label "$MODEL_LABEL" \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --lens_device "$LENS_DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --max_samples "$MAX_SAMPLES" \
  "${THINK_ARGS[@]}"
