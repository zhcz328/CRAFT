#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
DATA_DIR="${DATA_DIR:-$(join_output_path "tuned_lens/data")}"
RESULTS_ROOT="${RESULTS_ROOT:-$(join_output_path "tuned_lens/results")}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-train_${POSITION}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${DATA_DIR}/train_pair_stratified_${POSITION}.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${DATA_DIR}/val_pair_stratified_${POSITION}.jsonl}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/${TRAIN_RUN_NAME}}"
PROMPT_TYPES="${PROMPT_TYPES:-base,support,conflict}"
LAYER_SPEC="${LAYER_SPEC:-}"
TRANSLATOR_RANK="${TRANSLATOR_RANK:-0}"
DTYPE="${DTYPE:-bfloat16}"
TRANSLATOR_DTYPE="${TRANSLATOR_DTYPE:-float32}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LENS_DEVICE="${LENS_DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

mkdir -p "$OUT_DIR"

LAYER_ARGS=()
if [[ -n "$LAYER_SPEC" ]]; then
  LAYER_ARGS+=(--layer_spec "$LAYER_SPEC")
fi

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

"$PYTHON_BIN" tuned_lens/train_tuned_lens.py \
  --train_manifest "$TRAIN_MANIFEST" \
  --val_manifest "$VAL_MANIFEST" \
  --model "$MODEL_PATH" \
  --out_dir "$OUT_DIR" \
  --prompt_types "$PROMPT_TYPES" \
  "${LAYER_ARGS[@]}" \
  --translator_rank "$TRANSLATOR_RANK" \
  --dtype "$DTYPE" \
  --translator_dtype "$TRANSLATOR_DTYPE" \
  --device_map "$DEVICE_MAP" \
  --lens_device "$LENS_DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --kl_temperature "$KL_TEMPERATURE" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_val_samples "$MAX_VAL_SAMPLES" \
  --log_every "$LOG_EVERY" \
  "${THINK_ARGS[@]}"
