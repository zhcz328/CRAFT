#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
TRAIN_PAIRS="${TRAIN_PAIRS:-$(join_output_path "data/kept_pairs_a12_b10_all_train.jsonl")}"
VAL_PAIRS="${VAL_PAIRS:-$(join_output_path "data/kept_pairs_a12_b10_all_val.jsonl")}"
CONFLICT_POSITIONS="${CONFLICT_POSITIONS:-$(join_output_path "result_all_positions/conflict_positions.jsonl")}"
OUT_DIR="${OUT_DIR:-$(probe_save_path "$POSITION")/data}"
PAIR_VAL_RATIO="${PAIR_VAL_RATIO:-0.1}"
PAIR_SPLIT_SEED="${PAIR_SPLIT_SEED:-42}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" probe/prepare_data.py \
  --train_pairs "$TRAIN_PAIRS" \
  --val_pairs "$VAL_PAIRS" \
  --conflict_positions "$CONFLICT_POSITIONS" \
  --position "$POSITION" \
  --out_dir "$OUT_DIR" \
  --pair_val_ratio "$PAIR_VAL_RATIO" \
  --pair_split_seed "$PAIR_SPLIT_SEED"
