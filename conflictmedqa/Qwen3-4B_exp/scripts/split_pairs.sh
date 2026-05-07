#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

IN_PATH="${IN_PATH:-$(join_output_path "data/kept_pairs_a12_b10_all.jsonl")}"
TRAIN_OUT="${TRAIN_OUT:-$(join_output_path "data/kept_pairs_a12_b10_all_train.jsonl")}"
VAL_OUT="${VAL_OUT:-$(join_output_path "data/kept_pairs_a12_b10_all_val.jsonl")}"
TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
SPLIT_SEED="${SPLIT_SEED:-42}"
NO_SHUFFLE="${NO_SHUFFLE:-0}"

ensure_parent_dir "$TRAIN_OUT"
ensure_parent_dir "$VAL_OUT"

CMD=(
  "$PYTHON_BIN" split_kept_pairs.py
  --in_path "$IN_PATH"
  --train_out "$TRAIN_OUT"
  --val_out "$VAL_OUT"
  --train_ratio "$TRAIN_RATIO"
  --seed "$SPLIT_SEED"
)

if [[ "$NO_SHUFFLE" == "1" ]]; then
  CMD+=(--no_shuffle)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
