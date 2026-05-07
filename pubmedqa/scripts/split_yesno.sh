#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

INPUT_PATH="${INPUT_PATH:-$(join_output_path "data/filter_yesno/${MODEL_SLUG}.filtered.json")}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
SEED="${SEED:-42}"
OUT_TRAIN="${OUT_TRAIN:-$(join_output_path "data/train_yesno.train.json")}"
OUT_VAL="${OUT_VAL:-$(join_output_path "data/train_yesno.val.json")}"
OUT_META="${OUT_META:-$(join_output_path "data/train_yesno.split_meta.json")}"

"$PYTHON_BIN" split_train_val.py \
  --input "$INPUT_PATH" \
  --train_ratio "$TRAIN_RATIO" \
  --seed "$SEED" \
  --out_train "$OUT_TRAIN" \
  --out_val "$OUT_VAL" \
  --out_meta "$OUT_META"
