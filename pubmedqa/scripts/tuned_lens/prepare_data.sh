#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
TRAIN_RECORDS="${TRAIN_RECORDS:-$(join_output_path "data/train_yesno.train.json")}"
VAL_RECORDS="${VAL_RECORDS:-$(join_output_path "data/train_yesno.val.json")}"
CONFLICT_POSITIONS="${CONFLICT_POSITIONS:-$(join_output_path "result_all_positions_yesno/${MODEL_SLUG}/${POSITION}/preds.jsonl")}"
OUT_DIR="${OUT_DIR:-$(join_output_path "tuned_lens/data")}"

"$PYTHON_BIN" tuned_lens/prepare_data.py \
  --train_records "$TRAIN_RECORDS" \
  --val_records "$VAL_RECORDS" \
  --conflict_positions "$CONFLICT_POSITIONS" \
  --position "$POSITION" \
  --out_dir "$OUT_DIR"
