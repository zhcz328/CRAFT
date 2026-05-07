#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITION="${POSITION:-before_question}"
DATA_PATH="${DATA_PATH:-$(join_output_path "data/train_yesno.train.json")}"
DATA_FMT="${DATA_FMT:-json}"
SELECTED_HEADS="${SELECTED_HEADS:-$(join_output_path "result_yesno/${POSITION}/headscan_pubmed/selected_conflict_heads.json")}"
OUT_DIR="${OUT_DIR:-$(join_output_path "result_yesno/${POSITION}/ablate_random")}"
RANDOM_SEED="${RANDOM_SEED:-42}"
LIMIT="${LIMIT:-0}"
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" ablate_selected_heads_yesno.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --model "$MODEL_PATH" \
  --selected_heads "$SELECTED_HEADS" \
  --position "$POSITION" \
  --mask_scope all \
  --random_ablate \
  --random_seed "$RANDOM_SEED" \
  --limit "$LIMIT" \
  --out_jsonl "${OUT_DIR}/ablate_random_heads_train.jsonl" \
  --out_summary "${OUT_DIR}/ablate_random_heads_train_summary.json"
