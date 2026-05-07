#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DATA_PATH="${DATA_PATH:-data/train-00000-of-00001.parquet}"
DATA_FMT="${DATA_FMT:-auto}"
OUT_DIR="${OUT_DIR:-$(join_output_path "data/filter_yesno")}"
LIMIT="${LIMIT:-10000}"
NC_YN_TAU="${NC_YN_TAU:-2.0}"
NC_UNK_TAU="${NC_UNK_TAU:-1.0}"
CC_YN_TAU="${CC_YN_TAU:-2.0}"
CC_UNK_TAU="${CC_UNK_TAU:-1.0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

"$PYTHON_BIN" filter_data_yesno.py \
  --data "$DATA_PATH" \
  --fmt "$DATA_FMT" \
  --out_dir "$OUT_DIR" \
  --models "$MODEL_PATH" \
  --model_labels "$MODEL_SLUG" \
  --limit "$LIMIT" \
  --device_map auto \
  --dtype bf16 \
  --nc_yn_tau "$NC_YN_TAU" \
  --nc_unk_tau "$NC_UNK_TAU" \
  --cc_yn_tau "$CC_YN_TAU" \
  --cc_unk_tau "$CC_UNK_TAU" \
  "${THINK_ARGS[@]}"
