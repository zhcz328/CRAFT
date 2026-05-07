#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

RETEST_MODE="${RETEST_MODE:-logit}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.01}"
TOP_P="${TOP_P:-1.0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"

if [[ -z "${PAIRS_PATH:-}" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    PAIRS_PATH="$(join_output_path "result/kept_pairs_a12_b12.jsonl")"
  else
    PAIRS_PATH="kept_pairs_a12_b12.jsonl"
  fi
fi
if [[ -z "${RETEST_OUT:-}" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    RETEST_OUT="$(join_output_path "result/retest_logit.jsonl")"
  else
    RETEST_OUT="retest_logit.jsonl"
  fi
fi

ensure_parent_dir "$RETEST_OUT"

CMD=(
  "$PYTHON_BIN" retest_pairs.py
  --pairs "$PAIRS_PATH"
  --model "$MODEL_PATH"
  --mode "$RETEST_MODE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --dtype "$DTYPE"
  --device_map "$DEVICE_MAP"
  --out "$RETEST_OUT"
)

if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable_thinking)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
