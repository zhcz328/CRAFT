#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITIONS="${POSITIONS:-prefix,before_question,before_answer}"
EVAL_OUT="${EVAL_OUT:-$(join_output_path "result_all_positions/conflict_positions.jsonl")}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

if [[ -z "${PAIRS_PATH:-}" ]]; then
  PAIRS_PATH="$(join_output_path "result/kept_pairs_a12_b12.jsonl")"
fi

ensure_parent_dir "$EVAL_OUT"

CMD=(
  "$PYTHON_BIN" conflict_context_internal_eval.py
  --pairs "$PAIRS_PATH"
  --model "$MODEL_PATH"
  --positions "$POSITIONS"
  --dtype "$DTYPE"
  --device_map "$DEVICE_MAP"
  --out "$EVAL_OUT"
)

if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable_thinking)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
