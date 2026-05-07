#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

DRIFTMED_CONFIG="${DRIFTMED_CONFIG:-gpt4o}"
DRIFTMED_SPLIT="${DRIFTMED_SPLIT:-train}"
DRIFTMED_LIMIT="${DRIFTMED_LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
EVAL_MODE="${EVAL_MODE:-logit}"
LOCAL_FILE="${LOCAL_FILE:-/root/logit_lens/conflictmedqa/Qwen3-4B_exp/data/ConflictMed_v2.jsonl}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

if [[ -z "${OUT_PATH:-}" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    OUT_PATH="$(join_output_path "result/confilctmedqa_pred_raw.jsonl")"
  else
    OUT_PATH="confilctmedqa_pred_raw.jsonl"
  fi
fi

ensure_parent_dir "$OUT_PATH"

CMD=(
  "$PYTHON_BIN" conflictmedqa_eval_local.py
  --model "$MODEL_PATH"
  --config "$DRIFTMED_CONFIG"
  --split "$DRIFTMED_SPLIT"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --limit "$DRIFTMED_LIMIT"
  --mode "$EVAL_MODE"
  --out "$OUT_PATH"
)

if [[ -n "$LOCAL_FILE" ]]; then
  CMD+=(--local_file "$LOCAL_FILE")
fi
if [[ "$ENABLE_THINKING" == "1" ]]; then
  CMD+=(--enable_thinking)
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
