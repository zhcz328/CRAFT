#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

if [[ -z "${FILTER_IN:-}" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    FILTER_IN="$(join_output_path "result/confilctmedqa_pred_logits.jsonl")"
  else
    FILTER_IN="/home/zengjiaqi/icl/interp/logit_lens/confilctmedqa_pred_logits.jsonl"
  fi
fi
if [[ -z "${FILTER_OUT:-}" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    FILTER_OUT="$(join_output_path "result/kept_pairs_a12_b12.jsonl")"
  else
    FILTER_OUT="kept_pairs_a12_b12.jsonl"
  fi
fi
MARGIN_A="${MARGIN_A:-12}"
MARGIN_B="${MARGIN_B:-10}"
PAIR_OFFSET="${PAIR_OFFSET:-2145}"
TOPK="${TOPK:-0}"

ensure_parent_dir "$FILTER_OUT"

CMD=(
  "$PYTHON_BIN" filter_pairs_margin.py
  --in "$FILTER_IN"
  --out "$FILTER_OUT"
  --offset "$PAIR_OFFSET"
  --a "$MARGIN_A"
  --b "$MARGIN_B"
)

if [[ "$TOPK" != "0" ]]; then
  CMD+=(--topk "$TOPK")
fi

echo "==> ${CMD[*]}"
"${CMD[@]}"
