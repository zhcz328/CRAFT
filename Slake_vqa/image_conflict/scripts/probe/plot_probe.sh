#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET="${DATASET:-slake_vqa}"
DATASET_KEY="$(resolve_dataset_key "$DATASET")"
SAVE_ROOT="${SAVE_ROOT:-$(resolve_dataset_save_root "$ROOT_DIR" "$DATASET_KEY")}"
POSITION="${POSITION:-image_conflict}"
PROBE_ROOT="${PROBE_ROOT:-${SAVE_ROOT}/probe/${MODEL_SLUG}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROBE_ROOT}/results/${POSITION}}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_TASKS="${PROBE_TASKS:-conflict hallucination}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

for task in $PROBE_TASKS; do
  if [[ "$task" == "conflict" ]]; then
    metric_key="auroc"
  else
    metric_key="macro_f1"
  fi
  python probe/plot_probe.py \
    --summary_path "${RESULTS_ROOT}/${task}_${PROBE_TYPE}/summary.json" \
    --metric_key "$metric_key" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --position "$POSITION" \
    --title "Probe: ${task} (${MODEL_SLUG}, ${POSITION})" \
    --out_path "${RESULTS_ROOT}/${task}_${PROBE_TYPE}/${metric_key}_curve.png" \
    "${RESUME_ARGS[@]}"
done
