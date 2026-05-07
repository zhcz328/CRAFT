#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
POSITION="${POSITION:-before_question}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/probe/${MODEL_SLUG}}"
FEATURE_DIR="${FEATURE_DIR:-${PROBE_ROOT}/features}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROBE_ROOT}/results/${POSITION}}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"
PLOT_MODE="${PLOT_MODE:-mean}"
SCORE_TYPE="${SCORE_TYPE:-both}"
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
  python probe/plot_sample_trajectory.py \
    --features "${FEATURE_DIR}/val_${POSITION}.pt" \
    --probe_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}" \
    --task "$task" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --position "$POSITION" \
    --plot_mode "$PLOT_MODE" \
    --score_type "$SCORE_TYPE" \
    --out_png "${RESULTS_ROOT}/${task}_${PROBE_TYPE}/mean_trajectory.png" \
    --out_json "${RESULTS_ROOT}/${task}_${PROBE_TYPE}/mean_trajectory.json" \
    "${RESUME_ARGS[@]}"
done
