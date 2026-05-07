#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
FEATURE_DIR="${FEATURE_DIR:-$(join_output_path "probe/features")}"
RESULTS_ROOT="${RESULTS_ROOT:-$(join_output_path "probe/results")}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"
PLOT_MODE="${PLOT_MODE:-mean}"
SCORE_TYPE="${SCORE_TYPE:-both}"

for task in $PROBE_TASKS; do
  "$PYTHON_BIN" probe/plot_sample_trajectory.py \
    --features "${FEATURE_DIR}/val_pair_stratified_${POSITION}.pt" \
    --probe_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}" \
    --task "$task" \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --position "$POSITION" \
    --plot_mode "$PLOT_MODE" \
    --score_type "$SCORE_TYPE" \
    --out_png "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}/mean_trajectory.png" \
    --out_json "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}/mean_trajectory.json"
done
