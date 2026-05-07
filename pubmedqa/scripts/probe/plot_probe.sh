#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
RESULTS_ROOT="${RESULTS_ROOT:-$(join_output_path "probe/results")}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"

for task in $PROBE_TASKS; do
  if [[ "$task" == "conflict" ]]; then
    metric_key="auroc"
  else
    metric_key="macro_f1"
  fi
  "$PYTHON_BIN" probe/plot_probe.py \
    --summary_path "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}/summary.json" \
    --metric_key "$metric_key" \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --position "$POSITION" \
    --title "Probe: ${task} (${OUTPUT_ROOT:-default}, ${POSITION})" \
    --out_path "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}/${metric_key}_curve.png"
done
