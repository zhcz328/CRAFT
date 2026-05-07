#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
FEATURE_DIR="${FEATURE_DIR:-$(join_output_path "probe/features")}"
RESULTS_ROOT="${RESULTS_ROOT:-$(join_output_path "probe/results")}"
TRAIN_FEATURES="${TRAIN_FEATURES:-${FEATURE_DIR}/train_pair_stratified_${POSITION}.pt}"
VAL_FEATURES="${VAL_FEATURES:-${FEATURE_DIR}/val_pair_stratified_${POSITION}.pt}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

mkdir -p "$RESULTS_ROOT"

for task in $PROBE_TASKS; do
  "$PYTHON_BIN" probe/train_probe.py \
    --train_features "$TRAIN_FEATURES" \
    --val_features "$VAL_FEATURES" \
    --task "$task" \
    --probe_type "$PROBE_TYPE" \
    --mlp_hidden_dim "$MLP_HIDDEN_DIM" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --out_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}"
done
