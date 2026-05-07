#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
source "$ROOT_DIR/scripts/model_env.sh"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
POSITION="${POSITION:-before_question}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/probe/${MODEL_SLUG}}"
FEATURE_DIR="${FEATURE_DIR:-${PROBE_ROOT}/features}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROBE_ROOT}/results/${POSITION}}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
MLP_HIDDEN_DIM="${MLP_HIDDEN_DIM:-256}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
RESUME="${RESUME:-0}"
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

for task in $PROBE_TASKS; do
  python probe/train_probe.py \
    --train_features "${FEATURE_DIR}/train_${POSITION}.pt" \
    --val_features "${FEATURE_DIR}/val_${POSITION}.pt" \
    --task "$task" \
    --probe_type "$PROBE_TYPE" \
    --mlp_hidden_dim "$MLP_HIDDEN_DIM" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --out_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}" \
    "${RESUME_ARGS[@]}"
done


