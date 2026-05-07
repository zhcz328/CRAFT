#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

POSITION="${POSITION:-before_question}"
RESULTS_ROOT="${RESULTS_ROOT:-$(join_output_path "probe/results")}"
SCORES_ROOT="${SCORES_ROOT:-$(join_output_path "probe/score_on_manifest")}"
MANIFEST="${MANIFEST:-$(join_output_path "tuned_lens/data/val_pair_stratified_${POSITION}.jsonl")}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_TASKS="${PROBE_TASKS:-conflict follow_conflict}"
PROMPT_TYPES="${PROMPT_TYPES:-base}"
GROUP_FIELD="${GROUP_FIELD:-none}"
GROUP_VALUES="${GROUP_VALUES:-}"
GROUP_NAME="${GROUP_NAME:-base_all}"
SCORE_TYPE="${SCORE_TYPE:-logit}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

THINK_ARGS=()
if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS+=(--enable_thinking)
fi

for task in $PROBE_TASKS; do
  "$PYTHON_BIN" probe/score_probe_on_manifest.py \
    --manifest "$MANIFEST" \
    --model "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --probe_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}_${POSITION}" \
    --probe_label "${task}_probe" \
    --prompt_types "$PROMPT_TYPES" \
    --group_field "$GROUP_FIELD" \
    --group_values "$GROUP_VALUES" \
    --group_name "$GROUP_NAME" \
    --score_type "$SCORE_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_samples "$MAX_SAMPLES" \
    --out_png "${SCORES_ROOT}/${POSITION}/${task}_${PROBE_TYPE}_${PROMPT_TYPES}_${SCORE_TYPE}.png" \
    --out_json "${SCORES_ROOT}/${POSITION}/${task}_${PROBE_TYPE}_${PROMPT_TYPES}_${SCORE_TYPE}.json" \
    "${THINK_ARGS[@]}"
done
