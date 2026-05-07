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
TUNED_LENS_ROOT="${TUNED_LENS_ROOT:-${SAVE_ROOT}/tuned_lens/${MODEL_SLUG}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROBE_ROOT}/results/${POSITION}}"
SCORES_ROOT="${SCORES_ROOT:-${PROBE_ROOT}/score_on_manifest/${POSITION}}"
MANIFEST="${MANIFEST:-${TUNED_LENS_ROOT}/data/val_${POSITION}.jsonl}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_TASKS="${PROBE_TASKS:-conflict hallucination}"
PROMPT_TYPES="${PROMPT_TYPES:-ic}"
GROUP_FIELD="${GROUP_FIELD:-hallucination_label}"
GROUP_VALUES="${GROUP_VALUES:-}"
GROUP_NAME="${GROUP_NAME:-ic}"
SCORE_TYPE="${SCORE_TYPE:-logit}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
MASK_SCALE="${MASK_SCALE:-1.0}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
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
  python probe/score_probe_on_manifest.py \
    --manifest "$MANIFEST" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --probe_dir "${RESULTS_ROOT}/${task}_${PROBE_TYPE}" \
    --probe_label "${task}_probe" \
    --prompt_types "$PROMPT_TYPES" \
    --group_field "$GROUP_FIELD" \
    --group_values "$GROUP_VALUES" \
    --group_name "$GROUP_NAME" \
    --score_type "$SCORE_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --image_size "$IMAGE_SIZE" \
    --mask_scale "$MASK_SCALE" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --max_samples "$MAX_SAMPLES" \
    --out_png "${SCORES_ROOT}/${task}_${PROBE_TYPE}_${PROMPT_TYPES}_${SCORE_TYPE}.png" \
    --out_json "${SCORES_ROOT}/${task}_${PROBE_TYPE}_${PROMPT_TYPES}_${SCORE_TYPE}.json" \
    "${RESUME_ARGS[@]}"
done
