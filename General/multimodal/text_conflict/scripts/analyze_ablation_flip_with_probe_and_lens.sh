#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
POSITION="${POSITION:-before_question}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/${DATASET_NAME}/result_${POSITION}_slake}"
MASK_SCOPE="${MASK_SCOPE:-ctx_only}"
ABLATE_VARIANT="${ABLATE_VARIANT:-ablate}"
ABLATE_SPLIT="${ABLATE_SPLIT:-val}"
SCORE_TYPE="${SCORE_TYPE:-logit}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/probe/${MODEL_SLUG}}"
PROBE_RESULTS_ROOT="${PROBE_RESULTS_ROOT:-${PROBE_ROOT}/results/${POSITION}}"
PROBE_TYPE="${PROBE_TYPE:-linear}"
TL_ROOT="${TL_ROOT:-/root/autodl-tmp/tuned_lens/${MODEL_SLUG}}"
TL_RESULTS_ROOT="${TL_RESULTS_ROOT:-${TL_ROOT}/results/${POSITION}}"
LENS_RUN_NAME="${LENS_RUN_NAME:-train_idreg}"
LENS_CKPT="${LENS_CKPT:-${TL_RESULTS_ROOT}/${LENS_RUN_NAME}/tuned_lens.pt}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/ablation_flip_${ABLATE_VARIANT}_${MASK_SCOPE}_${ABLATE_SPLIT}}"
RESUME="${RESUME:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

case "$ABLATE_VARIANT" in
  ablate)
    ABLATION_JSON_DEFAULT="${RESULT_ROOT}/ablate/ablate_selected_heads_${MASK_SCOPE}_${ABLATE_SPLIT}.json"
    ;;
  ablate_random)
    ABLATION_JSON_DEFAULT="${RESULT_ROOT}/ablate_random/ablate_random_${MASK_SCOPE}_${ABLATE_SPLIT}.json"
    ;;
  *)
    echo "Unsupported ABLATE_VARIANT: $ABLATE_VARIANT" >&2
    echo "Use one of: ablate, ablate_random" >&2
    exit 1
    ;;
esac
ABLATION_JSON="${ABLATION_JSON:-$ABLATION_JSON_DEFAULT}"

if [[ -n "${PROBE_TASKS:-}" ]]; then
  RAW_PROBE_TASKS="${PROBE_TASKS//,/ }"
elif [[ -n "${PROBE_TASK:-}" ]]; then
  RAW_PROBE_TASKS="${PROBE_TASK}"
else
  RAW_PROBE_TASKS="follow_conflict conflict"
fi

PROBE_TASK_LIST=()
for probe_task in $RAW_PROBE_TASKS; do
  case "$probe_task" in
    follow_conflict|conflict)
      PROBE_TASK_LIST+=("$probe_task")
      ;;
    *)
      echo "Unsupported PROBE_TASK/PROBE_TASKS item: $probe_task" >&2
      echo "Use one or more of: follow_conflict, conflict" >&2
      exit 1
      ;;
  esac
done

if [[ ${#PROBE_TASK_LIST[@]} -eq 0 ]]; then
  echo "No probe tasks selected." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

for probe_task in "${PROBE_TASK_LIST[@]}"; do
  probe_dir="${PROBE_RESULTS_ROOT}/${probe_task}_${PROBE_TYPE}"
  task_out_dir="${OUT_DIR}/${probe_task}"

  EXTRA_ARGS=()
  if [[ -n "${DATA_CSV:-}" ]]; then
    EXTRA_ARGS+=(--data_csv "$DATA_CSV")
  fi
  if [[ -n "${SELECTED_HEADS:-}" ]]; then
    EXTRA_ARGS+=(--selected_heads "$SELECTED_HEADS")
  fi

  python analyze_ablation_flip_with_probe_and_lens.py \
    --ablation_json "$ABLATION_JSON" \
    --model "$MODEL_PATH" \
    "${MODEL_NAME_ARGS[@]}" \
    --probe_dir "$probe_dir" \
    --lens_ckpt "$LENS_CKPT" \
    --out_dir "$task_out_dir" \
    --position "$POSITION" \
    --score_type "$SCORE_TYPE" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    "${EXTRA_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
done


