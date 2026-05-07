#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/model_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
POSITION="${POSITION:-before_question}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

TRAIN_CSV="${TRAIN_CSV:-data/slake_nc_cc_both_correct_train.csv}"
VAL_CSV="${VAL_CSV:-data/slake_nc_cc_both_correct_val.csv}"
PREDS_JSONL="${PREDS_JSONL:-}"

FEATURE_DTYPE="${FEATURE_DTYPE:-bfloat16}"
FEATURE_SAVE_DTYPE="${FEATURE_SAVE_DTYPE:-float16}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-672}"

PROBE_TYPE="${PROBE_TYPE:-linear}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
PROBE_PATIENCE="${PROBE_PATIENCE:-10}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-64}"
PROBE_LR="${PROBE_LR:-1e-3}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-1e-4}"

LENS_EPOCHS="${LENS_EPOCHS:-3}"
LENS_BATCH_SIZE="${LENS_BATCH_SIZE:-2}"
LENS_LR="${LENS_LR:-5e-4}"
LENS_WEIGHT_DECAY="${LENS_WEIGHT_DECAY:-1e-5}"
TRANSLATOR_RANK="${TRANSLATOR_RANK:-0}"
TRANSLATOR_DTYPE="${TRANSLATOR_DTYPE:-float32}"
LENS_PROMPT_TYPES="${LENS_PROMPT_TYPES:-base,support,conflict}"
LENS_LAYER_SPEC="${LENS_LAYER_SPEC:-}"
MARGIN_EPS="${MARGIN_EPS:-0.1}"

RUN_PREPARE_DATA="${RUN_PREPARE_DATA:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_TUNED_LENS="${RUN_TUNED_LENS:-1}"
RUN_PROBE_FEATURES="${RUN_PROBE_FEATURES:-1}"
RUN_PROBE_TRAIN="${RUN_PROBE_TRAIN:-1}"
RUN_LENS_TRAIN="${RUN_LENS_TRAIN:-1}"
RUN_LENS_ANALYZE="${RUN_LENS_ANALYZE:-1}"

PROBE_DATA_DIR="${MODEL_SLUG}/probe/data"
PROBE_FEATURE_DIR="${MODEL_SLUG}/probe/features"
PROBE_RESULT_DIR="${MODEL_SLUG}/probe/results"
LENS_DATA_DIR="${MODEL_SLUG}/tuned_lens/data"
LENS_RESULT_DIR="${MODEL_SLUG}/tuned_lens/results"

PROBE_DATA_TRAIN="${PROBE_DATA_DIR}/train_${POSITION}.jsonl"
PROBE_DATA_VAL="${PROBE_DATA_DIR}/val_${POSITION}.jsonl"
PROBE_FEATURE_TRAIN="${PROBE_FEATURE_DIR}/train_${POSITION}.pt"
PROBE_FEATURE_VAL="${PROBE_FEATURE_DIR}/val_${POSITION}.pt"
PROBE_CONFLICT_OUT="${PROBE_RESULT_DIR}/conflict_${PROBE_TYPE}_${POSITION}"
PROBE_FOLLOW_OUT="${PROBE_RESULT_DIR}/follow_${PROBE_TYPE}_${POSITION}"

LENS_DATA_TRAIN="${LENS_DATA_DIR}/train_${POSITION}.jsonl"
LENS_DATA_VAL="${LENS_DATA_DIR}/val_${POSITION}.jsonl"
LENS_TRAIN_OUT="${LENS_RESULT_DIR}/train_${POSITION}"
LENS_CKPT="${LENS_TRAIN_OUT}/tuned_lens.pt"
LENS_TRAJ_OUT="${LENS_RESULT_DIR}/trajectory_${POSITION}"

mkdir -p "$PROBE_FEATURE_DIR" "$PROBE_RESULT_DIR" "$LENS_RESULT_DIR"

run_cmd() {
  echo ""
  echo "==> $*"
  "$@"
}

should_run_if_missing() {
  local target="$1"
  if [[ -e "$target" ]]; then
    echo "[skip] exists: $target"
    return 1
  fi
  return 0
}

prepare_args=(
  --train_csv "$TRAIN_CSV"
  --val_csv "$VAL_CSV"
  --model "$MODEL_PATH"
  "${MODEL_NAME_ARGS[@]}"
  --position "$POSITION"
)
if [[ -n "$PREDS_JSONL" ]]; then
  prepare_args+=(--preds_jsonl "$PREDS_JSONL")
fi

if [[ "$RUN_PREPARE_DATA" == "1" ]]; then
  if should_run_if_missing "${PROBE_DATA_DIR}/summary_${POSITION}.json"; then
    run_cmd $PYTHON_BIN probe/prepare_data.py \
      "${prepare_args[@]}" \
      --out_dir "$PROBE_DATA_DIR"
  fi
  if should_run_if_missing "${LENS_DATA_DIR}/summary_${POSITION}.json"; then
    run_cmd $PYTHON_BIN tuned_lens/prepare_data.py \
      "${prepare_args[@]}" \
      --out_dir "$LENS_DATA_DIR"
  fi
fi

if [[ "$RUN_PROBE" == "1" ]]; then
  if [[ "$RUN_PROBE_FEATURES" == "1" ]]; then
    if should_run_if_missing "$PROBE_FEATURE_TRAIN"; then
      run_cmd $PYTHON_BIN probe/extract_features.py \
        --manifest "$PROBE_DATA_TRAIN" \
        --model "$MODEL_PATH" \
        "${MODEL_NAME_ARGS[@]}" \
        --out "$PROBE_FEATURE_TRAIN" \
        --batch_size "$FEATURE_BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --dtype "$FEATURE_DTYPE" \
        --save_dtype "$FEATURE_SAVE_DTYPE"
    fi
    if should_run_if_missing "$PROBE_FEATURE_VAL"; then
      run_cmd $PYTHON_BIN probe/extract_features.py \
        --manifest "$PROBE_DATA_VAL" \
        --model "$MODEL_PATH" \
        "${MODEL_NAME_ARGS[@]}" \
        --out "$PROBE_FEATURE_VAL" \
        --batch_size "$FEATURE_BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --dtype "$FEATURE_DTYPE" \
        --save_dtype "$FEATURE_SAVE_DTYPE"
    fi
  fi

  if [[ "$RUN_PROBE_TRAIN" == "1" ]]; then
    if should_run_if_missing "${PROBE_CONFLICT_OUT}/summary.json"; then
      run_cmd $PYTHON_BIN probe/train_probe.py \
        --train_features "$PROBE_FEATURE_TRAIN" \
        --val_features "$PROBE_FEATURE_VAL" \
        --task conflict \
        --probe_type "$PROBE_TYPE" \
        --epochs "$PROBE_EPOCHS" \
        --patience "$PROBE_PATIENCE" \
        --batch_size "$PROBE_BATCH_SIZE" \
        --lr "$PROBE_LR" \
        --weight_decay "$PROBE_WEIGHT_DECAY" \
        --out_dir "$PROBE_CONFLICT_OUT"
    fi

    if should_run_if_missing "${PROBE_FOLLOW_OUT}/summary.json"; then
      run_cmd $PYTHON_BIN probe/train_probe.py \
        --train_features "$PROBE_FEATURE_TRAIN" \
        --val_features "$PROBE_FEATURE_VAL" \
        --task follow_conflict \
        --probe_type "$PROBE_TYPE" \
        --epochs "$PROBE_EPOCHS" \
        --patience "$PROBE_PATIENCE" \
        --batch_size "$PROBE_BATCH_SIZE" \
        --lr "$PROBE_LR" \
        --weight_decay "$PROBE_WEIGHT_DECAY" \
        --out_dir "$PROBE_FOLLOW_OUT"
    fi
  fi
fi

if [[ "$RUN_TUNED_LENS" == "1" ]]; then
  if [[ "$RUN_LENS_TRAIN" == "1" ]]; then
    layer_args=()
    if [[ -n "$LENS_LAYER_SPEC" ]]; then
      layer_args+=(--layer_spec "$LENS_LAYER_SPEC")
    fi
    if should_run_if_missing "$LENS_CKPT"; then
      run_cmd $PYTHON_BIN tuned_lens/train_tuned_lens.py \
        --train_manifest "$LENS_DATA_TRAIN" \
        --val_manifest "$LENS_DATA_VAL" \
        --model "$MODEL_PATH" \
        "${MODEL_NAME_ARGS[@]}" \
        --out_dir "$LENS_TRAIN_OUT" \
        --prompt_types "$LENS_PROMPT_TYPES" \
        --translator_rank "$TRANSLATOR_RANK" \
        --translator_dtype "$TRANSLATOR_DTYPE" \
        --epochs "$LENS_EPOCHS" \
        --batch_size "$LENS_BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --lr "$LENS_LR" \
        --weight_decay "$LENS_WEIGHT_DECAY" \
        "${layer_args[@]}"
    fi
  fi

  if [[ "$RUN_LENS_ANALYZE" == "1" ]]; then
    if should_run_if_missing "${LENS_TRAJ_OUT}/summary.json"; then
      run_cmd $PYTHON_BIN tuned_lens/analyze_trajectories.py \
        --manifest "$LENS_DATA_VAL" \
        --model "$MODEL_PATH" \
        "${MODEL_NAME_ARGS[@]}" \
        --lens_ckpt "$LENS_CKPT" \
        --prompt_types conflict \
        --out_dir "$LENS_TRAJ_OUT" \
        --batch_size "$LENS_BATCH_SIZE" \
        --image_size "$IMAGE_SIZE" \
        --margin_eps "$MARGIN_EPS"
    fi
  fi
fi

echo ""
echo "Probe conflict summary: ${PROBE_CONFLICT_OUT}/summary.json"
echo "Probe follow summary:   ${PROBE_FOLLOW_OUT}/summary.json"
echo "Lens train summary:     ${LENS_TRAIN_OUT}/summary.json"
echo "Lens traj summary:      ${LENS_TRAJ_OUT}/summary.json"
