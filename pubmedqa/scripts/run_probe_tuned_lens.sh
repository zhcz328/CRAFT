#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITION="${POSITION:-before_question}"
PROBE_TYPE="${PROBE_TYPE:-linear}"

FEATURE_DTYPE="${FEATURE_DTYPE:-bfloat16}"
FEATURE_SAVE_DTYPE="${FEATURE_SAVE_DTYPE:-float16}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"

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
ENABLE_THINKING="${ENABLE_THINKING:-0}"

TRAIN_RECORDS="${TRAIN_RECORDS:-$(join_output_path "data/train_yesno.train.json")}"
VAL_RECORDS="${VAL_RECORDS:-$(join_output_path "data/train_yesno.val.json")}"
CONFLICT_POSITIONS="${CONFLICT_POSITIONS:-$(join_output_path "result_all_positions_yesno/${MODEL_SLUG}/${POSITION}/preds.jsonl")}"

PROBE_DATA_DIR="${PROBE_DATA_DIR:-$(join_output_path "probe/data")}"
PROBE_FEATURE_DIR="${PROBE_FEATURE_DIR:-$(join_output_path "probe/features")}"
PROBE_RESULTS_DIR="${PROBE_RESULTS_DIR:-$(join_output_path "probe/results")}"

LENS_DATA_DIR="${LENS_DATA_DIR:-$(join_output_path "tuned_lens/data")}"
LENS_RESULTS_DIR="${LENS_RESULTS_DIR:-$(join_output_path "tuned_lens/results")}"

PROBE_DATA_TRAIN="${PROBE_DATA_TRAIN:-${PROBE_DATA_DIR}/train_pair_stratified_${POSITION}.jsonl}"
PROBE_DATA_VAL="${PROBE_DATA_VAL:-${PROBE_DATA_DIR}/val_pair_stratified_${POSITION}.jsonl}"
PROBE_FEATURE_TRAIN="${PROBE_FEATURE_TRAIN:-${PROBE_FEATURE_DIR}/train_pair_stratified_${POSITION}.pt}"
PROBE_FEATURE_VAL="${PROBE_FEATURE_VAL:-${PROBE_FEATURE_DIR}/val_pair_stratified_${POSITION}.pt}"
PROBE_CONFLICT_OUT="${PROBE_CONFLICT_OUT:-${PROBE_RESULTS_DIR}/conflict_${PROBE_TYPE}_${POSITION}}"
PROBE_FOLLOW_OUT="${PROBE_FOLLOW_OUT:-${PROBE_RESULTS_DIR}/follow_conflict_${PROBE_TYPE}_${POSITION}}"

LENS_DATA_TRAIN="${LENS_DATA_TRAIN:-${LENS_DATA_DIR}/train_pair_stratified_${POSITION}.jsonl}"
LENS_DATA_VAL="${LENS_DATA_VAL:-${LENS_DATA_DIR}/val_pair_stratified_${POSITION}.jsonl}"
LENS_TRAIN_OUT="${LENS_TRAIN_OUT:-${LENS_RESULTS_DIR}/train_${POSITION}}"
LENS_CKPT="${LENS_CKPT:-${LENS_TRAIN_OUT}/tuned_lens.pt}"
LENS_TRAJ_OUT="${LENS_TRAJ_OUT:-${LENS_RESULTS_DIR}/trajectory_${POSITION}}"

mkdir -p "$PROBE_FEATURE_DIR" "$PROBE_RESULTS_DIR" "$LENS_RESULTS_DIR" "$PROBE_DATA_DIR" "$LENS_DATA_DIR"

run_cmd() {
  echo ""
  echo "==> $*"
  "$@"
}

if [[ "$RUN_PREPARE_DATA" == "1" ]]; then
  run_cmd "$PYTHON_BIN" probe/prepare_data.py \
    --train_records "$TRAIN_RECORDS" \
    --val_records "$VAL_RECORDS" \
    --conflict_positions "$CONFLICT_POSITIONS" \
    --position "$POSITION" \
    --out_dir "$PROBE_DATA_DIR"

  run_cmd "$PYTHON_BIN" tuned_lens/prepare_data.py \
    --train_records "$TRAIN_RECORDS" \
    --val_records "$VAL_RECORDS" \
    --conflict_positions "$CONFLICT_POSITIONS" \
    --position "$POSITION" \
    --out_dir "$LENS_DATA_DIR"
fi

if [[ "$RUN_PROBE" == "1" ]]; then
  THINK_ARGS=()
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINK_ARGS+=(--enable_thinking)
  fi

  if [[ "$RUN_PROBE_FEATURES" == "1" ]]; then
    run_cmd "$PYTHON_BIN" probe/extract_features.py \
      --manifest "$PROBE_DATA_TRAIN" \
      --model "$MODEL_PATH" \
      --out "$PROBE_FEATURE_TRAIN" \
      --batch_size "$FEATURE_BATCH_SIZE" \
      --dtype "$FEATURE_DTYPE" \
      --save_dtype "$FEATURE_SAVE_DTYPE" \
      "${THINK_ARGS[@]}"

    run_cmd "$PYTHON_BIN" probe/extract_features.py \
      --manifest "$PROBE_DATA_VAL" \
      --model "$MODEL_PATH" \
      --out "$PROBE_FEATURE_VAL" \
      --batch_size "$FEATURE_BATCH_SIZE" \
      --dtype "$FEATURE_DTYPE" \
      --save_dtype "$FEATURE_SAVE_DTYPE" \
      "${THINK_ARGS[@]}"
  fi

  if [[ "$RUN_PROBE_TRAIN" == "1" ]]; then
    run_cmd "$PYTHON_BIN" probe/train_probe.py \
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

    run_cmd "$PYTHON_BIN" probe/train_probe.py \
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

if [[ "$RUN_TUNED_LENS" == "1" ]]; then
  THINK_ARGS=()
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINK_ARGS+=(--enable_thinking)
  fi
  LAYER_ARGS=()
  if [[ -n "$LENS_LAYER_SPEC" ]]; then
    LAYER_ARGS+=(--layer_spec "$LENS_LAYER_SPEC")
  fi

  if [[ "$RUN_LENS_TRAIN" == "1" ]]; then
    run_cmd "$PYTHON_BIN" tuned_lens/train_tuned_lens.py \
      --train_manifest "$LENS_DATA_TRAIN" \
      --val_manifest "$LENS_DATA_VAL" \
      --model "$MODEL_PATH" \
      --out_dir "$LENS_TRAIN_OUT" \
      --prompt_types "$LENS_PROMPT_TYPES" \
      --translator_rank "$TRANSLATOR_RANK" \
      --translator_dtype "$TRANSLATOR_DTYPE" \
      --epochs "$LENS_EPOCHS" \
      --batch_size "$LENS_BATCH_SIZE" \
      --lr "$LENS_LR" \
      --weight_decay "$LENS_WEIGHT_DECAY" \
      "${LAYER_ARGS[@]}" \
      "${THINK_ARGS[@]}"
  fi

  if [[ "$RUN_LENS_ANALYZE" == "1" ]]; then
    run_cmd "$PYTHON_BIN" tuned_lens/analyze_trajectories.py \
      --manifest "$LENS_DATA_VAL" \
      --model "$MODEL_PATH" \
      --lens_ckpt "$LENS_CKPT" \
      --prompt_types conflict \
      --out_dir "$LENS_TRAJ_OUT" \
      --batch_size "$LENS_BATCH_SIZE" \
      --margin_eps "$MARGIN_EPS" \
      "${THINK_ARGS[@]}"
  fi
fi
