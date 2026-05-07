#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
DATASET_NAME="${DATASET_NAME:-gqa}"
DATA_ROOT="${DATA_ROOT:-./data/${MODEL_SLUG}/${DATASET_NAME}}"
DATA_CSV="${DATA_CSV:-${DATA_ROOT}/${DATASET_NAME}_nc_cc_both_correct_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bf16}"
TRACE_MODE="${TRACE_MODE:-conflict}"
POSITION="${POSITION:-before_question}"
METRICS="${METRICS:-follow_conflict}"
RESULT_ROOT="${RESULT_ROOT:-${MODEL_SLUG}/${DATASET_NAME}/result_${POSITION}_slake}"
PLAN_PATH="${PLAN_PATH:-${RESULT_ROOT}/trace_conflict_scan_plan.json}"
PLAN_OUT_DIR="${PLAN_OUT_DIR:-${RESULT_ROOT}/headscan_slake_mm_accel}"
LIMIT="${LIMIT:-0}"
COARSE_LIMIT="${COARSE_LIMIT:-128}"
COARSE_TOPK="${COARSE_TOPK:-36}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_IDX="${SHARD_IDX:-0}"
MERGE_SHARDS="${MERGE_SHARDS:-0}"
PARALLEL_STAGE="${PARALLEL_STAGE:-full}"
RESUME="${RESUME:-0}"
QWEN_PREFILL_BATCH_SIZE="${QWEN_PREFILL_BATCH_SIZE:-0}"
HEAD_ABLATION_IMPL="${HEAD_ABLATION_IMPL:-mask}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-1}"
QWEN_VL_VISION_CACHE="${QWEN_VL_VISION_CACHE:-0}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
MERGE_ARGS=()
if [[ "$MERGE_SHARDS" == "1" ]]; then
  MERGE_ARGS+=(--merge_shards)
fi
QWEN_PREFILL_ARGS=()
if [[ "$QWEN_PREFILL_BATCH_SIZE" != "0" ]]; then
  QWEN_PREFILL_ARGS+=(--qwen_prefill_batch_size "$QWEN_PREFILL_BATCH_SIZE")
fi
HEAD_ABLATION_ARGS=(--head_ablation_impl "$HEAD_ABLATION_IMPL")
ATTENTION_BACKEND_ARGS=(--attention_backend "$ATTENTION_BACKEND")
HEAD_BATCH_ARGS=(--head_batch_size "$HEAD_BATCH_SIZE")
QWEN_VL_VISION_CACHE_ARGS=()
if [[ "$QWEN_VL_VISION_CACHE" == "1" ]]; then
  QWEN_VL_VISION_CACHE_ARGS+=(--qwen_vl_vision_cache)
fi

python head_scan_slake_mm_fastcache_accel.py \
  --data_csv "$DATA_CSV" \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --trace_mode "$TRACE_MODE" \
  --position "$POSITION" \
  --metrics "$METRICS" \
  --plan "$PLAN_PATH" \
  --plan_out_dir "$PLAN_OUT_DIR" \
  --limit "$LIMIT" \
  --coarse_limit "$COARSE_LIMIT" \
  --coarse_topk "$COARSE_TOPK" \
  --num_shards "$NUM_SHARDS" \
  --shard_idx "$SHARD_IDX" \
  --parallel_stage "$PARALLEL_STAGE" \
  "${HEAD_ABLATION_ARGS[@]}" \
  "${ATTENTION_BACKEND_ARGS[@]}" \
  "${HEAD_BATCH_ARGS[@]}" \
  "${QWEN_PREFILL_ARGS[@]}" \
  "${QWEN_VL_VISION_CACHE_ARGS[@]}" \
  "${MERGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}"


