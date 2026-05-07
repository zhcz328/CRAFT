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
GPU_IDS="${GPU_IDS:-0,1}"
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ID_ARR[@]}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/${MODEL_SLUG}/${DATASET_NAME}/logs/head_scan_accel_parallel/${POSITION}}"
mkdir -p "$LOG_DIR"

if [[ "${#GPU_ID_ARR[@]}" -lt "$NUM_SHARDS" ]]; then
  echo "GPU_IDS provides ${#GPU_ID_ARR[@]} devices but NUM_SHARDS=$NUM_SHARDS" >&2
  exit 1
fi

kill_stage_pids() {
  local current_pid="${1:-}"
  shift || true
  local pid
  for pid in "$@"; do
    if [[ -n "$pid" && "$pid" != "$current_pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

run_parallel_stage() {
  local stage_name="$1"
  local pids=()
  local shard_labels=()
  local log_paths=()
  for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
    local gpu_id="${GPU_ID_ARR[$shard_idx]}"
    local log_path="${LOG_DIR}/head_scan_accel_${stage_name}_shard${shard_idx}.log"
    echo "[launch] head_scan_accel ${stage_name} shard ${shard_idx}/${NUM_SHARDS} on GPU ${gpu_id} -> ${log_path}"
    (
      cd "$ROOT_DIR"
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      DEVICE="cuda:0" \
      NUM_SHARDS="$NUM_SHARDS" \
      SHARD_IDX="$shard_idx" \
      MERGE_SHARDS=0 \
      PARALLEL_STAGE="$stage_name" \
      bash "$ROOT_DIR/scripts/head_scan_accel.sh"
    ) >"$log_path" 2>&1 &
    pids+=($!)
    shard_labels+=("${shard_idx}/${NUM_SHARDS} on GPU ${gpu_id}")
    log_paths+=("$log_path")
  done

  local remaining="${#pids[@]}"
  local failure_pid=""
  local failure_label=""
  local failure_log=""
  local failure_code=1
  while [[ "$remaining" -gt 0 ]]; do
    local idx
    for idx in "${!pids[@]}"; do
      local pid="${pids[$idx]}"
      [[ -z "$pid" ]] && continue
      if ! kill -0 "$pid" 2>/dev/null; then
        local exit_code=0
        if ! wait "$pid"; then
          exit_code=$?
        fi
        pids[$idx]=""
        remaining=$((remaining - 1))
        if [[ "$exit_code" -ne 0 ]]; then
          failure_pid="$pid"
          failure_label="${shard_labels[$idx]}"
          failure_log="${log_paths[$idx]}"
          failure_code="$exit_code"
          kill_stage_pids "$failure_pid" "${pids[@]}"
          wait || true
          echo "[error] stage ${stage_name} failed at shard ${failure_label} (pid ${failure_pid}, exit ${failure_code})" >&2
          echo "[error] see log: ${failure_log}" >&2
          exit "$failure_code"
        fi
      fi
    done
    sleep 1
  done
}

run_merge_stage() {
  local stage_name="$1"
  local log_path="${LOG_DIR}/head_scan_accel_${stage_name}.log"
  echo "[merge] head_scan_accel ${stage_name} -> ${log_path}"
  if ! NUM_SHARDS="$NUM_SHARDS" SHARD_IDX=0 MERGE_SHARDS=0 PARALLEL_STAGE="$stage_name" bash "$ROOT_DIR/scripts/head_scan_accel.sh" >"$log_path" 2>&1; then
    echo "[error] merge stage ${stage_name} failed" >&2
    echo "[error] see log: ${log_path}" >&2
    exit 1
  fi
}

run_parallel_stage "coarse"
run_merge_stage "merge_coarse"
run_parallel_stage "coarse_rerank"
run_merge_stage "merge_coarse_rerank"
run_parallel_stage "refine"
run_merge_stage "merge_refine"



