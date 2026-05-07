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
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/${MODEL_SLUG}/${DATASET_NAME}/logs/layer_trace_parallel/${POSITION}}"
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

pids=()
shard_labels=()
log_paths=()
for (( shard_idx=0; shard_idx<NUM_SHARDS; shard_idx++ )); do
  gpu_id="${GPU_ID_ARR[$shard_idx]}"
  log_path="${LOG_DIR}/layer_trace_shard${shard_idx}.log"
  echo "[launch] layer_trace shard ${shard_idx}/${NUM_SHARDS} on GPU ${gpu_id} -> ${log_path}"
  (
    cd "$ROOT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    DEVICE="cuda:0" \
    NUM_SHARDS="$NUM_SHARDS" \
    SHARD_IDX="$shard_idx" \
    MERGE_SHARDS=0 \
    bash "$ROOT_DIR/scripts/layer_trace.sh"
  ) >"$log_path" 2>&1 &
  pids+=($!)
  shard_labels+=("${shard_idx}/${NUM_SHARDS} on GPU ${gpu_id}")
  log_paths+=("$log_path")
done

remaining="${#pids[@]}"
while [[ "$remaining" -gt 0 ]]; do
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      exit_code=0
      if ! wait "$pid"; then
        exit_code=$?
      fi
      pids[$idx]=""
      remaining=$((remaining - 1))
      if [[ "$exit_code" -ne 0 ]]; then
        kill_stage_pids "$pid" "${pids[@]}"
        wait || true
        echo "[error] layer_trace failed at shard ${shard_labels[$idx]} (pid ${pid}, exit ${exit_code})" >&2
        echo "[error] see log: ${log_paths[$idx]}" >&2
        exit "$exit_code"
      fi
    fi
  done
  sleep 1
done

merge_log_path="${LOG_DIR}/layer_trace_merge.log"
echo "[merge] layer_trace shards -> ${merge_log_path}"
if ! NUM_SHARDS="$NUM_SHARDS" SHARD_IDX=0 MERGE_SHARDS=1 bash "$ROOT_DIR/scripts/layer_trace.sh" >"$merge_log_path" 2>&1; then
  echo "[error] layer_trace merge failed" >&2
  echo "[error] see log: ${merge_log_path}" >&2
  exit 1
fi



