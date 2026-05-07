#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITIONS="${POSITIONS:-before_answer}"
PAIR_FILE_STEM="${PAIR_FILE_STEM:-kept_pairs_a12_b10_all}"
DATA_SPLITS="${DATA_SPLITS:-train,val}"
PAIRS_PATH="${PAIRS_PATH:-}"
HEAD_FILE="${HEAD_FILE:-${HEAD_GROUPS_PATH:-}}"
ABLATE_OUT="${ABLATE_OUT:-}"
MASK_SCOPE="${MASK_SCOPE:-all}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
MAX_PAIRS="${MAX_PAIRS:-0}"

model_hint_lc="$(printf '%s %s' "${MODEL_NAME:-}" "${MODEL_PATH:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "$model_hint_lc" == *"llama32_3b"* ]] || [[ "$model_hint_lc" == *"llama-3.2"* ]]; then
  # Avoid duplicating subdir when OUTPUT_ROOT already points to llama32_3b.
  output_root_norm="${OUTPUT_ROOT%/}"
  if [[ -n "${output_root_norm:-}" && "${output_root_norm##*/}" == "llama32_3b" ]]; then
    MODEL_SUBDIR=""
  else
    MODEL_SUBDIR="llama32_3b/"
  fi
else
  MODEL_SUBDIR=""
fi

MODEL_DATA_DIR="${MODEL_DATA_DIR:-$(join_output_path "${MODEL_SUBDIR}data")}"
MODEL_RESULT_DIR="${MODEL_RESULT_DIR:-$(join_output_path "${MODEL_SUBDIR}result/${POSITIONS}")}"

IFS=',' read -r -a POSITIONS_ARR <<< "$POSITIONS"
NORMALIZED_POSITIONS=()
for pos in "${POSITIONS_ARR[@]}"; do
  pos="$(echo "$pos" | xargs)"
  [[ -z "$pos" ]] && continue
  NORMALIZED_POSITIONS+=("$pos")
done
if [[ "${#NORMALIZED_POSITIONS[@]}" -eq 0 ]]; then
  echo "[error] no valid positions in POSITIONS=${POSITIONS}" >&2
  exit 1
fi

# Safety fallback: if auto-detected directory has no expected split file but llama32_3b does, switch to llama32_3b.
if [[ -z "${PAIRS_PATH:-}" ]]; then
  default_train="${MODEL_DATA_DIR}/${PAIR_FILE_STEM}_train.jsonl"
  llama_train="$(join_output_path "llama32_3b/data/${PAIR_FILE_STEM}_train.jsonl")"
  if [[ ! -f "$default_train" && -f "$llama_train" ]]; then
    MODEL_SUBDIR="llama32_3b/"
    MODEL_DATA_DIR="$(join_output_path "${MODEL_SUBDIR}data")"
    MODEL_RESULT_DIR="$(join_output_path "${MODEL_SUBDIR}result/${POSITIONS}")"
    echo "[info] auto-switched to llama32_3b directories based on data availability"
  fi
fi

run_one_split() {
  local split="$1"
  local pairs_path="$2"
  local out_path="$3"
  local positions_arg="${4:-$POSITIONS}"
  local head_file_arg="${5:-$HEAD_FILE}"

  ensure_parent_dir "$out_path"

  CMD=(
    "$PYTHON_BIN" ablate_head_inf.py
    --pairs "$pairs_path"
    --model "$MODEL_PATH"
    --positions "$positions_arg"
    --out "$out_path"
    --mask_scope "$MASK_SCOPE"
    --dtype "$DTYPE"
    --device_map "$DEVICE_MAP"
  )

  if [[ -n "$head_file_arg" ]]; then
    CMD+=(--head_file "$head_file_arg")
  fi
  if [[ "$MAX_PAIRS" != "0" ]]; then
    CMD+=(--max_pairs "$MAX_PAIRS")
  fi
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    CMD+=(--enable_thinking)
  fi

  echo "==> [split=${split}] ${CMD[*]}"
  "${CMD[@]}"
}

resolve_head_file_for_position() {
  local position="$1"
  local result_dir
  result_dir="$(join_output_path "${MODEL_SUBDIR}result/${position}")"

  local scan_dir
  for scan_dir in \
    "headscan_rounds_top50_inf" \
    "headscan_rounds_top30_inf" \
    "headscan_rounds_top30" \
    "headscan_rounds_v2" \
    "headscan_rounds"; do
    if [[ -f "${result_dir}/${scan_dir}/selected_heads.json" ]]; then
      printf '%s' "${result_dir}/${scan_dir}/selected_heads.json"
      return 0
    fi
    if [[ -f "${result_dir}/${scan_dir}/head_groups.json" ]]; then
      printf '%s' "${result_dir}/${scan_dir}/head_groups.json"
      return 0
    fi
  done

  echo "[error] could not find head file for position=${position} under ${result_dir}" >&2
  return 1
}

run_default_splits_for_position() {
  local position="$1"
  local result_dir
  local position_head_file
  result_dir="$(join_output_path "${MODEL_SUBDIR}result/${position}")"
  position_head_file="$(resolve_head_file_for_position "$position")"

  for split in "${SPLITS_ARR[@]}"; do
    split="$(echo "$split" | xargs)"
    [[ -z "$split" ]] && continue
    pairs_path="${MODEL_DATA_DIR}/${PAIR_FILE_STEM}_${split}.jsonl"
    if [[ ! -f "$pairs_path" ]]; then
      echo "[warn] skip position=${position} split=${split}: missing file $pairs_path"
      continue
    fi
    out_path="${result_dir}/conflict_retest_ablated_heads_inf_${position}_${split}.jsonl"
    run_one_split "$split" "$pairs_path" "$out_path" "$position" "$position_head_file"
  done
}

if [[ -n "$PAIRS_PATH" ]]; then
  # Backward-compatible single-run mode when user explicitly provides PAIRS_PATH.
  if [[ -n "$ABLATE_OUT" ]]; then
    OUT_PATH="$ABLATE_OUT"
  else
    OUT_PATH="${MODEL_RESULT_DIR}/conflict_retest_ablated_heads_inf_${POSITIONS//,/__}.jsonl"
  fi
  run_one_split "custom" "$PAIRS_PATH" "$OUT_PATH"
  exit 0
fi

# Default mode: run both train and val from model-specific data directory.
IFS=',' read -r -a SPLITS_ARR <<< "$DATA_SPLITS"
if [[ "${#NORMALIZED_POSITIONS[@]}" -gt 1 && -z "$HEAD_FILE" && -z "$ABLATE_OUT" ]]; then
  echo "[info] multiple positions detected; running each position with its own inferred head file"
  for position in "${NORMALIZED_POSITIONS[@]}"; do
    run_default_splits_for_position "$position"
  done
  exit 0
fi

for split in "${SPLITS_ARR[@]}"; do
  split="$(echo "$split" | xargs)"
  [[ -z "$split" ]] && continue
  pairs_path="${MODEL_DATA_DIR}/${PAIR_FILE_STEM}_${split}.jsonl"
  if [[ ! -f "$pairs_path" ]]; then
    echo "[warn] skip split=${split}: missing file $pairs_path"
    continue
  fi
  out_path="${MODEL_RESULT_DIR}/conflict_retest_ablated_heads_inf_${POSITIONS//,/__}_${split}.jsonl"
  run_one_split "$split" "$pairs_path" "$out_path"
done
