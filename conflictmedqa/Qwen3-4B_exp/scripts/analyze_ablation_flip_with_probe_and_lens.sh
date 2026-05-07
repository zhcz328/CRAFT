#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

POSITION="${POSITION:-before_question}"
SCORE_TYPE="${SCORE_TYPE:-logit}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
PROBE_TYPE="${PROBE_TYPE:-linear}"

if [[ -z "${ABLATION_JSON:-}" ]]; then
  ABLATION_JSON="$(join_output_path "result/${POSITION}/conflict_retest_ablated_heads_inf_${POSITION}_val.jsonl")"
fi
if [[ -z "${BASELINE_JSON:-}" ]]; then
  BASELINE_JSON="$(join_output_path "result_all_positions/conflict_positions.jsonl")"
fi
if [[ -z "${PAIRS_PATH:-}" ]]; then
  PAIRS_PATH="$(join_output_path "data/kept_pairs_a12_b10_all_val.jsonl")"
fi
if [[ -z "${LENS_CKPT:-}" ]]; then
  LENS_CKPT="$(tuned_lens_save_path "$POSITION")/tuned_lens.pt"
fi

PROBE_TASKS_RAW="${PROBE_TASKS:-follow_conflict,conflict}"
ANALYZE_TAG="${ANALYZE_TAG:-$(basename "${ABLATION_JSON%.*}")}"
BASE_OUT_DIR="${BASE_OUT_DIR:-$(join_output_path "analysis/${ANALYZE_TAG}")}"

if [[ "$ENABLE_THINKING" == "1" ]]; then
  THINK_ARGS=(--enable_thinking)
else
  THINK_ARGS=()
fi

HEAD_GROUP_ARGS=()
if [[ -n "${HEAD_GROUPS_PATH:-}" ]]; then
  HEAD_GROUP_ARGS=(--head_groups "$HEAD_GROUPS_PATH")
fi

IFS=',' read -r -a PROBE_TASK_LIST <<< "$PROBE_TASKS_RAW"
for probe_task in "${PROBE_TASK_LIST[@]}"; do
  probe_task="$(echo "$probe_task" | xargs)"
  [[ -z "$probe_task" ]] && continue
  probe_name="$probe_task"
  if [[ "$probe_task" == "follow_conflict" ]]; then
    probe_name="follow"
  fi

  if [[ -z "${PROBE_DIR:-}" ]]; then
    resolved_probe_dir="$(probe_save_path "$POSITION")/${probe_name}_${PROBE_TYPE}"
  else
    resolved_probe_dir="$PROBE_DIR"
  fi
  task_out_dir="${BASE_OUT_DIR}/${probe_task}"
  mkdir -p "$task_out_dir"

  CMD=(
    "$PYTHON_BIN" analyze_ablation_flip_with_probe_and_lens.py
    --ablation_json "$ABLATION_JSON"
    --baseline_json "$BASELINE_JSON"
    --pairs "$PAIRS_PATH"
    --model "$MODEL_PATH"
    --probe_dir "$resolved_probe_dir"
    --lens_ckpt "$LENS_CKPT"
    --out_dir "$task_out_dir"
    --position "$POSITION"
    --score_type "$SCORE_TYPE"
    --dtype "$DTYPE"
    --device_map "$DEVICE_MAP"
    --batch_size "$BATCH_SIZE"
    --max_samples "$MAX_SAMPLES"
    "${HEAD_GROUP_ARGS[@]}"
    "${THINK_ARGS[@]}"
  )

  echo "==> ${CMD[*]}"
  "${CMD[@]}"
done
