#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/model_env.sh"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-hulumed-4b}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_SLUG="$(resolve_model_slug "${MODEL_NAME:-$MODEL_PATH}")"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
MODEL_NAME_ARGS=()
if [[ -n "$MODEL_NAME" ]]; then
  MODEL_NAME_ARGS+=(--model_name "$MODEL_NAME")
fi

python head_scan_slake_mm_fastcache_accel.py \
  --data_csv ./data/slake_nc_cc_both_correct_train.csv \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  "${MODEL_NAME_ARGS[@]}" \
  --device auto \
  --dtype bf16 \
  --trace_mode conflict \
  --position before_question \
  --metrics follow_conflict \
  --plan "${MODEL_SLUG}/result_tmp300/trace_conflict_scan_plan.json" \
  --plan_out_dir "${MODEL_SLUG}/result_tmp300/headscan_slake_mm_accel_96g" \
  --limit 0 \
  --coarse_limit 256 \
  --coarse_topk 96
