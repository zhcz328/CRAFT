#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/InternVL3_5-4B}"
MODEL_NAME="${MODEL_NAME:-internvl3_5-4b}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/logit_lens/Slake_vqa/image_conflict}"
DATA_ROOT="${DATA_ROOT:-/root/logit_lens/Slake_vqa/image_conflict/data/internvl35_4b}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"
TRACE_DIR="${TRACE_DIR:-${OUT_DIR}/layer_scale_sweep}"
ORIG_RESULT_ROOT="${ORIG_RESULT_ROOT:-/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake}"
POSITION="${POSITION:-image_conflict}"
TRACE_MODE="${TRACE_MODE:-conflict}"
MASK_SCALE="${MASK_SCALE:-1.0}"
METRICS="${METRICS:-follow_context}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-auto}"
LIMIT="${LIMIT:-0}"
LAYER_SCALES="${LAYER_SCALES:-0.0 0.2 0.4 0.6 0.8 1.0}"
TRAIN_CSV="${TRAIN_CSV:-${DATA_ROOT}/slake_nc_correct_ic_ready_train.csv}"
VAL_CSV="${VAL_CSV:-${DATA_ROOT}/slake_nc_correct_ic_ready_val.csv}"
HEADSCAN_PLAN_LAYER_KEY="${HEADSCAN_PLAN_LAYER_KEY:-core_layers}"
HEADSCAN_COARSE_LIMIT="${HEADSCAN_COARSE_LIMIT:-128}"
HEADSCAN_COARSE_TOPK="${HEADSCAN_COARSE_TOPK:-36}"
HEADSCAN_METRICS="${HEADSCAN_METRICS:-follow_context}"
SELECT_SOURCE="${SELECT_SOURCE:-results}"
SELECT_MODE="${SELECT_MODE:-conflict_specific}"
SELECT_TOPK="${SELECT_TOPK:-50}"
SELECT_TARGET_MIN="${SELECT_TARGET_MIN:-6}"
SELECT_TARGET_MAX="${SELECT_TARGET_MAX:-7}"
SELECT_EFF_FLOOR="${SELECT_EFF_FLOOR:-0.000001}"
SELECT_EFF_CAP="${SELECT_EFF_CAP:-0.2}"
SELECT_BASE_MAX_CAP="${SELECT_BASE_MAX_CAP:-0.2}"
ABLATE_METRICS="${ABLATE_METRICS:-follow_context}"
ABLATE_MASK_SCOPE="${ABLATE_MASK_SCOPE:-all}"
ABLATE_KEEP_MODE="${ABLATE_KEEP_MODE:-self}"
ABLATE_MAX_EXAMPLES="${ABLATE_MAX_EXAMPLES:-0}"

mkdir -p "$TRACE_DIR"

for scale in $LAYER_SCALES; do
  scale_tag="${scale//./p}"
  scale_dir="${TRACE_DIR}/layer_scale_${scale_tag}"
  trace_dir="${scale_dir}/trace_train"
  headscan_dir="${scale_dir}/headscan_train"
  select_dir="${scale_dir}/select_train"
  ablate_dir="${scale_dir}/ablate"
  mkdir -p "$trace_dir" "$headscan_dir" "$select_dir" "$ablate_dir"

  if [[ "$scale" == "0.2" ]]; then
    cp "${ORIG_RESULT_ROOT}/trace_image_conflict.json" "${trace_dir}/trace_train_layerscale_${scale_tag}.json"
    cp "${ORIG_RESULT_ROOT}/trace_image_conflict_scan_plan.json" "${trace_dir}/trace_train_layerscale_${scale_tag}_scan_plan.json"
    cp "${ORIG_RESULT_ROOT}/headscan_slake_mm_accel/head_scan_core_layers.json" "${headscan_dir}/head_scan_core_layers.json"
    cp "${ORIG_RESULT_ROOT}/headscan_slake_mm_accel/head_scan_summary.json" "${headscan_dir}/head_scan_summary.json"
    cp "${ORIG_RESULT_ROOT}/selected_heads_core_layers.json" "${select_dir}/selected_heads_auto.json"
    cp "${ORIG_RESULT_ROOT}/selected_heads_core_layers.csv" "${select_dir}/selected_heads_auto.csv"
    cp "${ORIG_RESULT_ROOT}/ablate/ablate_selected_heads_all_val.json" "${ablate_dir}/ablate_selected_heads_all_val.json"
    continue
  fi

  python3 "${ROOT_DIR}/layer_trace_with_unknown_summary.py" \
      --data_csv "$TRAIN_CSV" \
      --image_root "$IMAGE_ROOT" \
      --model "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --trace_mode "$TRACE_MODE" \
      --position "$POSITION" \
      --layer_scale "$scale" \
      --mask_scale "$MASK_SCALE" \
      --limit "$LIMIT" \
      --metrics "$METRICS" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      --out "${trace_dir}/trace_train_layerscale_${scale_tag}.json" \
      --plot_prefix "${trace_dir}/trace_train_layerscale_${scale_tag}" \
      --scan_plan_out "${trace_dir}/trace_train_layerscale_${scale_tag}_scan_plan.json"

  python3 /root/logit_lens/Slake_vqa/image_conflict/head_scan_slake_mm_fastcache_accel.py \
      --data_csv "$TRAIN_CSV" \
      --image_root "$IMAGE_ROOT" \
      --model "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --trace_mode "$TRACE_MODE" \
      --position "$POSITION" \
      --mask_scale "$MASK_SCALE" \
      --metrics "$HEADSCAN_METRICS" \
      --plan "${trace_dir}/trace_train_layerscale_${scale_tag}_scan_plan.json" \
      --plan_layer_key "$HEADSCAN_PLAN_LAYER_KEY" \
      --plan_out_dir "$headscan_dir" \
      --limit "$LIMIT" \
      --coarse_limit "$HEADSCAN_COARSE_LIMIT" \
      --coarse_topk "$HEADSCAN_COARSE_TOPK"

  python3 "${ROOT_DIR}/select_heads_auto.py" \
      --input_json "${headscan_dir}/head_scan_core_layers.json" \
      --source "$SELECT_SOURCE" \
      --mode "$SELECT_MODE" \
      --topk "$SELECT_TOPK" \
      --target_min "$SELECT_TARGET_MIN" \
      --target_max "$SELECT_TARGET_MAX" \
      --eff_floor "$SELECT_EFF_FLOOR" \
      --eff_cap "$SELECT_EFF_CAP" \
      --base_max_cap "$SELECT_BASE_MAX_CAP" \
      --out_json "${select_dir}/selected_heads_auto.json" \
      --out_csv "${select_dir}/selected_heads_auto.csv"

  n_selected="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("n_selected", 0)))' "${select_dir}/selected_heads_auto.json")"
  if [[ "$n_selected" -le 0 ]]; then
    python3 "${ROOT_DIR}/write_zero_ablate_summary.py" \
      --out_json "${ablate_dir}/ablate_selected_heads_all_val.json" \
      --selected_heads "${select_dir}/selected_heads_auto.json" \
      --data_csv "$VAL_CSV" \
      --model "$MODEL_PATH" \
      --model_name "$MODEL_NAME" \
      --trace_mode "$TRACE_MODE" \
      --position "$POSITION" \
      --mask_scale "$MASK_SCALE" \
      --metrics "$ABLATE_METRICS" \
      --mask_scope "$ABLATE_MASK_SCOPE" \
      --keep_mode "$ABLATE_KEEP_MODE" \
      --dtype "$DTYPE" \
      --device "$DEVICE"
  else
    python3 /root/logit_lens/Slake_vqa/image_conflict/ablate_head.py \
        --data_csv "$VAL_CSV" \
        --image_root "$IMAGE_ROOT" \
        --model "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --selected_heads "${select_dir}/selected_heads_auto.json" \
        --trace_mode "$TRACE_MODE" \
        --position "$POSITION" \
        --mask_scale "$MASK_SCALE" \
        --metrics "$ABLATE_METRICS" \
        --mask_scope "$ABLATE_MASK_SCOPE" \
        --keep_mode "$ABLATE_KEEP_MODE" \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --out_json "${ablate_dir}/ablate_selected_heads_all_val.json" \
        --max_examples "$ABLATE_MAX_EXAMPLES"
  fi
done

python3 "${ROOT_DIR}/summarize_layer_scale_sweep.py" \
  --input_dir "$TRACE_DIR" \
  --split train \
  --out_json "${TRACE_DIR}/layer_scale_sweep_train_summary.json" \
  --out_csv "${TRACE_DIR}/layer_scale_sweep_train_summary.csv" \
  --out_layer_csv "${TRACE_DIR}/layer_scale_sweep_train_layers.csv"

python3 "${ROOT_DIR}/summarize_ablate_sweep.py" \
  --input_dir "$TRACE_DIR" \
  --out_json "${TRACE_DIR}/layer_scale_sweep_val_ablate_summary.json" \
  --out_csv "${TRACE_DIR}/layer_scale_sweep_val_ablate_summary.csv"
