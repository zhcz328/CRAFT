MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Hulu-Med-4B}"
IMAGE_ROOT="${IMAGE_ROOT:-/root/autodl-tmp/data/SLAKE}"

python head_scan_slake_mm_fastcache_accel.py \
  --data_csv ./data/slake_nc_cc_both_correct_train.csv \
  --image_root "$IMAGE_ROOT" \
  --model "$MODEL_PATH" \
  --device auto \
  --dtype bf16 \
  --trace_mode conflict \
  --position before_answer \
  --metrics follow_conflict \
  --plan ./result_slake_hulumed4b_before_answer/trace_conflict_scan_plan.json \
  --plan_out_dir ./result_slake_hulumed4b_before_answer/headscan_slake_mm_accel_hulumed4b_96g \
  --limit 0 \
  --coarse_limit 256 \
  --coarse_topk 50
