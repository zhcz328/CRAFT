MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Hulu-Med-4B}"

python filter_fine_grained.py \
  --in_csv ./data/slake_closed_all.csv \
  --model "$MODEL_PATH" \
  --out_csv ./data/slake_nc_cc_both_correct.csv \
  --dtype fp16 \
  --device_map auto \
  --resize_max_side 672 \
  --limit 0 \
  --image_root /root/autodl-tmp/data/SLAKE/imgs
