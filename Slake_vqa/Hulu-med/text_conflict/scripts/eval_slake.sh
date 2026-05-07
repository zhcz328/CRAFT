python eval_slake_mm_support_conflict_use_wrong_csv.py \
  --data_csv ./data/slake_nc_cc_both_correct.csv \
  --image_root /root/autodl-tmp/data/SLAKE/imgs \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out_dir eval-results_slake_hulumed4b_all \
  --position before_answer \
  --dtype auto \
  --device_map cuda:0 \
  --limit 0
