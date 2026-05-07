python eval_vqarad_mm_support_conflict_use_wrong_csv.py \
  --data_csv ./data/nc_cc_both_correct_rerun_tmp.csv \
  --image_root . \
  --model /root/autodl-tmp/hulumed-4B/Hulu-Med-4B \
  --out_dir eval-results_vqarad_hulumed4b_all \
  --position before_answer \
  --dtype auto \
  --device_map cuda:0 \
  --limit 0
