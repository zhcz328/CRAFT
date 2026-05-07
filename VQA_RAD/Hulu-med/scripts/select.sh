python select_heads_merged_unique_layers.py \
  --input_json /root/logit_lens/cross_modal/Hulu-med/result_train_hulumed4b_before_answer/headscan_vqarad_mm_hulumed4b_before_answer/head_scan_merged_unique_layers.json \
  --source results \
  --mode conflict_specific \
  --eff_min 0.25 \
  --base_max 0.25 \
  --topk 50 \
  --out_json /root/logit_lens/cross_modal/Hulu-med/result_train_hulumed4b_before_answer/headscan_vqarad_mm_hulumed4b_before_answer/selected_heads_stable_hulumed4b.json \
  --out_csv /root/logit_lens/cross_modal/Hulu-med/result_train_hulumed4b_before_answer/headscan_vqarad_mm_hulumed4b_before_answer/selected_heads_stable_hulumed4b.csv
