python select_heads_merged_unique_layers.py \
  --input_json /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_answer/headscan_slake_mm_accel_hulumed4b_96g/head_scan_merged_unique_layers.json \
  --source results \
  --mode conflict_specific \
  --eff_min 0.2 \
  --base_max 0.35 \
  --topk 50 \
  --out_json /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_answer/headscan_slake_mm_accel_hulumed4b_96g/selected_heads_stable_hulumed4b.json \
  --out_csv /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_answer/headscan_slake_mm_accel_hulumed4b_96g/selected_heads_stable_hulumed4b.csv
