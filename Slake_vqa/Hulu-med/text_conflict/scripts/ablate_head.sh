python ablate_head.py \
  --data_csv ./data/slake_nc_cc_both_correct_train.csv \
  --image_root /root/autodl-tmp/data/SLAKE/imgs \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --selected_heads /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_answer/headscan_slake_mm_accel_hulumed4b_96g/selected_heads_stable_hulumed4b.json \
  --trace_mode conflict \
  --position before_answer \
  --metrics follow_conflict \
  --mask_scope all \
  --keep_mode self \
  --dtype bf16 \
  --out_json /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_answer/ablate_selected_heads_all_hulumed4b_train.json \
  
