python ablate_head.py \
  --data_csv  /root/logit_lens/VQA_RAD/Hulu-med/data/nc_cc_both_correct_rerun_tmp_val.csv \
  --image_root . \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --selected_heads /root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_prefix/headscan_vqarad_mm_hulumed4b_prefix/selected_heads_stable_hulumed4b.json \
  --trace_mode conflict \
  --position prefix \
  --metrics follow_conflict \
  --mask_scope all \
  --keep_mode self \
  --dtype bf16 \
  --out_json /root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_prefix/ablate_selected_heads_all_hulumed4b_val_random.json \
  --random_ablate \
  --random_seed 42 \
  --device cuda:1
 
