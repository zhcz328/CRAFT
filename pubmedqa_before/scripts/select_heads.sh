python select_heads.py \
  --round_files /root/logit_lens/pubmedqa/result_train_prefix/headscan_pubmed/head_scan_round0_rise.json,/root/logit_lens/pubmedqa/result_train_prefix/headscan_pubmed/head_scan_round1_topk_fallback.json \
  --round_names round0_rise,round1_topk_fallback \
  --mode conflict_specific \
  --eff_min 2.5 \
  --base_max 1.5 \
  --combine union \
  --out_json /root/logit_lens/pubmedqa/result_train_prefix/selected_heads/selected_conflict_heads.json \
  --out_csv /root/logit_lens/pubmedqa/result_train_prefix/selected_heads/selected_conflict_heads.csv
