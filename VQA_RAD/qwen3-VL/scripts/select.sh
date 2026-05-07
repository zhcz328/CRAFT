python select_heads_merged_unique_layers.py \
  --input_json /home/zengjiaqi/icl/interp/logit_lens/cross_modal/result/headscan_vqarad_mm/head_scan_merged_unique_layers.json \
  --source results \
  --mode conflict_specific \
  --eff_min 0.28 \
  --base_max 0.30 \
  --topk 50 \
  --out_json selected_heads_stable.json \
  --out_csv selected_heads_stable.csv