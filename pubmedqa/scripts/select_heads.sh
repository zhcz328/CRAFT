python select_heads.py \
  --round_files /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/head_scan_round0_rise.json,/home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/head_scan_round1_preplateau.json,/home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/head_scan_round2_tail.json \
  --round_names round0_rise,round1_preplateau,round2_tail \
  --mode conflict_specific \
  --eff_min 3.5 \
  --base_max 1.5 \
  --combine union \
  --out_json /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/selected_conflict_heads.json \
  --out_csv /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/selected_conflict_heads.csv
