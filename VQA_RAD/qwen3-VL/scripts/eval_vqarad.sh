python eval_vqarad_mm_support_conflict_use_wrong_csv.py \
  --data_csv ./data/nc_cc_both_correct_rerun.csv \
  --image_root . \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3-VL-8B-Instruct \
  --out_dir eval-results_vqarad_qwen3vl_all \
  --position before_question \
  --dtype auto \
  --device_map cuda:1 \
  --limit 0