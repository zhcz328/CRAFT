python filter_data.py \
  --data ./ori_pqal.json --fmt json \
  --models \
    /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --preset balanced \
  --out_dir data_results_pubmedqa_filter_nc_cc_qwen3_4B \
  --dtype bf16 --device_map auto
