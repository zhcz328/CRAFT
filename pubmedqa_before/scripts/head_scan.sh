export CUDA_VISIBLE_DEVICES=0
python head_scan_on_pubmed.py \
  --data data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.train.json \
  --fmt json \
  --model /root/autodl-tmp/qwen3-4B \
  --position prefix \
  --plan result_train_prefix/layer_trace/scan_plan.json \
  --plan_out_dir result_train_prefix/headscan_pubmed \
  --max_examples 121 \
  --metrics abstain_adv,follow_conflict,yesno_margin \
  --weights 0.05,0.95,0.0 \
  --keep_mode self \
  --include_context
