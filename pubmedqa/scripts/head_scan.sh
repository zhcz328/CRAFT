export CUDA_VISIBLE_DEVICES=1
python head_scan_on_pubmed.py \
  --data /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json \
  --fmt json \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --position before_question \
  --plan /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/layer_trace/scan_plan.json \
  --plan_out_dir result/headscan_pubmed \
  --max_examples 202 \
  --metrics abstain_adv,follow_conflict,yesno_margin \
  --weights 0.35,0.65,0.0 \
  --keep_mode self \
  --include_context
