export CUDA_VISIBLE_DEVICES=0
python layer_trace_on_pubmed.py \
  --data /root/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.train.json \
  --fmt json \
  --model /root/autodl-tmp/qwen3-4B \
  --position prefix \
  --patch_k 8 \
  --limit 0 \
  --metrics abstain_adv,follow_conflict,yesno_margin \
  --scan_plan_out result_train_prefix/layer_trace/scan_plan.json \
  --scan_plan_metrics abstain_adv,follow_conflict \
  --scan_plan_weights 0.05,0.95 \
  --out result_train_prefix/layer_trace/trace.json \
  --plot_prefix result_train_prefix/layer_trace/trace \
  --include_context
 
