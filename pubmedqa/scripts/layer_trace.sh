python layer_trace_on_pubmed.py \
  --data /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json \
  --fmt json \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --position before_answer \
  --patch_all_tokens \
  --limit 202 \
  --metrics abstain_adv,follow_conflict,yesno_margin \
  --scan_plan_out result/layer_trace/scan_plan.json \
  --scan_plan_metrics abstain_adv,follow_conflict \
  --scan_plan_weights 0.35,0.65 \
  --out result/layer_trace/trace.json \
  --plot_prefix result/layer_trace/trace \
  --include_context
 
