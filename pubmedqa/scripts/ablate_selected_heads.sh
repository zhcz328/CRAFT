CUDA_VISIBLE_DEVICES=1
python ablate_selected_heads.py \
  --data /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json --fmt json \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --selected_heads /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/result/headscan_pubmed/selected_conflict_heads.json \
  --position before_answer --include_context \
  --mask_scope ic_only --keep_mode self \
  --nc_yn_tau 2.0 --nc_unk_tau 0.0 \
  --ic_tau 2.0 --ic_unk_tau 0.0 \
  --out_summary result/ablation_evalstyle_ic_only_summary_before_answer.json \
  --out_jsonl result/ablation_evalstyle_ic_only_preds_before_answer.jsonl
