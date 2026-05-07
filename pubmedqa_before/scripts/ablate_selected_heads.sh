CUDA_VISIBLE_DEVICES=0,1
python ablate_selected_heads.py \
  --data /root/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json --fmt json \
  --model /root/autodl-tmp/qwen3-4B \
  --selected_heads /root/logit_lens/pubmedqa/result_train_prefix/selected_heads/selected_conflict_heads.json \
  --position prefix --include_context \
  --mask_scope ic_only --keep_mode self \
  --nc_yn_tau 2.0 --nc_unk_tau 0.0 \
  --ic_tau 2.0 --ic_unk_tau 0.0 \
  --out_summary /root/logit_lens/pubmedqa/result_train_prefix/ablation_evalstyle_ic_only_summary_prefix_all.json \
  --out_jsonl /root/logit_lens/pubmedqa/result_train_prefix/ablation_evalstyle_ic_only_preds_prefix_all.jsonl
