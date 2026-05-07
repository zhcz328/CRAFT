CUDA_VISIBLE_DEVICES=1 python eval_nc_ic_cc_conflict.py \
  --data /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/data_results_pubmedqa_filter_nc_cc_qwen3_4B/_archive_zengjiaqi_Medical_LLM_Qwen3_model_Qwen_Qwen3-4B.filtered.json \
  --fmt json \
  --models /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --positions prefix before_question before_answer \
  --nc_yn_tau 2.0 --nc_unk_tau 0.0 \
  --cc_tau 3.0 --cc_unk_tau 0.0 --ic_tau 2.0 --ic_unk_tau 0.0 \
  --dtype bf16 --device_map auto \
  --out_dir result/qwen3-4b_follow_conflict