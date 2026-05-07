python conflictmedqa_eval_local.py \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --config gpt4o \
  --max_new_tokens 32\
  --temperature 0 \
  --local_file /home/zengjiaqi/Medical_rag/simple_evaluation/demo2/data/ConflictMed_v2.jsonl \
  --limit 0 \
  --mode logit \
  --out confilctmedqa_pred_logit_all.jsonl
  #--enable_thinking \
