python pubmedqa_conflict_local.py \
  --data ori_pqal.json \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --select_k 400 \
  --max_scan 200000 \
  --max_new_token 32
