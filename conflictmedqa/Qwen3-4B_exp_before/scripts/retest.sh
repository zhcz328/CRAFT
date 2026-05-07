export CUBLAS_WORKSPACE_CONFIG=:4096:8
python retest_pairs.py \
  --pairs kept_pairs_a12_b12.jsonl \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --mode logit \
  --max_new_tokens 32 \
  --temperature 0.01 \
  --out retest_logit.jsonl
