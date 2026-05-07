python ablate_head_inf.py \
  --pairs /home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/data/kept_pairs_a12_b10_all_train.jsonl \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --head_groups /home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/headscan_rounds_top50_inf/head_groups.json \
  --positions before_question\
  --out result_train/before_question/ablate_heads_train.jsonl \
  --mask_scope conflict_only
