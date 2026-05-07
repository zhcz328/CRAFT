python ablate_head_inf_random.py \
  --pairs /home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/data/kept_pairs_a12_b10_all_val.jsonl --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B --head_groups /home/zengjiaqi/icl/interp/logit_lens/conflictmedqa/Qwen3-4B_exp/result/headscan_rounds_top30_inf/head_groups.json \
  --ablate_set random --random_seed 42 --random_match total \
  --positions before_question \
  --out result_train/before_question/ablate_random \
  --mask_scope conflict_only \
