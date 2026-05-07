python tuned_lens/train_tuned_lens.py \
  --train_manifest tuned_lens/data/train_before_answer.jsonl \
  --val_manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out_dir /root/autodl-tmp/tuned_lens/hulumed_4b/results/train_before_answer_idreg \
  --prompt_types base,support,conflict \
  --lr 4e-5 \
  --identity_reg_weight 3e-3