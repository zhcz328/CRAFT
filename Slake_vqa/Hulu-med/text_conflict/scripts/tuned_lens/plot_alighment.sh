python tuned_lens/plot_alignment_figure.py \
  --manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --lens_ckpt /root/autodl-tmp/tuned_lens/hulumed_4b/results/train_before_answer_idreg/tuned_lens.pt \
  --out_dir /root/autodl-tmp/tuned_lens/hulumed_4b/results/train_before_answer_idreg/alignment_before_answer_idreg \
  --prompt_types base,support,conflict