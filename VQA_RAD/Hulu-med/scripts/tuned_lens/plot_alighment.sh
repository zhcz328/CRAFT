python tuned_lens/plot_alignment_figure.py \
  --manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --lens_ckpt tuned_lens/results/train_before_answer_idreg_tmp_ok/tuned_lens.pt \
  --out_dir tuned_lens/results/alignment_before_answer_idreg_tmp_ok \
  --prompt_types base,support,conflict