python tuned_lens/analyze_trajectories.py \
  --manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --lens_ckpt ./tuned_lens/results/train_before_answer_idreg_tmp_ok/tuned_lens.pt \
  --out_dir tuned_lens/results/trajectory_before_answer_idreg_tmp_ok \
  --prompt_types conflict \
  --plot_max_records 20