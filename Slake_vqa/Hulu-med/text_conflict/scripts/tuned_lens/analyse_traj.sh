python tuned_lens/analyze_trajectories.py \
  --manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --lens_ckpt /root/autodl-tmp/tuned_lens/hulumed_4b/results/train_before_answer_idreg/tuned_lens.pt \
  --out_dir /root/autodl-tmp/tuned_lens/hulumed_4b/results/trajectory_before_answer_idreg \
  --prompt_types conflict \
  --plot_max_records 20