python probe/train_probe.py \
  --train_features /root/autodl-tmp/Hulumed/probe/features/train_before_answer.pt \
  --val_features /root/autodl-tmp/Hulumed/probe/features/val_before_answer.pt \
  --task conflict \
  --probe_type linear \
  --out_dir /root/autodl-tmp/Hulumed/probe/results/conflict_linear_before_answer
