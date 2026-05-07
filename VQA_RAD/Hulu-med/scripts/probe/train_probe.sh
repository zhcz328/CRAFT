python probe/train_probe.py \
  --train_features /root/autodl-tmp/Hulumed/probe_vqa_rad/features_ok/train_before_question.pt \
  --val_features /root/autodl-tmp/Hulumed/probe_vqa_rad/features_ok/val_before_question.pt \
  --task conflict \
  --probe_type linear \
  --epoch 80 \
  --stage_epochs 3 10 80 \
  --out_dir /root/autodl-tmp/Hulumed/probe_vqa_rad/results/conflict_linear_before_question_ok
