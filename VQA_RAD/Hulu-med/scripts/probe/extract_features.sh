python probe/extract_features.py \
  --manifest probe/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out /root/autodl-tmp/Hulumed/probe_vqa_rad/features_ok/val_before_answer.pt \
  --batch_size 2 \
  --image_size 672