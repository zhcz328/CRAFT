python probe/extract_features.py \
  --manifest probe/data/train_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out /root/autodl-tmp/Hulumed/probe/features/train_before_answer.pt \
  --batch_size 2 \
  --image_size 672