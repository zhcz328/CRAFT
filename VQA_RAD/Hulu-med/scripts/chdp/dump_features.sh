python CHDP/dump_features.py \
  --manifest CHDP/data_before_question/train.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out CHDP/features_before_question/train_atomic.pt \
  --batch_size 1 \
  --image_size 672 \
  --dtype bfloat16 \
  --attn_implementation eager \
  --attention_mode compact \
  --save_dtype float16

python CHDP/dump_features.py \
  --manifest CHDP/data_before_question/val.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --out CHDP/features_before_question/val_atomic.pt \
  --batch_size 1 \
  --image_size 672 \
  --dtype bfloat16 \
  --attn_implementation eager \
  --attention_mode compact \
  --save_dtype float16
