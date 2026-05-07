python CHDP/build_dataset.py \
  --atomic_bundle /root/autodl-tmp/CHDP/VQA_RAD/hulumed4b/features_before_question/train_atomic.pt \
  --out /root/autodl-tmp/CHDP/VQA_RAD/hulumed4b/features_before_question/train_dataset.pt

python CHDP/build_dataset.py \
  --atomic_bundle /root/autodl-tmp/CHDP/VQA_RAD/hulumed4b/features_before_question/val_atomic.pt \
  --out /root/autodl-tmp/CHDP/VQA_RAD/hulumed4b/features_before_question/val_dataset.pt
