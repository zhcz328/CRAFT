`probe` directory layout:

- `prepare_data.py`: build prompt-level manifests from the existing kept-pairs files
- `extract_features.py`: cache last-token hidden states for every layer
- `train_probe.py`: train one probe per layer for `conflict` or `follow_conflict`

Suggested workflow:

```bash
python probe/prepare_data.py --position before_question

python probe/extract_features.py \
  --manifest probe/data/train_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --out probe/features/train_before_question.pt

python probe/extract_features.py \
  --manifest probe/data/val_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --out probe/features/val_before_question.pt

python probe/train_probe.py \
  --train_features probe/features/train_before_question.pt \
  --val_features probe/features/val_before_question.pt \
  --task conflict \
  --probe_type linear \
  --out_dir /root/autodl-tmp/probe/conflictmedqa/qwen3-4b/before_question/conflict_linear

python probe/train_probe.py \
  --train_features probe/features/train_before_question.pt \
  --val_features probe/features/val_before_question.pt \
  --task follow_conflict \
  --probe_type linear \
  --out_dir /root/autodl-tmp/probe/conflictmedqa/qwen3-4b/before_question/follow_linear
```

For `follow_conflict` probing, prefer the pair-stratified manifests:

```bash
python probe/extract_features.py \
  --manifest probe/data/train_pair_stratified_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --out probe/features/train_pair_stratified_before_question.pt

python probe/extract_features.py \
  --manifest probe/data/val_pair_stratified_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --out probe/features/val_pair_stratified_before_question.pt
```
