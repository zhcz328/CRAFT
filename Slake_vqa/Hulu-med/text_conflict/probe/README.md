`probe` directory layout:

- `prepare_data.py`: build Hulu-med prompt manifests from the existing train/val CSV split and a position-specific `preds.jsonl`
- `extract_features.py`: cache multimodal last-token hidden states for every layer
- `train_probe.py`: train one probe per layer for `conflict` or `follow_conflict`

Suggested workflow:

```bash
python probe/prepare_data.py --position prefix

python probe/extract_features.py \
  --manifest probe/data/train_prefix.jsonl \
  --model /path/to/Hulu-Med-4B \
  --out probe/features/train_prefix.pt

python probe/extract_features.py \
  --manifest probe/data/val_prefix.jsonl \
  --model /path/to/Hulu-Med-4B \
  --out probe/features/val_prefix.pt

python probe/train_probe.py \
  --train_features probe/features/train_prefix.pt \
  --val_features probe/features/val_prefix.pt \
  --task conflict \
  --probe_type linear \
  --out_dir probe/results/conflict_linear_prefix

python probe/train_probe.py \
  --train_features probe/features/train_prefix.pt \
  --val_features probe/features/val_prefix.pt \
  --task follow_conflict \
  --probe_type linear \
  --out_dir probe/results/follow_linear_prefix
```

Notes:

- The default split files are `data/slake_nc_cc_both_correct_train.csv` and `data/slake_nc_cc_both_correct_val.csv`.
- Outcome labels are matched by `img_id + question`, not by raw `id`, because the saved eval outputs do not always use the same integer ids as the split CSVs.
- If your `before_answer` or `before_question` `preds.jsonl` lives somewhere else, pass it explicitly with `--preds_jsonl`.
