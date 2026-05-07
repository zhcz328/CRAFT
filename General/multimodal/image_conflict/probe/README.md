`probe` directory layout:

- `prepare_data.py`: build `nc/ic` manifests from the filtered train/val CSV split and the image-conflict eval `preds.jsonl`
- `extract_features.py`: cache multimodal last-token hidden states for every layer
- `train_probe.py`: train one probe per layer for `conflict` or `follow_conflict`
- `image_eval.py`: evaluate how image perturbations change `follow_context`, `follow_conflict`, and `unknown` tendency

In this project:

- `conflict` means distinguishing `nc` vs `ic`
- `follow_conflict` keeps the old task name for compatibility, but now means the binary hallucination task: `unknown` = no hallucination, non-`unknown` = hallucination

Suggested workflow:

```bash
python probe/prepare_data.py --model hulumed-4b --position image_conflict --out_dir probe/data

python probe/extract_features.py \
  --manifest hulumed4b/probe/data/train_image_conflict.jsonl \
  --model hulumed-4b \
  --out probe/features/train_image_conflict.pt

python probe/extract_features.py \
  --manifest hulumed4b/probe/data/val_image_conflict.jsonl \
  --model hulumed-4b \
  --out probe/features/val_image_conflict.pt

python probe/train_probe.py \
  --train_features hulumed4b/probe/features/train_image_conflict.pt \
  --val_features hulumed4b/probe/features/val_image_conflict.pt \
  --task conflict \
  --probe_type linear \
  --out_dir hulumed4b/probe/results/conflict_linear_image_conflict

python probe/train_probe.py \
  --train_features hulumed4b/probe/features/train_image_conflict.pt \
  --val_features hulumed4b/probe/features/val_image_conflict.pt \
  --task follow_conflict \
  --probe_type linear \
  --out_dir hulumed4b/probe/results/follow_linear_image_conflict

python probe/image_eval.py \
  --data_csv data/hulumed4b/slake_nc_correct_ic_ready_val.csv \
  --image_root . \
  --model hulumed-4b \
  --ablation object_mask \
  --sample_filter base_gold \
  --mask_scale 2.0 \
  --out_jsonl probe/image_eval/val_object_mask.jsonl \
  --out_summary probe/image_eval/val_object_mask_summary.json
```

Notes:

- The default split files are `data/<model_slug>/slake_nc_correct_ic_ready_train.csv` and `data/<model_slug>/slake_nc_correct_ic_ready_val.csv`.
- `--model` accepts the four workflow aliases as well as direct local/HF model paths.
- If `--model` is a local path, also pass `--model_name` such as `hulumed-4b` or `InternVL3_5-4B`.
- Outcome labels are matched by `img_id + question`, not by raw `id`, because the saved eval outputs do not always use the same integer ids as the split CSVs.
- If your image-conflict `preds.jsonl` lives somewhere else, pass it explicitly with `--preds_jsonl`.
- `extract_features.py` builds the masked `ic` image on the fly from `image_path + detection_path + mask_path`, so you do not need to materialize a second image dataset in advance.
