`tuned_lens` directory layout:

- `prepare_data.py`: build multimodal `nc/ic` manifests from the train/val CSV split and the image-conflict `preds.jsonl`
- `train_tuned_lens.py`: train one translator per layer on next-token distributions
- `analyze_trajectories.py`: compare raw-vs-tuned `unknown` vs `best_wrong_candidate` first-token trajectories

Suggested workflow:

```bash
python tuned_lens/prepare_data.py --model hulumed-4b --position image_conflict --out_dir tuned_lens/data

python tuned_lens/train_tuned_lens.py \
  --train_manifest hulumed4b/tuned_lens/data/train_image_conflict.jsonl \
  --val_manifest hulumed4b/tuned_lens/data/val_image_conflict.jsonl \
  --model hulumed-4b \
  --out_dir tuned_lens/results/train_image_conflict

python tuned_lens/analyze_trajectories.py \
  --manifest hulumed4b/tuned_lens/data/val_image_conflict.jsonl \
  --model hulumed-4b \
  --lens_ckpt hulumed4b/tuned_lens/results/train_image_conflict/tuned_lens.pt \
  --prompt_types ic \
  --out_dir tuned_lens/results/trajectory_image_conflict
```

Notes:

- `--model` accepts the four workflow aliases as well as direct local/HF model paths.
- If `--model` is a local path, also pass `--model_name` such as `qwen3vl-4B` or `Qwen3.5-4B`.
- Like the probe pipeline, manifests are matched back to eval outputs by `img_id + question`.
- `wrong_answer` is fixed to `unknown`, because the masked-image intervention is intended to push the model toward `unknown`; in this project, failing to move to `unknown` is treated as hallucination under masking.
- The manifest now exposes `target_answer = unknown` and `competing_answers = wrong_candidates + gold_answer` for the masked-image setting.
- The trajectory analyzer compares `target_answer` against the highest-logit single-token answer drawn from each sample's `competing_answers` pool.
- If your image-conflict `preds.jsonl` lives elsewhere, pass it explicitly to `prepare_data.py` with `--preds_jsonl`.
