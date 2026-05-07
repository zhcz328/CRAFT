`tuned_lens` directory layout:

- `prepare_data.py`: build multimodal prompt manifests from the train/val CSV split and a position-specific `preds.jsonl`
- `train_tuned_lens.py`: train one translator per layer on next-token distributions
- `analyze_trajectories.py`: compare raw-vs-tuned wrong-minus-gold first-token trajectories

Suggested workflow:

```bash
python tuned_lens/prepare_data.py --model hulumed-4b --position prefix --out_dir tuned_lens/data

python tuned_lens/train_tuned_lens.py \
  --train_manifest hulumed4b/tuned_lens/data/train_prefix.jsonl \
  --val_manifest hulumed4b/tuned_lens/data/val_prefix.jsonl \
  --model hulumed-4b \
  --out_dir tuned_lens/results/train_prefix

python tuned_lens/analyze_trajectories.py \
  --manifest hulumed4b/tuned_lens/data/val_prefix.jsonl \
  --model hulumed-4b \
  --lens_ckpt hulumed4b/tuned_lens/results/train_prefix/tuned_lens.pt \
  --prompt_types conflict \
  --out_dir tuned_lens/results/trajectory_prefix
```

Notes:

- `--model` accepts the four workflow aliases as well as direct local/HF model paths.
- If `--model` is a local path, also pass `--model_name` such as `qwen3vl-4B` or `Qwen3.5-4B`.
- Like the probe pipeline, manifests are matched back to eval outputs by `img_id + question`.
- The trajectory analyzer only keeps samples whose `gold_answer` and `wrong_answer` are single tokens under the model tokenizer. This matches the original conflictmedqa analysis style and avoids misleading multi-token deltas.
- If your full `before_answer` or `before_question` `preds.jsonl` lives elsewhere, pass it explicitly to `prepare_data.py` with `--preds_jsonl`.
