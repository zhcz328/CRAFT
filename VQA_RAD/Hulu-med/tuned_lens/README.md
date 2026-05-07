`tuned_lens` directory layout:

- `prepare_data.py`: build base/support/conflict prompt manifests
- `train_tuned_lens.py`: train per-layer translators against the model's final next-token distribution
- `analyze_trajectories.py`: compare raw logit lens and tuned lens on gold-vs-wrong deltas
- `plot_alignment_figure.py`: draw a paper-style alignment figure for input embeddings, raw hidden states, and aligned vectors

Suggested workflow:

```bash
python tuned_lens/prepare_data.py --position before_question

python tuned_lens/train_tuned_lens.py \
  --train_manifest tuned_lens/data/train_before_question.jsonl \
  --val_manifest tuned_lens/data/val_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --out_dir tuned_lens/results/train_before_question

python tuned_lens/analyze_trajectories.py \
  --manifest tuned_lens/data/val_pair_stratified_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --lens_ckpt tuned_lens/results/train_before_question/tuned_lens.pt \
  --prompt_types conflict \
  --out_dir tuned_lens/results/trajectory_before_question

python tuned_lens/plot_alignment_figure.py \
  --manifest tuned_lens/data/val_before_question.jsonl \
  --model /path/to/Qwen3-4B \
  --comparison_target final_distribution \
  --alignment_mode tuned_lens \
  --lens_ckpt tuned_lens/results/train_before_question/tuned_lens.pt \
  --layer 20 \
  --prompt_types conflict \
  --out_dir tuned_lens/results/alignment_before_question
```

Memory note:

- Full affine translators are the default (`--translator_rank 0`)
- If full tuned lens is too heavy, set `--translator_rank 64` or `128` first for a lighter approximation

Plotting note:

- `plot_alignment_figure.py` now defaults to `--comparison_target final_distribution`, so it compares a chosen layer's raw distribution and translated distribution against the final-layer distribution
- `plot_alignment_figure.py` saves a PNG with a density panel and a 2D PCA scatter panel, plus a JSON summary; in `final_distribution` mode the key metrics are KL, JS, cosine, and L2 to the final distribution
- Set `--alignment_mode ridge_io` to visualize a paper-style fixed ridge realignment matrix instead of the tuned-lens translator
- Set `--comparison_target input_embedding` if you specifically want the earlier input-embedding-style figure
