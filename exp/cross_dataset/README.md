## Cross-dataset ablation

Text:

- `bash /root/logit_lens/exp/cross_dataset/text/run_slake_heads_on_vqarad.sh`
- `bash /root/logit_lens/exp/cross_dataset/text/run_vqarad_heads_on_slake.sh`
- `bash /root/logit_lens/exp/cross_dataset/text/run_all_cross_text.sh`

Image:

- `bash /root/logit_lens/exp/cross_dataset/image/run_heal_heads_on_slake.sh`
- `bash /root/logit_lens/exp/cross_dataset/image/run_slake_heads_on_heal.sh`
- `bash /root/logit_lens/exp/cross_dataset/image/run_all_cross_image.sh`

Defaults:

- Text uses `before_question`.
- Image uses the existing `image_conflict` position.
- Outputs are written under `/root/logit_lens/exp/cross_dataset/{text,image}/results/...`.
- You can override `MASK_SCOPE`, `SPLITS`, `MODEL_PATH`, `DEVICE`, `MAX_EXAMPLES`, and `OUT_DIR` via environment variables.
