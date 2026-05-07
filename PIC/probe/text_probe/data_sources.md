# Probe Ablation Panel Sources

Each panel in this directory uses:

- Main plot: Probe-A validation AUROC and Probe-B validation Macro-F1 from the saved `summary.json` files, plus their post-ablation curves recomputed after installing the model-specific ablation hooks.
- Legend: exported separately as its own figure for layout flexibility.

For HuluMed, the solid curves intentionally use the `probe_vqa_rad` artifacts so the original and post-ablation curves are aligned to the same VQA-RAD ablation setup.

| Model | Probe-A summary | Probe-B summary | Probe-B val manifest | Selected heads | Ablated Probe-B cache |
|---|---|---|---|---|---|
| HuluMed-4B | `/root/autodl-tmp/Hulumed/probe_vqa_rad/results/conflict_linear_before_question/summary.json` | `/root/autodl-tmp/Hulumed/probe_vqa_rad/results/follow_linear_before_question/summary.json` | `/root/logit_lens/VQA_RAD/Hulu-med/probe/data/val_before_question.jsonl` | `/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json` | `/root/logit_lens/PIC/probe/hulumed4b_probe_a_ablation_auroc.json ; /root/logit_lens/PIC/probe/hulumed4b_probe_b_ablation_macro_f1.json` |