# Fig. 9 Data Sources

Plotted preference margin:

`Delta_l = log p(gold) - log p(wrong)`

Curves:
- Red: follow-conflict samples from `tuned_delta_by_layer` grouped by `follow_label == follow_conflict`
- Green: resist samples from `tuned_delta_by_layer` grouped by `follow_label == resist`
- Blue dashed: after-ablation curve when an ablation summary exists, otherwise the tuned-lens overall curve as a fallback

Shaded regions show one standard error when per-record trajectories are available.

| Model | Curve | Samples | Source |
|---|---|---:|---|
| Qwen3-4B | follow_conflict | 35 | `trajectories.jsonl` |
| Qwen3-4B | resist | 3 | `trajectories.jsonl` |
| Qwen3-4B | after_ablation | 12 | `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/analyze_probe_lens/ablation_flip_before_question_follow/summary.json` |
| InternVL3.5-4B | follow_conflict | 52 | `trajectories.jsonl` |
| InternVL3.5-4B | resist | 564 | `trajectories.jsonl` |
| InternVL3.5-4B | after_ablation | 39 | `/root/logit_lens/Slake_vqa/text_conflict/internvl35_4b/result_before_question_slake/ablation_flip_ablate_ctx_only_val_follow_conflict/summary.json` |
| Hulu-med-4B | follow_conflict | 112 | `trajectories.jsonl` |
| Hulu-med-4B | resist | 499 | `trajectories.jsonl` |
| Hulu-med-4B | after_ablation | 111 | `/root/logit_lens/Slake_vqa/Hulu-med/text_conflict/analyze_probe_lens/ablation_flip_before_question_follow/summary.json` |
| Llama3.2-3B | follow_conflict | 27 | `trajectories.jsonl` |
| Llama3.2-3B | resist | 7 | `trajectories.jsonl` |
| Llama3.2-3B | after_ablation | 22 | `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/analysis/conflict_retest_ablated_heads_inf_before_question_val/follow_conflict/summary.json` |