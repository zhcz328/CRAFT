# Tuned Lens Preference Trajectory Data Sources

All curves are tuned-lens curves.

Plotted value:

`Delta_l = log p(gold) - log p(wrong)`

For grouped trajectory files, saved `tuned_delta_by_layer` is `log p(wrong) - log p(gold)`, so the plotted value is `-tuned_delta_by_layer`.

Curves:
- Red: follow-conflict samples, grouped by `follow_label == follow_conflict`, using `tuned_delta_by_layer`
- Green: resist samples, grouped by `follow_label == resist`, using `tuned_delta_by_layer`
- Blue dashed: follow-conflict samples after ablation, using `after_ablation.tuned_lens_mean_gold_logprob_by_layer - after_ablation.tuned_lens_mean_wrong_logprob_by_layer`

Shaded regions for red/green are one standard error from per-record tuned-lens trajectories when `trajectories.jsonl` is available. The ablation summaries only store mean curves, so the blue dashed line has no error band.

| Model | Curve | Samples | Source |
|---|---|---:|---|
| Qwen3-4B | Follow-conflict samples | 35 | `/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/before_question/results/trajectory_before_question/trajectories.jsonl` |
| Qwen3-4B | Resist samples | 3 | `/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/before_question/results/trajectory_before_question/trajectories.jsonl` |
| InternVL3.5-4B | Follow-conflict samples | 52 | `/root/autodl-tmp/tuned_lens/internvl35_4b/results/before_question/trajectory_train_idreg/trajectories.jsonl` |
| InternVL3.5-4B | Resist samples | 564 | `/root/autodl-tmp/tuned_lens/internvl35_4b/results/before_question/trajectory_train_idreg/trajectories.jsonl` |
| InternVL3.5-4B | After ablation | 39 | `/root/logit_lens/Slake_vqa/text_conflict/internvl35_4b/result_before_question_slake/ablation_flip_ablate_ctx_only_val_follow_conflict/summary.json` |
| Hulu-med-4B | Follow-conflict samples | 112 | `/root/autodl-tmp/tuned_lens/hulumed_4b/results/trajectory_before_question_idreg/trajectories.jsonl` |
| Hulu-med-4B | Resist samples | 499 | `/root/autodl-tmp/tuned_lens/hulumed_4b/results/trajectory_before_question_idreg/trajectories.jsonl` |
| Hulu-med-4B | After ablation | 107 | `/root/logit_lens/VQA_RAD/Hulu-med/analyze_probe_lens/ablation_flip_before_question_follow/summary.json` |

Missing data:

| Model | Missing curve | Directory/Source | Reason |
|---|---|---|---|
| Qwen3-4B | After ablation | `/root/autodl-tmp/tuned_lens/conflictmedqa/qwen3-4b/before_question/results` | ablation tuned-lens summary not found in provided tuned_lens outputs |
| Llama3.2-3B | follow/resist tuned-lens trajectories | `/root/autodl-tmp/tuned_lens/conflictmedqa/llama3.2-3b/before_question` | trajectory summary/jsonl not found |