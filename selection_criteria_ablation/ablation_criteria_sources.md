# Ablation Criteria Table Sources

All rows use validation split results at `before_question`.

Delta definition:

- `\left|\Delta\mathrm{CFR}\right| = \left|\mathrm{CFR}_{method} - \mathrm{CFR}_{baseline}\right|`
- Values are reported in percentage points.
- Larger values mean a larger change from baseline, regardless of direction.

Metric extraction:

- ConflictMedQA (`Qwen3-4B`, `Llama-3.2-3B`):
  - `CFR = follow_conflict_rate_after` for new `*_val_summary.json` outputs.
  - For the older Qwen Dual file, `CFR = selected_ablation.metrics.attack_success_rate`.
  - `C2W = count(conflict_before.pred == gold and conflict_after.pred != gold) / N`.
  - For the older Qwen Dual file, `conflict_before.pred` is read from `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_all_positions/conflict_positions_val.jsonl`, and `conflict_after.pred` is read from `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/ablate_heads_val.jsonl`.
- VQA_RAD (`Hulu-med-4B`, `InternVL3.5-4B`):
  - `CFR = 1 - pred_ctx.ab_acc`.
  - `C2W = count(base_ctx_pred == "gold" and ab_ctx_pred == "wrong") / N`.
  - This is the same context-condition transition ratio used by the main VQA table.

Result files:

- Qwen3-4B CER-only: `/root/logit_lens/selection_criteria_ablation/results/qwen3-4b/before_question/cer-only/cer-only_val_summary.json`
- Qwen3-4B BCP-only: `/root/logit_lens/selection_criteria_ablation/results/qwen3-4b/before_question/bcp-only/bcp-only_val_summary.json`
- Qwen3-4B Dual: `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/standard_metrics_val.json`
- Llama-3.2-3B CER-only: `/root/logit_lens/selection_criteria_ablation/results/llama32-3b/before_question/cer-only/cer-only_val_summary.json`
- Llama-3.2-3B BCP-only: `/root/logit_lens/selection_criteria_ablation/results/llama32-3b/before_question/bcp-only/bcp-only_val_summary.json`
- Llama-3.2-3B Dual: `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/conflict_retest_ablated_heads_inf_before_question_val_summary.json`
- Hulu-med-4B CER-only: `/root/logit_lens/selection_criteria_ablation/results/hulumed4b/before_question/cer-only/cer-only_val.json`
- Hulu-med-4B BCP-only: `/root/logit_lens/selection_criteria_ablation/results/hulumed4b/before_question/bcp-only/bcp-only_val.json`
- Hulu-med-4B Dual: `/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/ablate_selected_heads_all_hulumed4b_val.json`
- InternVL3.5-4B CER-only: `/root/logit_lens/selection_criteria_ablation/results/internvl35-4b/before_question/cer-only/cer-only_val.json`
- InternVL3.5-4B BCP-only: `/root/logit_lens/selection_criteria_ablation/results/internvl35-4b/before_question/bcp-only/bcp-only_val.json`
- InternVL3.5-4B Dual: `/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_val.json`

Baseline sources for `\Delta\mathrm{CFR}`:

- Qwen3-4B: `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/standard_metrics_val.json`
- Llama-3.2-3B: baseline-before value from `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/before_question/conflict_retest_ablated_heads_inf_before_question_val_summary.json`
- Hulu-med-4B: `pred_ctx.base_acc` from `/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/ablate_selected_heads_all_hulumed4b_val.json`
- InternVL3.5-4B: `pred_ctx.base_acc` from `/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/ablate/ablate_selected_heads_all_val.json`

Validation data:

- Qwen3-4B: latest rerun outputs under `/root/logit_lens/selection_criteria_ablation/results/qwen3-4b/before_question/`; transition baseline for Dual is `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_all_positions/conflict_positions_val.jsonl`
- Llama-3.2-3B: `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/data/kept_pairs_a12_b10_all_val.jsonl`
- Hulu-med-4B: `/root/logit_lens/VQA_RAD/Hulu-med/data/nc_cc_both_correct_rerun_tmp_val.csv`
- InternVL3.5-4B: `/root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv`

Note: Qwen3-4B CER-only and BCP-only were updated from the latest rerun in `/root/logit_lens/selection_criteria_ablation/results/qwen3-4b/before_question/` and now contain 174 samples, matching the existing Qwen3-4B Dual validation result size.
For VQA-RAD, `CFR = 1 - pred_ctx.ab_acc` and `C2W = count(base_ctx_pred == "gold" and ab_ctx_pred == "wrong") / N` to match the main benchmark table.
