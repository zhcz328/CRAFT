#!/usr/bin/env bash
set -euo pipefail

cd /root/logit_lens/conflictmedqa/Qwen3-4B_exp
mkdir -p /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/logs
/root/miniconda3/bin/python ablate_head_inf.py --pairs /root/logit_lens/conflictmedqa/Qwen3-4B_exp/data/kept_pairs_a12_b10_all_val.jsonl --model /root/autodl-tmp/qwen3-4B --head_file /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/selected_heads/cer-only/cer-only_selected_heads.json --out /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/cer-only/cer-only_val.jsonl --summary_out /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/cer-only/cer-only_val_summary.json --positions before_question --mask_scope all --dtype bfloat16 --device_map cuda:1 2>&1 | tee /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/logs/cer-only_val.log

cd /root/logit_lens/conflictmedqa/Qwen3-4B_exp
mkdir -p /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/logs
/root/miniconda3/bin/python ablate_head_inf.py --pairs /root/logit_lens/conflictmedqa/Qwen3-4B_exp/data/kept_pairs_a12_b10_all_val.jsonl --model /root/autodl-tmp/qwen3-4B --head_file /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/selected_heads/bcp-only/bcp-only_selected_heads.json --out /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/bcp-only/bcp-only_val.jsonl --summary_out /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/bcp-only/bcp-only_val_summary.json --positions before_question --mask_scope all --dtype bfloat16 --device_map cuda:1 2>&1 | tee /root/logit_lens/selection_criteria_ablation/results_path_check/qwen3-4b/before_question/logs/bcp-only_val.log

