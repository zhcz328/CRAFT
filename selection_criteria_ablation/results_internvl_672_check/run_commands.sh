#!/usr/bin/env bash
set -euo pipefail

cd /root/logit_lens/VQA_RAD/text_conflict
mkdir -p /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/logs
/root/miniconda3/bin/python ablate_head.py --data_csv /root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv --image_root . --model /root/autodl-tmp/InternVL3_5-4B --selected_heads /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/selected_heads/cer-only/cer-only_selected_heads.json --out_json /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/cer-only/cer-only_val.json --device cuda:0 --dtype bf16 --trace_mode conflict --position before_question --metrics follow_conflict --mask_scope all --keep_mode self --seed 0 --max_image_side 672 --model_name InternVL3_5-4B 2>&1 | tee /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/logs/cer-only_val.log

cd /root/logit_lens/VQA_RAD/text_conflict
mkdir -p /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/logs
/root/miniconda3/bin/python ablate_head.py --data_csv /root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_val.csv --image_root . --model /root/autodl-tmp/InternVL3_5-4B --selected_heads /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/selected_heads/bcp-only/bcp-only_selected_heads.json --out_json /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/bcp-only/bcp-only_val.json --device cuda:0 --dtype bf16 --trace_mode conflict --position before_question --metrics follow_conflict --mask_scope all --keep_mode self --seed 0 --max_image_side 672 --model_name InternVL3_5-4B 2>&1 | tee /root/logit_lens/selection_criteria_ablation/results_internvl_672_check/internvl35-4b/before_question/logs/bcp-only_val.log

