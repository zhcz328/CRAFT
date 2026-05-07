#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/logit_lens"
EXP_DIR="${ROOT_DIR}/exp/Computational_cost"
TASK_DIR="${ROOT_DIR}/VQA_RAD/text_conflict"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/InternVL3_5-4B}"
ABLATE_HEAD_PATH="${ABLATE_HEAD_PATH:-${TASK_DIR}/ablate_head.py}"
DATA_CSV="${DATA_CSV:-${TASK_DIR}/data/internvl35_4b/vqa_rad_nc_cc_both_correct_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-${TASK_DIR}}"
SELECTED_HEADS="${SELECTED_HEADS:-${TASK_DIR}/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json}"
POSITION="${POSITION:-before_question}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda}"
WARMUP_ABLATE="${WARMUP_ABLATE:-2}"
REPEATS_ABLATE="${REPEATS_ABLATE:-5}"
WARMUP_PRUNED="${WARMUP_PRUNED:-1}"
REPEATS_PRUNED="${REPEATS_PRUNED:-3}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-672}"

cd "${ROOT_DIR}"

python "${EXP_DIR}/benchmark_hulumed_ablation_cost.py" \
  --ablate_head_path "${ABLATE_HEAD_PATH}" \
  --model "${MODEL_PATH}" \
  --data_csv "${DATA_CSV}" \
  --image_root "${IMAGE_ROOT}" \
  --selected_heads "${SELECTED_HEADS}" \
  --position "${POSITION}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --sample_index "${SAMPLE_INDEX}" \
  --max_image_side "${MAX_IMAGE_SIDE}" \
  --warmup "${WARMUP_ABLATE}" \
  --repeats "${REPEATS_ABLATE}" \
  --out_dir "${EXP_DIR}" \
  --output_stem "internvl35_vqarad_before_question_ablation_cost"

python "${EXP_DIR}/benchmark_hulumed_pruned_head_cost.py" \
  --ablate_head_path "${ABLATE_HEAD_PATH}" \
  --model "${MODEL_PATH}" \
  --data_csv "${DATA_CSV}" \
  --image_root "${IMAGE_ROOT}" \
  --selected_heads "${SELECTED_HEADS}" \
  --position "${POSITION}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --sample_index "${SAMPLE_INDEX}" \
  --max_image_side "${MAX_IMAGE_SIDE}" \
  --warmup "${WARMUP_PRUNED}" \
  --repeats "${REPEATS_PRUNED}" \
  --out_dir "${EXP_DIR}" \
  --output_stem "internvl35_vqarad_before_question_true_pruned_cost"
