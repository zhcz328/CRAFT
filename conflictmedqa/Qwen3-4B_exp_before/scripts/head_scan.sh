export CUDA_VISIBLE_DEVICES=1
python head_scan_inf.py \
  --pairs data/kept_pairs_a12_b10_all_train.jsonl \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --plan result_train/before_question/layer_trace_rise/scan_plan.json \
  --plan_out_dir result_train/before_question/headscan_rounds_top50_inf \
  --position before_question \
  --max_pairs 106 \
  --cs_eff_min 3.0 \
  --cs_base_max 2.5 \
  --bb_eff_min 4.0 \
  --bb_base_min 4.5
  # --cs_eff_min 1.2 \
  # --cs_base_max 0.7 \
  # --bb_base_min 1.5 \
  # --bb_eff_min 0.1 \
