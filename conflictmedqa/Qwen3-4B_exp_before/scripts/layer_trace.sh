python layer_trace.py \
  --pairs data/kept_pairs_a12_b10_all_train.jsonl \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
  --position before_question \
  --out result_train/before_question/layer_trace_rise/layer_trace.json \
  --plot_out result_train/before_question/layer_trace_rise/layer_trace_scores.png \
  --plan_out result_train/before_question/layer_trace_rise/scan_plan.json
