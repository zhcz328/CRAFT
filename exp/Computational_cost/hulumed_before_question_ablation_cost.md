# Hulu-Med computational cost benchmark

## Setup

- Model: /root/autodl-tmp/Hulu-Med-4B
- Data CSV: /root/logit_lens/VQA_RAD/Hulu-med/data/nc_cc_both_correct_rerun_tmp_val.csv
- Selected heads: /root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json
- Sample row id: 988
- Question: is there any bleeding in this patient brain
- Gold / wrong: no / yes
- Image path: /root/logit_lens/VQA_RAD/Hulu-med/data/VQA_RAD_Image_Folder/synpic23631.jpg
- Image size after resize: [537, 672]
- Warmup / repeats: 1 / 3

## Timing

- Normal ctx mean latency: 206.387 ms
- Ablated ctx mean latency: 210.311 ms
- Latency delta: 3.924 ms (1.90%)
- Normal peak memory mean: 9398.789 MB
- Ablated peak memory mean: 9483.598 MB

## FLOPs

- Normal reported FLOPs: 19539.960512 GFLOPs
- Ablated reported FLOPs: 19539.960512 GFLOPs
- FLOPs delta: 0.000000 GFLOPs
- Note: this ablation is implemented with attention-mask hooks, so theoretical matmul FLOPs are expected to stay essentially unchanged; any runtime gap mainly reflects hook and mask overhead.

## Outputs

- Normal pred: gold
- Ablated pred: gold
- Normal follow_conflict: -0.312500
- Ablated follow_conflict: -1.437500
