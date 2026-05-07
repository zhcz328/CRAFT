# Hulu-Med true head-skip benchmark

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

- Normal ctx mean latency: 207.114 ms
- True-pruned ctx mean latency: 208.101 ms
- Latency delta: 0.987 ms (0.48%)

## FLOPs

- Normal reported FLOPs: 19539.960512 GFLOPs
- True-pruned reported FLOPs: 19517.964285 GFLOPs
- FLOPs delta: -21.996226 GFLOPs (-0.1126%)

## Outputs

- Normal pred: gold
- True-pruned pred: gold
- Normal follow_conflict: -0.312500
- True-pruned follow_conflict: -0.562500
