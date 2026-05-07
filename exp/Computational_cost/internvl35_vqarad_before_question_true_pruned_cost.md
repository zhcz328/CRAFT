# Hulu-Med true head-skip benchmark

## Setup

- Model: /root/autodl-tmp/InternVL3_5-4B
- Data CSV: /root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_train.csv
- Selected heads: /root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json
- Sample row id: 0
- Question: are regions of the brain infarcted
- Gold / wrong: yes / no
- Image path: /root/logit_lens/VQA_RAD/text_conflict/data/VQA_RAD_Image_Folder/synpic54610.jpg
- Image size after resize: [566, 555]
- Warmup / repeats: 1 / 3

## Timing

- Normal ctx mean latency: 43.908 ms
- True-pruned ctx mean latency: 47.031 ms
- Latency delta: 3.123 ms (7.11%)

## FLOPs

- Normal reported FLOPs: 3629.544068 GFLOPs
- True-pruned reported FLOPs: 3625.879091 GFLOPs
- FLOPs delta: -3.664977 GFLOPs (-0.1010%)

## Outputs

- Normal pred: gold
- True-pruned pred: gold
- Normal follow_conflict: -8.000000
- True-pruned follow_conflict: -11.375000
