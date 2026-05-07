# Hulu-Med computational cost benchmark

## Setup

- Model: /root/autodl-tmp/InternVL3_5-4B
- Data CSV: /root/logit_lens/VQA_RAD/text_conflict/data/internvl35_4b/vqa_rad_nc_cc_both_correct_train.csv
- Selected heads: /root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/result_before_question_vqarad/selected_heads_merged_unique_layers.json
- Sample row id: 0
- Question: are regions of the brain infarcted
- Gold / wrong: yes / no
- Image path: /root/logit_lens/VQA_RAD/text_conflict/data/VQA_RAD_Image_Folder/synpic54610.jpg
- Image size after resize: [566, 555]
- Warmup / repeats: 2 / 5

## Timing

- Normal ctx mean latency: 43.271 ms
- Ablated ctx mean latency: 49.433 ms
- Latency delta: 6.162 ms (14.24%)
- Normal peak memory mean: 9249.616 MB
- Ablated peak memory mean: 9249.976 MB

## FLOPs

- Normal reported FLOPs: 3629.544068 GFLOPs
- Ablated reported FLOPs: 3629.544068 GFLOPs
- FLOPs delta: 0.000000 GFLOPs
- Note: this ablation is implemented with attention-mask hooks, so theoretical matmul FLOPs are expected to stay essentially unchanged; any runtime gap mainly reflects hook and mask overhead.

## Outputs

- Normal pred: gold
- Ablated pred: gold
- Normal follow_conflict: -8.000000
- Ablated follow_conflict: -15.625000
