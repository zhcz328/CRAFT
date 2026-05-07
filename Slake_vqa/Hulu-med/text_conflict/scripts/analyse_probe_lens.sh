python analyze_ablation_flip_with_probe_and_lens.py \
  --ablation_json /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_question/ablate_selected_heads_ctx_only_hulumed4b_val.json \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --probe_dir /root/autodl-tmp/Hulumed/probe_slake_vqa/results/conflict_linear_before_question \
  --lens_ckpt /root/autodl-tmp/tuned_lens/hulumed_4b/results/train_before_question_idreg/tuned_lens.pt \
  --out_dir analyze_probe_lens/ablation_flip_before_question_conflict \
  --score_type logit \
  --batch_size 1 \
  --selected_heads /root/logit_lens/Slake_vqa/Hulu-med/text_conflict/result_slake_hulumed4b_before_question/headscan_slake_mm_accel_hulumed4b_96g/selected_heads_stable_hulumed4b.json
