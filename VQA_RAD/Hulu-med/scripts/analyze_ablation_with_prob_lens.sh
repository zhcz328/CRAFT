python analyze_ablation_flip_with_probe_and_lens.py \
  --ablation_json result_train_hulumed4b_prefix/ablate_selected_heads_ctx_only_hulumed4b_val.json \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --probe_dir probe/results/conflict_linear_prefix_ok \
  --lens_ckpt tuned_lens/results/train_prefix_idreg_tmp_ok/tuned_lens.pt \
  --out_dir analyze_probe_lens/ablation_flip_prefix_conflict_ok_tmp \
  --score_type logit \
  --batch_size 1 \
  --selected_heads /root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_prefix/headscan_vqarad_mm_hulumed4b_prefix/selected_heads_stable_hulumed4b.json
