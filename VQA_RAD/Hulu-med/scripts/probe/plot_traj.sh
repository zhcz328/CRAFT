python probe/plot_sample_trajectory.py \
  --features probe/features/val_prefix.pt \
  --probe_dir probe/results/follow_linear_before_question_ok \
  --task follow_conflict \
  --plot_mode mean \
  --score_type both \
  --out_png probe/results/follow_linear_prefix_ok_before_question/mean_trajectory.png \
  --out_json probe/results/follow_linear_prefix_ok_before_question/mean_trajectory.json
