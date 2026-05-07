python probe/score_probe_on_manifest.py \
  --manifest tuned_lens/data/val_before_question.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --probe_dir \
    probe/results/follow_linear_before_question_ok \
  --probe_label follow_conflict_probe \
  --prompt_types base \
  --group_field none \
  --group_name base_all \
  --score_type logit \
  --out_png probe/results/base_follow_probes_logit_ok_before_question.png \
  --out_json probe/results/base_follow_probes_logit_ok_before_question.json
