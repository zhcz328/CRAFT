python probe/score_probe_on_manifest.py \
  --manifest tuned_lens/data/val_before_answer.jsonl \
  --model /root/autodl-tmp/Hulu-Med-4B \
  --probe_dir \
    /root/autodl-tmp/Hulumed/probe/results/follow_linear_before_answer \
  --probe_label follow_conflict_probe \
  --prompt_types base \
  --group_field none \
  --group_name base_all \
  --score_type logit \
  --out_png /root/autodl-tmp/Hulumed/probe/results/base_follow_probes_logit_before_answer.png \
  --out_json /root/autodl-tmp/Hulumed/probe/results/base_follow_probes_logit_before_answer.json
