# Fig. 8 Data Sources

This figure reads validation metrics from:

- Probe-A: `summary.json -> metrics_by_layer[*].val_metrics.auroc`
- Probe-B: `summary.json -> metrics_by_layer[*].val_metrics.macro_f1`

Layers in the plot are displayed as 1-based layer indices. The JSON files store `layer_idx` as 0-based indices.

The shaded regions are local visual bands estimated from adjacent-layer score variation because these result files contain one run per layer. They are not confidence intervals.

| Probe | Model | Metric | Source | Best plotted layer | Best validation score |
|---|---|---|---|---:|---:|
| Probe-A: Conflict Detection | Qwen3-4B | auroc | `/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/before_question/results/conflict_linear_before_question/summary.json` | 33 | 0.996537 |
| Probe-B: Follow-Conflict Prediction | Qwen3-4B | macro_f1 | `/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/before_question/results/follow_linear_before_question/summary.json` | 28 | 0.769697 |
| Probe-A: Conflict Detection | InternVL3.5-4B | auroc | `/root/autodl-tmp/probe/internvl35_4b/results/before_question/conflict_linear/summary.json` | 34 | 0.998147 |
| Probe-B: Follow-Conflict Prediction | InternVL3.5-4B | macro_f1 | `/root/autodl-tmp/probe/internvl35_4b/results/before_question/follow_conflict_linear/summary.json` | 33 | 0.872321 |
| Probe-A: Conflict Detection | HuluMed | auroc | `/root/autodl-tmp/Hulumed/probe_slake_vqa/results/conflict_linear_before_question/summary.json` | 29 | 0.998221 |
| Probe-B: Follow-Conflict Prediction | HuluMed | macro_f1 | `/root/autodl-tmp/Hulumed/probe_slake_vqa/results/follow_linear_before_question/summary.json` | 33 | 0.946796 |
| Probe-A: Conflict Detection | Llama3.2-3B | auroc | `/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/before_question/conflict_linear/summary.json` | 16 | 1.000000 |
| Probe-B: Follow-Conflict Prediction | Llama3.2-3B | macro_f1 | `/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/before_question/follow_linear/summary.json` | 22 | 0.882353 |