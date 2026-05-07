# Ablation Transition Tables

Definition:

- `C->W`: baseline correct under the `IC` prompt, but wrong after ablation
- `W->C`: baseline wrong under the `IC` prompt, but correct after ablation
- `Same C`: correct before and after ablation
- `Same W`: wrong before and after ablation
- Ratio columns are computed as `count / N`

Correctness criterion:

- Baseline uses the `baseline.ic` field inside each local ablation result file
- Ablation uses the `masked.ic` field inside each local ablation result file
- A sample is counted as correct when the prediction equals `gold`

Sources used in this file:

- `result_train_before_answer_5_95/ablation_evalstyle_ic_only_preds_before_answer_<split>.jsonl`
- `result_train_before_question_5_95/ablation_evalstyle_ic_only_preds_before_question_<split>.jsonl`
- `result_train_prefix_5_95/ablation_evalstyle_ic_only_preds_prefix_<split>.jsonl`

## Before Answer

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 202 | 4 | 1.98% | 190 | 94.06% | 1 | 0.50% | 7 | 3.47% |
| `train` | 121 | 3 | 2.48% | 115 | 95.04% | 0 | 0.00% | 3 | 2.48% |
| `val` | 81 | 1 | 1.23% | 75 | 92.59% | 1 | 1.23% | 4 | 4.94% |

## Before Question

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 202 | 0 | 0.00% | 129 | 63.86% | 62 | 30.69% | 11 | 5.45% |
| `train` | 121 | 0 | 0.00% | 72 | 59.50% | 43 | 35.54% | 6 | 4.96% |
| `val` | 81 | 0 | 0.00% | 57 | 70.37% | 19 | 23.46% | 5 | 6.17% |

## Prefix

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 202 | 0 | 0.00% | 158 | 78.22% | 14 | 6.93% | 30 | 14.85% |
| `train` | 121 | 0 | 0.00% | 91 | 75.21% | 11 | 9.09% | 19 | 15.70% |
| `val` | 81 | 0 | 0.00% | 67 | 82.72% | 3 | 3.70% | 11 | 13.58% |
