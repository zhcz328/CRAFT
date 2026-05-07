# Ablation Transition Tables

Definition:

- `C->W`: baseline correct under the `conflict` prompt, but wrong after ablation
- `W->C`: baseline wrong under the `conflict` prompt, but correct after ablation
- `Same C`: correct before and after ablation
- `Same W`: wrong before and after ablation
- Ratio columns are computed as `count / N`

Baseline sources:

- `result_all_positions/conflict_positions.jsonl`
- `result_all_positions/conflict_positions_train.jsonl`
- `result_all_positions/conflict_positions_val.jsonl`

Ablation sources:

- `result_train/<position>/ablate_heads_all.jsonl`
- `result_train/<position>/ablate_heads_val.jsonl`
- `result_train/<position>/ablate_heads_train.jsonl` when aligned

Note:

- For `before_question/train`, the current `ablate_heads_train.jsonl` does not align with the train split keys, so the train rows below are reconstructed as `ablate_heads_all.jsonl - ablate_heads_val.jsonl`.

## Before Answer

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 386 | 56 | 14.51% | 190 | 49.22% | 25 | 6.48% | 115 | 29.79% |
| `train` | 212 | 30 | 14.15% | 104 | 49.06% | 17 | 8.02% | 61 | 28.77% |
| `val` | 174 | 26 | 14.94% | 86 | 49.43% | 8 | 4.60% | 54 | 31.03% |

## Before Question

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 386 | 4 | 1.04% | 77 | 19.95% | 28 | 7.25% | 277 | 71.76% |
| `train` | 212 | 2 | 0.94% | 29 | 13.68% | 21 | 9.91% | 160 | 75.47% |
| `val` | 174 | 2 | 1.15% | 48 | 27.59% | 7 | 4.02% | 117 | 67.24% |

## Prefix

| Split | N | C->W | C->W Ratio | W->C | W->C Ratio | Same C | Same C Ratio | Same W | Same W Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | 386 | 1 | 0.26% | 183 | 47.41% | 67 | 17.36% | 135 | 34.97% |
| `train` | 212 | 1 | 0.47% | 103 | 48.58% | 38 | 17.92% | 70 | 33.02% |
| `val` | 174 | 0 | 0.00% | 80 | 45.98% | 29 | 16.67% | 65 | 37.36% |
