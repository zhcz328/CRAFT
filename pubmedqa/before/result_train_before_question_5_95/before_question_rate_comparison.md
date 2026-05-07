# Before Question Rate Comparison

`baseline` comes from the `baseline` section inside the local summaries in this directory.
`random` comes from `random_ablation_evalstyle_ic_only_summary_before_question_*.json`, using the `masked` section.
`ablate` comes from `ablation_evalstyle_ic_only_summary_before_question_*.json`, using the `masked` section.

## All

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 68.69% | 46.46% | 22.22% | 31.31% |
| random | 42.42% | 22.73% | 19.70% | 57.58% |
| ablate | 5.56% | 2.02% | 3.54% | 94.44% |

## Train

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 63.87% | 44.54% | 19.33% | 36.13% |
| random | 43.70% | 24.37% | 19.33% | 56.30% |
| ablate | 5.04% | 1.68% | 3.36% | 94.96% |

## Val

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 75.95% | 49.37% | 26.58% | 24.05% |
| random | 40.51% | 20.25% | 20.25% | 59.49% |
| ablate | 6.33% | 2.53% | 3.80% | 93.67% |
