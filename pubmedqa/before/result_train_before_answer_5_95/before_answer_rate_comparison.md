# Before Answer Rate Comparison

`baseline` comes from the `baseline` section inside the local summaries in this directory.
`random` comes from `random_ablation_evalstyle_ic_only_summary_before_answer_*.json`, using the `masked` section.
`ablate` comes from `ablation_evalstyle_ic_only_summary_before_answer_*.json`, using the `masked` section.

##### All

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 97.47% | 91.92% | 5.56% | 2.53% |
| random | 88.38% | 72.73% | 15.66% | 11.62% |
| ablate | 5.56% | 3.03% | 2.53% | 94.44% |

##### Train

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 97.48% | 91.60% | 5.88% | 2.52% |
| random | 90.76% | 73.95% | 16.81% | 9.24% |
| ablate | 5.04% | 2.52% | 2.52% | 94.96% |

##### Val

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 97.47% | 92.41% | 5.06% | 2.53% |
| random | 84.81% | 70.89% | 13.92% | 15.19% |
| ablate | 6.33% | 3.80% | 2.53% | 93.67% |
