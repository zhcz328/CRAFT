# Prefix Rate Comparison

`baseline` comes from the `baseline` section inside the local ablation summaries in this directory.
`random` comes from `random_ablation_evalstyle_ic_only_summary_prefix_*.json`, using the `masked` section.
`ablate` comes from `ablation_evalstyle_ic_only_summary_prefix_*.json`, using the `masked` section.

##### All

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 92.93% | 78.28% | 14.65% | 7.07% |
| random | 82.32% | 57.58% | 24.75% | 17.68% |
| ablate | 14.65% | 9.09% | 5.56% | 85.35% |

##### Train

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 90.76% | 73.95% | 16.81% | 9.24% |
| random | 79.83% | 56.30% | 23.53% | 20.17% |
| ablate | 15.13% | 10.08% | 5.04% | 84.87% |

##### Val

| Method | Changed Rate | Flip Rate | Abstain Rate | Resist Rate |
| --- | --- | --- | --- | --- |
| baseline | 96.20% | 84.81% | 11.39% | 3.80% |
| random | 86.08% | 59.49% | 26.58% | 13.92% |
| ablate | 13.92% | 7.59% | 6.33% | 86.08% |
