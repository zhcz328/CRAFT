# Patch Validate

This directory wraps the existing `layer_trace` implementations for:

- `/root/logit_lens/conflictmedqa/Qwen3-4B_exp`
- `/root/logit_lens/VQA_RAD/Hulu-med`
- `/root/logit_lens/VQA_RAD/text_conflict`

The goal is to avoid rewriting the tracing logic. Everything here only adds:

- a shared experiment manifest
- batch command generation for `k=2,4,8,16,32,all_token`
- metric aggregation
- appendix-style comparison plots

## Files

- `common.py`: shared config, result loaders, command builders, and metric helpers
- `generate_run_manifest.py`: writes the trace command manifest and shell launchers into `result/manifests/`
- `compute_patch_metrics.py`: reads completed trace JSONs and saves Smoothness / Robustness
- `plot_patch_comparison.py`: draws the `patch_k vs patch_all_token` comparison figures for `qwen3_4b` and `hulumed4b`

## Robustness definition

The default robustness score is:

- build a local-k consensus curve from `{k2, k4, k8, k16, k32}`
- min-max normalize each curve to focus on shape rather than raw scale
- compute `1 - mean(abs(curve_norm - consensus_norm))`

This gives a stable `[0, 1]` score where larger means the curve better matches the shared local-k pattern.

If you want a different appendix definition later, change `--robustness-method` in `compute_patch_metrics.py`.

## Usage

1. Generate commands and manifests:

```bash
python /root/logit_lens/PIC/patch_validate/generate_run_manifest.py
```

2. Run all traces, or only the two figure comparisons:

```bash
bash /root/logit_lens/PIC/patch_validate/result/manifests/run_patch_traces_all.sh
```

```bash
bash /root/logit_lens/PIC/patch_validate/result/manifests/run_patch_traces_compare_only.sh
```

3. After trace JSONs exist, compute metrics and plot figures:

```bash
python /root/logit_lens/PIC/patch_validate/compute_patch_metrics.py
python /root/logit_lens/PIC/patch_validate/plot_patch_comparison.py
```

## Assumptions used here

- Text setting uses `before_question`
- Multimodal setting uses `before_question`
- `qwen3_4b` local comparison uses `k=8`
- `hulumed4b` and `internvl35_4b` local comparison use `k=16`
- The local repo contains `llama32_3b` outputs and paths, so that model is wired as `Llama3.2-3B`

