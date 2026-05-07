# CRAFT: Causal Responsibility and Failure Tracing in Medical Vision Language Models

## Abstract

As vision language models are increasingly deployed in clinical diagnosis, understanding how they internally resolve competing visual and textual signals becomes a safety imperative. Existing mechanistic analyses remain confined to unimodal text and offer no explanation for why a single misleading sentence can override a correct image-based diagnosis, or why a model commits to a confident answer despite insufficient visual evidence. We find that these two safety risks, **arbitration failure** where textual context overrides visual grounding and **brake failure** where the model commits without adequate evidence, are mediated by spatially disjoint attention-head populations: arbitration heads form a mid-to-deep wideband reflecting cross-layer evidence competition, while brake heads concentrate at the deepest layers as late-stage answer-commitment gates. To ground these observations in causal circuitry, we introduce **CRAFT**, which localizes each failure mode to a minimal causal head set via dual selection criteria and then verifies necessity and sufficiency through temporal probes and Tuned Lens trajectory analysis. Excising arbitration heads sharply reduces conflict following with negligible degradation on clean inputs, while excising brake heads restores appropriate abstention under degraded visual evidence. The two interventions target spatially disjoint head sets and produce distinct corrective effects, underscoring the mechanistic separability of the two failure modes. Experiments across multiple medical VQA benchmarks and VLM architectures validate both the localization and the interventions, demonstrating that the identified heads causally drive each failure mode and that targeted modulation generalizes without retraining.

## Overview

This repository contains the experimental code for **CRAFT**, a mechanistic interpretability framework for tracing and intervening on failure modes in medical language models and medical vision-language models.

The paper studies two distinct safety-critical failure modes:

- **Arbitration failure**: the model has access to the correct visual evidence, but misleading text overrides the answer.
- **Brake failure**: the visual evidence is insufficient or degraded, but the model still commits to a concrete answer instead of abstaining with `unknown`.

CRAFT localizes the responsible internal components through a coarse-to-fine intervention pipeline:

1. **Layer-wise scoring** to identify a critical band.
2. **Head-level selection** using causal effect and clean-capacity preservation criteria.
3. **Targeted intervention** by excising the selected heads.
4. **Temporal validation** with probes and Tuned Lens trajectory analysis.

The implementation in this repository spans:

- text-only conflict analysis
- multimodal text-conflict analysis
- multimodal image-conflict / abstention analysis
- threshold sweeps and selection-criteria ablations
- observational vs. interventional comparisons
- figure-generation scripts for the paper

## Paper Summary

CRAFT is designed to answer a mechanistic question: **which internal attention heads causally drive unsafe conflict-following and over-commitment behaviors in medical models?**

The main findings reflected in the paper are:

- Arbitration-sensitive heads form a **broad mid-to-deep layer band**.
- Brake-sensitive heads concentrate in a **small late-layer circuit**.
- The two head sets are **spatially disjoint**, suggesting mechanistic separability.
- Excising arbitration heads reduces conflict following while preserving clean accuracy.
- Excising brake heads increases the model's willingness to abstain under degraded visual evidence.

## Repository Coverage

The repository is organized by task family rather than by a single unified training entrypoint. The main experiment groups are:

- **Text-only conflict**
  - `conflictmedqa/`
  - `pubmedqa/`
  - `pubmedqa_before/`
- **Multimodal text conflict**
  - `VQA_RAD/text_conflict/`
  - `VQA_RAD/Hulu-med/`
  - `VQA_RAD/qwen3-VL/`
  - `Slake_vqa/text_conflict/`
  - `General/multimodal/text_conflict/`
- **Multimodal image conflict / brake failure**
  - `Slake_vqa/image_conflict/`
  - `heal-medvqa/`
  - `General/multimodal/image_conflict/`
- **Analysis and visualization**
  - `selection_criteria_ablation/`
  - `ablation_tau/`
  - `image_ablation_tau/`
  - `PIC/`
  - `exp/`

The shared reusable multimodal logic is mostly concentrated under:

- `General/multimodal/text_conflict/`
- `General/multimodal/image_conflict/`
- `General/multimodal/dataset_adapters.py`

## Benchmarks in the Paper

The paper covers both text-only and multimodal medical QA settings:

- **ConflictMedQA**
- **PubMedQA**
- **VQA-RAD**
- **SLAKE**
- **Heal-MedVQA**

These map to the repository as follows:

- `conflictmedqa/` for ConflictMedQA
- `pubmedqa/` and `pubmedqa_before/` for PubMedQA
- `VQA_RAD/` for VQA-RAD
- `Slake_vqa/` for SLAKE
- `heal-medvqa/` plus shared image-conflict modules for Heal-MedVQA-related experiments

## Models in the Paper

The paper reports experiments across the following models:

- **Qwen3-4B**
- **Llama3.2-3B**
- **Hulu-Med 4B**
- **Hulu-Med 7B**
- **Hulu-Med 7B**
- **InternVL3.5-4B**
- **Qwen3-VL-8B**

## Core Workflow

Although each benchmark has its own scripts, the recurring CRAFT pipeline is:

1. Prepare or filter a clean/conflict/degraded split.
2. Run a **head scan** or layer scan.
3. Select intervention targets.
4. Run **head ablation / excision**.
5. Trace layer-wise behavior.
6. Optionally run probe and Tuned Lens analyses.

### Text-only workflow

Representative paths:

- `conflictmedqa/Qwen3-4B_exp/`
- `pubmedqa/`

Typical steps:

```bash
# PubMedQA example
python /root/logit_lens/pubmedqa/filter_data.py
python /root/logit_lens/pubmedqa/split_train_val.py
python /root/logit_lens/pubmedqa/head_scan_on_pubmed.py
python /root/logit_lens/pubmedqa/select_heads.py
python /root/logit_lens/pubmedqa/ablate_selected_heads.py
python /root/logit_lens/pubmedqa/layer_trace_on_pubmed.py
```

ConflictMedQA uses a parallel structure under:

```bash
/root/logit_lens/conflictmedqa/Qwen3-4B_exp
```

Important scripts there include:

- `head_scan.py`
- `head_scan_inf.py`
- `select_heads.py`
- `ablate_head.py`
- `ablate_head_inf.py`
- `layer_trace.py`

### Multimodal arbitration workflow

Representative paths:

- `VQA_RAD/text_conflict/`
- `VQA_RAD/Hulu-med/`
- `VQA_RAD/qwen3-VL/`
- `Slake_vqa/text_conflict/`
- `General/multimodal/text_conflict/`

Typical steps:

```bash
python /root/logit_lens/VQA_RAD/text_conflict/filter_fine_grained.py
python /root/logit_lens/VQA_RAD/text_conflict/head_scan_vqarad_mm_fastcache.py
python /root/logit_lens/VQA_RAD/text_conflict/select_heads_merged_unique_layers.py
python /root/logit_lens/VQA_RAD/text_conflict/ablate_head.py
python /root/logit_lens/VQA_RAD/text_conflict/layer_trace_vqarad_mm_current_fixed_fastcache.py
```

### Multimodal brake-failure workflow

Representative paths:

- `Slake_vqa/image_conflict/`
- `General/multimodal/image_conflict/`
- `heal-medvqa/`

Typical steps:

```bash
python /root/logit_lens/Slake_vqa/image_conflict/filter_visual_dependent_subset.py
python /root/logit_lens/Slake_vqa/image_conflict/head_scan_slake_mm_fastcache.py
python /root/logit_lens/Slake_vqa/image_conflict/select_heads.py
python /root/logit_lens/Slake_vqa/image_conflict/ablate_head.py
python /root/logit_lens/Slake_vqa/image_conflict/layer_trace_slake_mm_current_fixed_fastcache.py
```

For the more reusable shared code path, see:

- `General/multimodal/image_conflict/scripts/`
- `General/multimodal/image_conflict/probe/`
- `General/multimodal/image_conflict/tuned_lens/`

## Probe and Tuned Lens Analysis

The paper validates the selected circuits with linear probes and Tuned Lens trajectory analysis. Relevant code is available in:

- `pubmedqa/probe/`
- `pubmedqa/tuned_lens/`
- `General/multimodal/text_conflict/probe/`
- `General/multimodal/text_conflict/tuned_lens/`
- `General/multimodal/image_conflict/probe/`
- `General/multimodal/image_conflict/tuned_lens/`

These modules are generally run **after** the main scan and ablation outputs have been generated.

## Additional Analysis Modules

### Threshold sweeps

- `ablation_tau/`
- `image_ablation_tau/`

Examples:

```bash
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target qwen_conflictmedqa_before_question_val
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target hulumed4b_image_conflict_val
```

### Selection-criteria ablation

This module compares different head-selection rules such as CER-only, BCP-only, and the dual-criterion setting used by CRAFT.

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models all \
  --output_root /root/logit_lens/selection_criteria_ablation/results \
  --python python
```

### Patch / trace validation

```bash
python /root/logit_lens/PIC/patch_validate/generate_run_manifest.py
python /root/logit_lens/PIC/patch_validate/compute_patch_metrics.py
python /root/logit_lens/PIC/patch_validate/plot_patch_comparison.py
```

## Figure Reproduction

Paper plotting and summary scripts are mainly under `PIC/`:

- `PIC/fig1/` for conflict-follow summaries
- `PIC/fig4/` for layer-trace summaries
- `PIC/fig5/` for CER/BCP selection plots
- `PIC/fig6/` for downstream attention heatmaps
- `PIC/fig8/` for probe and Tuned Lens summaries
- `PIC/text_layer_head/` for text head-layout visualizations
- `PIC/image_layer_head/` for image head-layout visualizations
- `PIC/tuned_lens/` for tuned-lens paper plots

Examples:

```bash
python /root/logit_lens/PIC/fig4/draw_fig4_layer_trace.py
python /root/logit_lens/PIC/fig5/draw_fig5_bcp_cer.py
python /root/logit_lens/PIC/fig6/plot_fig6_attention_heatmap.py
python /root/logit_lens/PIC/fig8/plot_fig8.py
```

## Environment and Dependencies

This repository does **not** currently provide a single clean root-level `requirements.txt` for every experiment branch. In practice, most scripts assume:

- Python 3.10+
- PyTorch with CUDA
- `transformers`
- `accelerate`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You will also need:

- local access to the corresponding model checkpoints
- local dataset files for the target benchmark
- a GPU inference environment suitable for multi-billion-parameter models

A minimal starting point is:

```bash
cd /root/logit_lens
conda create -n craft python=3.10
conda activate craft
```

Then install the packages required by the subproject you want to run.

## Project Structure

```text
logit_lens/
├── conflictmedqa/               # text-only conflict analysis
├── pubmedqa/                    # text-only PubMedQA pipeline
├── pubmedqa_before/             # older PubMedQA branch
├── VQA_RAD/                     # VQA-RAD multimodal experiments
├── Slake_vqa/                   # SLAKE multimodal experiments
├── heal-medvqa/                 # Heal-MedVQA related utilities
├── General/multimodal/          # shared multimodal code
├── ablation_tau/                # threshold sweeps for text settings
├── image_ablation_tau/          # threshold sweeps for image conflict
├── selection_criteria_ablation/ # CER/BCP ablation studies
├── exp/                         # extra experiments and transfer analyses
├── PIC/                         # paper figure generation
├── attention_heatmap/           # additional visualization assets
└── readme/                      # paper source and README drafts
```

## Notes

- Many scripts rely on local absolute paths and research-environment assumptions.
- Some branches are historical or preserve earlier settings, for example `*_before`.
- Several result files and generated artifacts are intentionally retained in the repository because they support figure reproduction and appendix analysis.
- The paper's full model list is broader than the repository's cleanly exposed runnable directories; this is especially relevant for **Hulu-Med 7B**.

## Citation

If you use this codebase, please cite the paper:

```bibtex
@article{craft_medvlm,
  title={CRAFT: Causal Responsibility and Failure Tracing in Medical Vision Language Models},
  author={Anonymous Authors},
  journal={NeurIPS 2026 submission},
  year={2026}
}
```
