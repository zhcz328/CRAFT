# CRAFT: Causal Responsibility and Failure Tracing in Medical Vision Language Models

## Abstract

As vision language models are increasingly deployed in clinical diagnosis, understanding how they internally resolve competing visual and textual signals becomes a safety imperative. Existing mechanistic analyses remain confined to unimodal text and offer no explanation for why a single misleading sentence can override a correct image based diagnosis, or why a model commits to a confident answer despite insufficient visual evidence. We find that these two safety risks, arbitration failure where textual context overrides visual grounding and brake failure where the model commits without adequate evidence, are mediated by spatially disjoint attention head populations: arbitration heads form a mid to deep wideband reflecting cross layer evidence competition, while brake heads concentrate at the deepest layers as late stage answer commitment gates. To ground these observations in causal circuitry, we introduce \method, which localises each failure mode to a minimal causal head set via dual selection criteria and then verifies necessity and sufficiency through temporal probes and Tuned Lens trajectory analysis. Excising arbitration heads sharply reduces conflict following with negligible degradation on clean inputs, while excising brake heads restores appropriate abstention under degraded visual evidence. The two interventions target spatially disjoint head sets and produce distinct corrective effects, underscoring the mechanistic separability of the two failure modes. Experiments across multiple medical VQA benchmarks and VLM architectures validate both the localisation and the interventions, demonstrating that the identified heads causally drive each failure mode and that targeted modulation generalises without retraining. 

## Requirements

- Python 3.10+ recommended
- PyTorch + CUDA environment
- `transformers`, `accelerate`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Local access to the target model checkpoints

This repository does **not** currently expose a single unified `requirements.txt` at the root. Most scripts assume an existing research environment with Hugging Face models, GPU inference, and standard scientific Python packages installed.

## Quick Start

### Environment

```bash
cd /root/logit_lens
conda create -n logit_lens python=3.10
conda activate logit_lens
```

Then install the packages required by the subproject you want to run. In practice, most experiments rely on a standard `torch + transformers + pandas + matplotlib` stack.

### Text Workflow: PubMedQA / ConflictMedQA

Typical text-side pipeline:

```bash
# 1. Filter / prepare data
python /root/logit_lens/pubmedqa/filter_data.py

# 2. Split train/val if needed
python /root/logit_lens/pubmedqa/split_train_val.py

# 3. Run head scan
python /root/logit_lens/pubmedqa/head_scan_on_pubmed.py

# 4. Select heads
python /root/logit_lens/pubmedqa/select_heads.py

# 5. Run ablation
python /root/logit_lens/pubmedqa/ablate_selected_heads.py

# 6. Trace layer-wise behavior
python /root/logit_lens/pubmedqa/layer_trace_on_pubmed.py
```

For the original ConflictMedQA path, the parallel workflow lives under:

```bash
/root/logit_lens/conflictmedqa/Qwen3-4B_exp
```

with core scripts such as:

- `head_scan.py` / `head_scan_inf.py`
- `select_heads.py`
- `ablate_head.py` / `ablate_head_inf.py`
- `layer_trace.py`

### Multimodal Workflow: VQA_RAD / SLAKE

Representative multimodal pipelines live in:

- `/root/logit_lens/VQA_RAD/Hulu-med`
- `/root/logit_lens/VQA_RAD/text_conflict`
- `/root/logit_lens/VQA_RAD/qwen3-VL`
- `/root/logit_lens/Slake_vqa/text_conflict`
- `/root/logit_lens/Slake_vqa/image_conflict`

Typical workflow:

```bash
# 1. Filter or prepare benchmark subset
python /root/logit_lens/VQA_RAD/text_conflict/filter_fine_grained.py

# 2. Run head scan
python /root/logit_lens/VQA_RAD/text_conflict/head_scan_vqarad_mm_fastcache.py

# 3. Select heads
python /root/logit_lens/VQA_RAD/text_conflict/select_heads_merged_unique_layers.py

# 4. Run ablation
python /root/logit_lens/VQA_RAD/text_conflict/ablate_head.py

# 5. Run layer trace
python /root/logit_lens/VQA_RAD/text_conflict/layer_trace_vqarad_mm_current_fixed_fastcache.py
```

For image-conflict experiments on SLAKE, use the corresponding scripts under:

```bash
/root/logit_lens/Slake_vqa/image_conflict
```

## Extended Analyses

### Threshold Sweep

The repository contains external threshold-sweep utilities for generating alternative selected-head sets while constraining them not to outperform the current main configuration.

Text / text-conflict:

```bash
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target qwen_conflictmedqa_before_question_val
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target hulumed_before_question_val --run-ablation --limit 3
```

Image-conflict:

```bash
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target hulumed4b_image_conflict_val
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target internvl35_4b_image_conflict_val --run-ablation --limit 3
```

### Selection-Criteria Ablation

This module compares different head-selection criteria such as `cer-only`, `bcp-only`, and existing dual criteria, while reusing the original evaluation scripts rather than duplicating model logic.

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models all \
  --output_root /root/logit_lens/selection_criteria_ablation/results \
  --python python
```

To run the actual evaluations:

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models all \
  --run \
  --output_root /root/logit_lens/selection_criteria_ablation/results \
  --python python
```

### Patch / Trace Validation

The `PIC/patch_validate` module wraps existing `layer_trace` implementations and adds manifest generation, metric aggregation, and appendix-style comparisons:

```bash
python /root/logit_lens/PIC/patch_validate/generate_run_manifest.py
python /root/logit_lens/PIC/patch_validate/compute_patch_metrics.py
python /root/logit_lens/PIC/patch_validate/plot_patch_comparison.py
```

## Probe and Tuned Lens

Several subprojects include probe training / scoring and tuned-lens trajectory analysis for studying intermediate representations before and after head ablation. Representative locations include:

- `/root/logit_lens/pubmedqa/tuned_lens`
- `/root/logit_lens/General/multimodal/text_conflict/probe`
- `/root/logit_lens/General/multimodal/text_conflict/tuned_lens`
- `/root/logit_lens/General/multimodal/image_conflict/probe`
- `/root/logit_lens/General/multimodal/image_conflict/tuned_lens`

These modules are typically used after core ablation outputs have already been generated.

## Figure Reproduction

Figure and panel scripts are organized under `/root/logit_lens/PIC`, including:

- `fig1`: conflict-follow rate summary
- `fig4`: layer-trace summary panels
- `fig5`: BCP/CER scatter analysis
- `fig6`: attention heatmaps
- `fig8`: probe and tuned-lens plots
- `text_layer_head` / `image_layer_head`: layer-head heatmaps
- `image_layer_trace`: hallucination-relief bubble plots

Examples:

```bash
python /root/logit_lens/PIC/fig4/draw_fig4_layer_trace.py
python /root/logit_lens/PIC/fig5/draw_fig5_bcp_cer.py
python /root/logit_lens/PIC/fig6/plot_fig6_attention_heatmap.py
python /root/logit_lens/PIC/fig8/plot_fig8.py
python /root/logit_lens/PIC/text_layer_head/plot_text_layer_head_from_headscan.py
python /root/logit_lens/PIC/image_layer_head/plot_image_layer_head_from_headscan.py
```

## Supported Models and Benchmarks

Representative models already wired in different submodules include:

- Qwen3-4B
- Llama-3.2-3B
- Hulu-med-4B
- InternVL3.5-4B
- Qwen3-VL variants

Representative benchmarks include:

- ConflictMedQA
- PubMedQA
- VQA_RAD
- SLAKE
- Heal-MedVQA

Because this is a research repository accumulated across several experiment tracks, model-path configuration is sometimes handled through script arguments and sometimes through local constants or environment variables. Please check the target subdirectory before launching large runs.

## Project Structure

```text
logit_lens/
├── conflictmedqa/               # text-only conflict analysis on ConflictMedQA
├── pubmedqa/                    # text-only pipeline for PubMedQA
├── VQA_RAD/                     # multimodal and text-conflict experiments on VQA_RAD
├── Slake_vqa/                   # multimodal text-conflict and image-conflict on SLAKE
├── heal-medvqa/                 # additional medical VQA experiments
├── General/multimodal/          # shared multimodal probe / tuned-lens analysis
├── ablation_tau/                # threshold-sweep utilities for text settings
├── image_ablation_tau/          # threshold-sweep utilities for image-conflict
├── selection_criteria_ablation/ # CER-only vs BCP-only vs dual selection analysis
├── PIC/                         # figure-generation and visualization scripts
├── observational_reranking/     # observational comparison utilities
├── attention_heatmap/           # attention visualization assets / scripts
└── readme/                      # reference README drafts
```

## Notes

- Many scripts expect local dataset files and model checkpoints that are not distributed with the repository.
- Some directories contain historical or backup variants such as `*_before`, which preserve earlier experiment branches.
- Result files, manifests, cached outputs, and generated figures are intentionally kept inside the repository because they are part of the analysis workflow.

## License

This repository is intended for research use. Add your preferred license here if you plan to release it publicly.
