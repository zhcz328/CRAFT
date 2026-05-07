# Selection-Criteria Ablation

This folder contains a thin runner for the head-selection ablation:

- Qwen3-4B on ConflictMedQA
- Llama-3.2-3B on ConflictMedQA
- Hulu-med-4B on VQA_RAD
- InternVL3.5-4B on VQA_RAD

The runner reuses the existing project evaluators instead of duplicating model
logic:

- ConflictMedQA: `/root/logit_lens/conflictmedqa/Qwen3-4B_exp/ablate_head_inf.py`
- Hulu-med VQA_RAD: `/root/logit_lens/VQA_RAD/Hulu-med/ablate_head.py`
- InternVL VQA_RAD: `/root/logit_lens/VQA_RAD/text_conflict/ablate_head.py`

## What It Selects

For each model, the runner reads the existing head-scan output and writes two
new selected-head files:

- `cer-only`: sort by `mean_abs_effect_reduction` from high to low.
- `bcp-only`: sort by BCP from low to high.

BCP is read from `mean_abs_base_delta_change` when present. For VQA_RAD scan
files, the existing key is `mean_abs_base_change`, so the runner uses that as
the BCP fallback and records the source key in the CSV/JSON metadata.

Dual is not re-evaluated. The runner records the current Dual head file and
existing result paths in `dual_existing_result_manifest.json`.

## Dry Run

Generate selected heads, `manifest.json`, and a runnable shell command file
without loading any model:

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models all \
  --output_root /root/logit_lens/selection_criteria_ablation/results \
  --python python
```

Main outputs:

- `/root/logit_lens/selection_criteria_ablation/results/manifest.json`
- `/root/logit_lens/selection_criteria_ablation/results/run_commands.sh`
- Per-model selected heads under:
  `/root/logit_lens/selection_criteria_ablation/results/<model>/before_question/selected_heads/`

## Run Evaluations

Run everything directly through the Python runner:

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models all \
  --run \
  --output_root /root/logit_lens/selection_criteria_ablation/results \
  --python python
```

Or run the generated commands:

```bash
bash /root/logit_lens/selection_criteria_ablation/results/run_commands.sh
```

For a smoke test:

```bash
python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models qwen,llama \
  --run \
  --max_pairs 2

python /root/logit_lens/selection_criteria_ablation/run_selection_criteria_ablation.py \
  --models hulu,internvl \
  --run \
  --max_examples 2
```

## Useful Options

- `--models qwen,llama,hulu,internvl` selects a subset.
- `--position before_question` is the default and matches the current request.
- `--splits val` runs only the validation split.
- `--device cuda:0` controls VQA_RAD model placement.
- `--device_map cuda:0` controls ConflictMedQA model placement.
- `--skip_existing` avoids rerunning completed output files.
- `--target_tolerance 3` keeps CER-only/BCP-only head counts close to Dual.

Model paths can also be overridden with environment variables:

- `QWEN3_4B_MODEL_PATH`
- `LLAMA32_3B_MODEL_PATH`
- `HULUMED4B_MODEL_PATH`
- `INTERNVL35_4B_MODEL_PATH`

## Result Layout

Each model gets:

```text
results/<model>/before_question/
  dual_existing_result_manifest.json
  selected_heads/
    cer-only/cer-only_selected_heads.{json,csv}
    bcp-only/bcp-only_selected_heads.{json,csv}
  cer-only/
    cer-only_train...
    cer-only_val...
  bcp-only/
    bcp-only_train...
    bcp-only_val...
  logs/
```

ConflictMedQA outputs JSONL plus `_summary.json`. VQA_RAD outputs one JSON file
per split containing both summary fields and records, matching the existing
project scripts.
