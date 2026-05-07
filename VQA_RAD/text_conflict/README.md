# VQA-RAD Text-Conflict Pipeline

This directory is the VQA-RAD counterpart of `Slake_vqa/text_conflict`: same multi-model text-conflict workflow, but adapted to the VQA-RAD dataset layout under `VQA_RAD/data`.

Supported `--model` values:

- `hulumed-4b` -> `ZJU-AI4H/Hulu-Med-4B`
- `InternVL3_5-4B` -> `OpenGVLab/InternVL3_5-4B-HF`
- `Qwen3.5-4B` -> `Qwen/Qwen3.5-4B`
- `qwen3vl-4B` -> `Qwen/Qwen3-VL-4B-Instruct`

You can also pass a local model path or an HF repo id directly. If `--model` is a local path, also pass `--model_name` so the workflow can resolve the model family and output slug. Relative result paths are automatically rewritten to `<model_slug>/<original_path>`.

Expected local dataset layout:

```text
../data/
  vqa_rad_closed_all.csv
  VQA_RAD/
    vqa_rad_train.csv
    vqa_rad_valid.csv
    vqa_rad_test.csv
  VQA_RAD_Image_Folder/
    synpic*.jpg
```

Front-half data logic:

1. `generate_wrong_candidates.py`
   - Read `vqa_rad_closed_all.csv`
   - Add a heuristic `wrong` column without touching the model
   - Export both CSV and XLSX so you can manually edit `wrong`
2. `filter_fine_grained.py`
   - Read the CSV/XLSX with `wrong`
   - Keep only rows where the selected model prefers `gold` over `wrong` in both `NC` and `CC`
3. `split_train_val.py`
   - Split one model-specific filtered CSV into train/val CSVs for trace/head-scan/probe/tuned-lens

Recommended run order:

```bash
python generate_wrong_candidates.py \
  --in_csv ../data/vqa_rad_closed_all.csv \
  --out_csv ./data/vqa_rad_closed_with_wrong.csv

# optional: manually edit ./data/vqa_rad_closed_with_wrong.xlsx

python filter_fine_grained.py \
  --in_csv ./data/vqa_rad_closed_with_wrong.csv \
  --image_root ../data/VQA_RAD_Image_Folder \
  --model hulumed-4b \
  --out_csv ./data/hulumed4b/vqa_rad_nc_cc_both_correct.csv

python split_train_val.py \
  --in_csv ./data/hulumed4b/vqa_rad_nc_cc_both_correct.csv \
  --model hulumed-4b \
  --out_train ./data/hulumed4b/vqa_rad_nc_cc_both_correct_train.csv \
  --out_val ./data/hulumed4b/vqa_rad_nc_cc_both_correct_val.csv
```

The shell wrappers in `scripts/` assume the same layout by default:

- `scripts/generate_wrong_candidates.sh` reads `../data/vqa_rad_closed_all.csv`
- `scripts/re_filter.sh` reads `./data/vqa_rad_closed_with_wrong.csv`
- `scripts/eval_vqarad.sh`, `scripts/head_scan*.sh`, `scripts/layer_trace*.sh` read `./data/<model_slug>/vqa_rad_nc_cc_both_correct*.csv`
- Probe and tuned-lens outputs are stored under `<model_slug>/probe/...` and `<model_slug>/tuned_lens/...`

Launcher tip:

- If you are running several long experiments in parallel, use `scripts/run_job.py` so you only specify `task + model` once and let the launcher fill `MODEL_PATH` / `MODEL_NAME`.
- Copy `scripts/run_job.local.example.json` to `scripts/run_job.local.json`, set your local model paths once, then launch with:

```bash
python scripts/run_job.py --task head_scan --model qwen35_4b
python scripts/run_job.py --task layer_trace --model internvl35_4b
python scripts/run_job.py --task head_scan_accel_parallel --model internvl35_4b --env GPU_IDS=0,1,2
python scripts/run_job.py --task layer_trace_parallel --model internvl35_4b --env GPU_IDS=0,1,2
python scripts/run_job.py --task eval --model hulumed4b --env DEVICE_MAP=cuda:1
python scripts/run_job.py --list
```

- Every launch writes a record under `runs/` with the resolved script path, model path, and command.
- The parallel launchers pin one shard per visible GPU and merge results back into the same output layout used by the single-process scripts.
