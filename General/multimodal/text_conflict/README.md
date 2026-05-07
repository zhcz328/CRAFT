# SLAKE Text-Conflict Pipeline

This project mirrors the `Slake_vqa/Hulu-med/text_conflict` workflow, but adds model switching so the same pipeline can be reused across multiple 4B models without changing code.

Supported `--model` values:

- `hulumed-4b` -> `ZJU-AI4H/Hulu-Med-4B`
- `InternVL3_5-4B` -> `OpenGVLab/InternVL3_5-4B-HF`
- `Qwen3.5-4B` -> `Qwen/Qwen3.5-4B`
- `qwen3vl-4B` -> `Qwen/Qwen3-VL-4B-Instruct`

You can also pass a local model path or an HF repo id directly. If `--model` is a local path, also pass `--model_name` so the workflow knows which family-specific logic and output slug to use. Relative result paths are automatically rewritten to `<model_slug>/<original_path>`, for example `result` becomes `qwen3vl_4b/result`.

Expected local dataset layout:

```text
../../data/
  train.json
  validation.json
  test.json
  imgs/
    imgs/
      xmlab0/
        source.jpg
        mask.png
        question.json
        detection.json
  KG/
    KG/
      ...
```

Current front-half data logic:

1. `prepare_slake_dataset.py`
   - Keep only English, non-KG, `CLOSED` questions.
   - Resolve each row to a real `image_path`.
2. `generate_wrong_candidates.py`
   - Add a heuristic `wrong` column without touching the model.
   - Export both CSV and XLSX so you can manually edit `wrong` if needed.
3. `filter_fine_grained.py`
   - Read the CSV/XLSX with `wrong`.
   - Keep only rows where the selected model prefers `gold` over `wrong` in both `NC` and `CC`.

Recommended run order:

```bash
python prepare_slake_dataset.py --dataset_root ../../data --out_dir ./data
python generate_wrong_candidates.py --in_csv ./data/slake_closed_all.csv --out_csv ./data/slake_closed_with_wrong.csv
# optional: manually edit ./data/slake_closed_with_wrong.xlsx
python filter_fine_grained.py --in_csv ./data/slake_closed_with_wrong.csv --image_root . --model hulumed-4b --out_csv ./data/slake_nc_cc_both_correct.csv
# local-path example:
python filter_fine_grained.py --in_csv ./data/slake_closed_with_wrong.csv --image_root . --model /path/to/local/model --model_name InternVL3_5-4B --out_csv ./data/slake_nc_cc_both_correct.csv
python split_train_val.py --in_csv ./data/slake_nc_cc_both_correct.csv --out_train ./data/slake_nc_cc_both_correct_train.csv --out_val ./data/slake_nc_cc_both_correct_val.csv
```

Legacy note:

- `filter_data.py` is still kept in the repo, but the simpler recommended path is now:
  `prepare -> generate wrong -> fine-grained NC/CC filter`.

The reused analysis scripts (`ablate_head.py`, `analyze_ablation_flip_with_probe_and_lens.py`, `probe/*`, `tuned_lens/*`) keep the same logic as the VQA-RAD project, but now resolve model aliases and keep outputs split by model slug.

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

- Every launch writes a record under `runs/` with the resolved script path, model path, and actual command, so it is easy to check what each terminal is really running.
- The parallel launchers pin one shard per visible GPU and merge results back into the same output layout used by the single-process scripts.
