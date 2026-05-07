# SLAKE Image-Conflict Pipeline

This project mirrors `Slake_vqa/text_conflict`, but replaces text-evidence conflict with image masking conflict.

Supported `--model` values:

- `hulumed-4b` -> `ZJU-AI4H/Hulu-Med-4B`
- `InternVL3_5-4B` -> `OpenGVLab/InternVL3_5-4B-HF`
- `Qwen3.5-4B` -> `Qwen/Qwen3.5-4B`
- `qwen3vl-4b` -> `Qwen/Qwen3-VL-4B-Instruct`

Core setting:

- `NC` uses the original image.
- `IC` uses the same prompt, but masks the target region in the image.
- There is no evidence block.
- The prompt explicitly allows `unknown`.
- The desired `IC` behavior is that the model outputs `unknown` after the relevant region has been masked.

How target masking is resolved:

1. Read `detection.json` from the same image folder as `source.jpg`.
2. Try to match the correct answer to one or more detection labels.
3. If the answer itself is not an object label, fall back to matching labels mentioned in the question.
4. Use `mask.png` inside the matched bounding boxes when available; otherwise fall back to black rectangles.

Front-half data logic:

1. `prepare_slake_dataset.py`
   - Keep only English, non-KG, `CLOSED` questions by default.
   - Resolve `image_path`, `detection_path`, `mask_path`.
   - Classify each closed question into a maskable vs non-maskable type.
   - For maskable rows, resolve `IC` targets with a strict-first matcher that prefers exact organ labels over disease-label fallbacks.
   - Write helper fields such as `ic_question_type`, `ic_maskable`, `ic_match_policy`, `ic_query_targets`, and `ic_skip_reason`.
2. `filter_fine_grained.py`
   - Keep only rows where the model answers correctly on `NC`.
   - Unlike `text_conflict`, there is no `CC`; only `NC` correctness is required.
3. `split_train_val.py`
   - Split the filtered rows into train/val CSVs.
4. `eval_slake_mm_image_conflict.py`
   - Run generation on both `NC` and `IC`.
   - Report how often `IC` becomes `unknown`, stays `gold`, or becomes something else.

Recommended run order:

```bash
python prepare_slake_dataset.py --dataset_root ../data --out_dir ./data --proxy_strategy strict --target_mode resolved_targets
python filter_fine_grained.py --in_csv ./data/slake_closed_all.csv --wrong_csv ./data/slake_closed_all_wrong_backup.csv --image_root . --model hulumed-4b --out_csv ./data/hulumed4b/slake_nc_correct_ic_ready.csv
python split_train_val.py --in_csv ./data/hulumed4b/slake_nc_correct_ic_ready.csv --out_train ./data/hulumed4b/slake_nc_correct_ic_ready_train.csv --out_val ./data/hulumed4b/slake_nc_correct_ic_ready_val.csv
python eval_slake_mm_image_conflict.py --data_csv ./hulumed4b/data/slake_nc_correct_ic_ready.csv --image_root . --model hulumed-4b --out_dir ./hulumed4b/eval-results_slake_image_conflict
```

`re_filter` now uses closed-set log-prob scoring on `gold / wrong / unknown` instead of free generation. If the input CSV does not already contain a `wrong` column, it will try to fill one from `data/slake_closed_all_wrong_backup.csv` and then fall back to `../text_conflict/data/slake_closed_all.csv`.

To maximize data coverage for CLOSED questions, you can prepare the dataset with all detected targets masked instead of resolving one target from the question:

```bash
TARGET_MODE=all_detections bash scripts/prepare_slake_dataset.sh
```

In `all_detections` mode, `prepare_dataset` writes every label from `detection.json` into `ic_target_labels`, which usually drives `Unresolved IC targets` close to zero.

Server-style shell wrappers are available under `scripts/`.

`run_job` launcher:

```bash
python scripts/run_job.py --task prepare_dataset --model hulumed4b
python scripts/run_job.py --task re_filter --model hulumed4b
python scripts/run_job.py --task split_train_val --model hulumed4b
python scripts/run_job.py --task eval --model internvl35_4b --env DEVICE_MAP=auto
python scripts/run_job.py --task layer_trace --model internvl35_4b --resume
python scripts/run_job.py --task head_scan_accel --model internvl35_4b --resume
python scripts/run_job.py --task select_heads --model internvl35_4b
python scripts/run_job.py --task ablate_head --model internvl35_4b --resume
python scripts/run_job.py --task ablate_head_random --model internvl35_4b --resume
python scripts/run_job.py --task probe_prepare_data --model qwen35_4b
python scripts/run_job.py --task probe_extract_features --model qwen35_4b
python scripts/run_job.py --task probe_train --model qwen35_4b
python scripts/run_job.py --task tuned_lens_prepare_data --model qwen35_4b
python scripts/run_job.py --task tuned_lens_train --model qwen35_4b
python scripts/run_job.py --task ablation_flip_analyze --model internvl35_4b --resume
python scripts/run_job.py --task eval --model qwen3vl_4b --env DEVICE_MAP=auto --resume
python scripts/run_job.py --list
```

The launcher and all prepare/train/eval stages reuse the same resume/output style as `text_conflict`:

- per-task resume state lives under `resume/`
- model-relative outputs are rewritten under the selected model slug
- long-running shell wrappers accept `RESUME=1`
- launcher run records are saved under `runs/`

Suggested setup for server use:

1. Copy `scripts/run_job.local.example.json` to `scripts/run_job.local.json`.
2. Replace each `model_path` with your server-local model directory.
3. Run one task at a time with `python scripts/run_job.py --task ... --model ...`.
4. Add `--resume` when you want the stage to continue from the saved resume state.

`prepare_dataset` defaults to `strict` target matching. To allow organ-to-disease fallback when no clean organ label exists, run:

```bash
PROXY_STRATEGY=proxy bash scripts/prepare_slake_dataset.sh
```

Deep analysis chain:

1. `layer_trace`
2. `head_scan` or `head_scan_accel`
3. `select_heads`
4. `ablate_head` and `ablate_head_random`
5. `probe_*` and `tuned_lens_*`
6. `ablation_flip_analyze`

The deep analysis stages keep the old internal metric name `follow_conflict`, but in `image_conflict` it means the model followed the masked-image intervention by preferring `unknown` over the gold answer.
