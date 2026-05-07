# SLAKE Hulu-med Pipeline

This project mirrors the `VQA_RAD/Hulu-med` workflow, but the data entrypoint is adapted for the local SLAKE layout under `../../data`.

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
   - Keep only rows where Hulu-Med prefers `gold` over `wrong` in both `NC` and `CC`.

Recommended run order:

```bash
python prepare_slake_dataset.py --dataset_root ../../data --out_dir ./data
python generate_wrong_candidates.py --in_csv ./data/slake_closed_all.csv --out_csv ./data/slake_closed_with_wrong.csv
# optional: manually edit ./data/slake_closed_with_wrong.xlsx
python filter_fine_grained.py --in_csv ./data/slake_closed_with_wrong.csv --image_root . --model <your_model_path> --out_csv ./data/slake_nc_cc_both_correct.csv
python split_train_val.py --in_csv ./data/slake_nc_cc_both_correct.csv --out_train ./data/slake_nc_cc_both_correct_train.csv --out_val ./data/slake_nc_cc_both_correct_val.csv
```

Legacy note:

- `filter_data.py` is still kept in the repo, but the simpler recommended path is now:
  `prepare -> generate wrong -> fine-grained NC/CC filter`.

The reused analysis scripts (`ablate_head.py`, `analyze_ablation_flip_with_probe_and_lens.py`, `probe/*`, `tuned_lens/*`) keep the same logic as the VQA-RAD project, with default paths pointing to SLAKE-oriented outputs.
