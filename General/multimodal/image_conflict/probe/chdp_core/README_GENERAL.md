# CHDP Core Integration (General)

This folder is copied from `VQA_RAD/Hulu-med/CHDP` and is used as the probe core for General multimodal conflict experiments.

Recommended flow:
1. Prepare dataset CSV first (closed + wrong answers).
2. Run eval/re-filter to get NC/IC outcomes.
3. Build CHDP manifest (`prepare_manifest.py`).
4. Dump atomic features (`dump_features.py`).
5. Build dataset features (`build_dataset.py`).
6. Train probe (`train_probe.py`).

Use dataset-specific result roots:
`<MODEL_SLUG>/<DATASET_NAME>/result_<POSITION>_slake`
