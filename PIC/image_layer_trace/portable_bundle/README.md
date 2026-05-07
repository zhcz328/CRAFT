# Portable bundle for `image_layer_trace`

This bundle contains the minimum code and JSON data needed to regenerate:

- `image_conflict_hallucination_relief_bubble.png`
- `image_conflict_hallucination_relief_bubble.pdf`

## Layout

- `scripts/`: portable plotting script with relative paths
- `data/`: required JSON inputs
- `outputs/`: generated figure files

## Run

From the bundle root:

```bash
python3 scripts/draw_hallucination_relief_bubble.py
```

Generated files will appear in `outputs/`.

## Python dependencies

- `matplotlib`
- `numpy`
