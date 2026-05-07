# Portable bundle for `text_layer_head_all`

This bundle contains the minimum code and JSON data needed to regenerate these six files:

- `fig4_hulumed4b.png` / `fig4_hulumed4b.pdf`
- `fig4_internvl35_4b.png` / `fig4_internvl35_4b.pdf`
- `fig5_hulumed4b.png` / `fig5_hulumed4b.pdf`
- `fig5_internvl35_4b.png` / `fig5_internvl35_4b.pdf`
- `text_layer_head_hulumed4b_notext.png` / `text_layer_head_hulumed4b_notext.pdf`
- `text_layer_head_internvl35_4b_notext.png` / `text_layer_head_internvl35_4b_notext.pdf`

## Layout

- `scripts/`: portable plotting scripts with relative paths
- `data/`: required JSON inputs
- `outputs/single_panels/`: generated figures

## Run

From the bundle root:

```bash
python3 scripts/generate_requested_single_panels.py
```

Generated files will appear in `outputs/single_panels/`.

## Python dependencies

- `matplotlib`
- `numpy`
- `pillow`
