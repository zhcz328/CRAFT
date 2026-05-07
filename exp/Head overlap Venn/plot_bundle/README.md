# Head overlap Venn plotting bundle

This bundle is a lightweight plotting-only package.
It does not need model weights.

## Contents

- `plot_head_overlap_venn.py`: self-contained plotting script
- `data/*.json`: bundled selected-head inputs for both models

## Run

```bash
python plot_head_overlap_venn.py
```

Outputs:

- `./hulumed4b_head_overlap_venn.png`
- `./internvl35_4b_head_overlap_venn.png`
- `./head_overlap_summary.json`
- `./head_overlap_summary.md`
