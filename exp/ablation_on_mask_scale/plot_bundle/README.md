# ablation_on_mask_scale plotting bundle

This bundle is a lightweight plotting-only package.
It does not need model weights.

## Contents

- `plot_mask_scale_from_summary.py`: plot script using bundled summary data
- `data/mask_scale_sweep_summary.csv`: main input for plotting
- `data/mask_scale_sweep_summary.json`: JSON copy of the same aggregate data
- `data/scale_summaries/*.json`: per-scale summary files from the original run

## Run

```bash
python plot_mask_scale_from_summary.py
```

The output PNG is written to:

```bash
./mask_scale_vs_ic_unknown_rate.png
```

If needed, you can override paths:

```bash
python plot_mask_scale_from_summary.py \
  --summary_csv ./data/mask_scale_sweep_summary.csv \
  --out_png ./mask_scale_vs_ic_unknown_rate.png
```
