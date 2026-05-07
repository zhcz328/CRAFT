# headscan_slake_mm threshold recommendation

Source:
- `head_scan_core_layers.json`
- `head_scan_summary.json`

Scan setting:
- metric: `mean_abs_effect_reduction` on `follow_context`
- samples: `87`
- scanned layers: `11, 14, 15, 16, 17, 35`
- total scanned heads: `192`

Distribution:
- max: `0.5453`
- mean: `-0.0272`
- median: `-0.0158`
- positive heads: `69 / 192`
- clear rank gaps:
  - rank `1 -> 2`: `0.5453 -> 0.2299`
  - rank `5 -> 6`: `0.2091 -> 0.1767`
  - rank `11 -> 12`: `0.1329 -> 0.1106`

Recommended thresholds:
- conservative: `0.15`
  - selected heads: `8`
  - heads: `(15,8), (14,6), (16,22), (15,0), (14,18), (16,10), (15,25), (17,14)`
- balanced: `0.12`
  - selected heads: `11`
  - heads: `(15,8), (14,6), (16,22), (15,0), (14,18), (16,10), (15,25), (17,14), (16,3), (14,17), (16,12)`
- aggressive: `0.10`
  - selected heads: `14`
  - heads: `(15,8), (14,6), (16,22), (15,0), (14,18), (16,10), (15,25), (17,14), (16,3), (14,17), (16,12), (14,27), (16,25), (14,31)`

Notes:
- If you want only the strongest and most stable heads, start from `0.15`.
- If you want a better coverage/size tradeoff, `0.12` is the best default because it keeps the heads above the rank-11 cutoff and avoids the next obvious drop to `0.1106`.
- If downstream validation still benefits from more coverage, then expand to `0.10`.
