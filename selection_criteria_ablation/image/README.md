# SLAKE Image Selection-Criteria Ablation

这个目录专门放 `Slake_vqa/image_conflict` 的 selection-criteria ablation 代码。

代码独立保存在这里，不修改原始 `image_conflict` 项目；运行时只会复用那边现有的：

- `head_scan_core_layers.json`
- `selected_heads_core_layers.json`
- `ablate_head.py`

主要指标：

- `ctx_unknown_rate_delta`

比较的选择准则：

- `dual`：当前已有选头结果，只读取现有结果，不重复跑
- `cer-only`：按 `mean_abs_effect_reduction` 降序
- `bcp-only`：按 `mean_abs_base_change` 升序

Dry run:

```bash
python /root/logit_lens/selection_criteria_ablation/image/run_selection_criteria_ablation_image.py \
  --models all \
  --output_root /root/logit_lens/selection_criteria_ablation/image/results
```

正式运行：

```bash
python /root/logit_lens/selection_criteria_ablation/image/run_selection_criteria_ablation_image.py \
  --models all \
  --run \
  --device cuda \
  --output_root /root/logit_lens/selection_criteria_ablation/image/results
```

输出：

- `manifest.json`
- `run_commands.sh`
- `ctx_unknown_rate_delta_comparison.json`
- `ctx_unknown_rate_delta_comparison.csv`
