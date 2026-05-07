# multi_layer_scale

独立实验目录，不改动 `/root/logit_lens/Slake_vqa/image_conflict` 主工程代码。

这里现在是完整实验链路，顺序固定为：

1. `train` 数据集跑 `layer_trace`
2. 根据该 `layer_trace` 产出的 `scan_plan` 跑 `train` 数据集 `head_scan`
3. 根据该 `head_scan` 结果自动选头，约束为：
   - `mean_abs_effect_reduction > 0`
   - `mean_abs_effect_reduction < 0.2`
   - 选头数量优先落在 `6~7`
4. 用该 `layer_scale` 下选出的头，在 `val` 数据集上跑 `ablate`

特殊规则：

- `layer_scale=0.2` 不重跑，直接复用原项目已有结果
- 如果某个 `layer_scale` 下最终选不出头，则不做后续 `ablate`，直接保存一个全 0 的结果文件
- 如果满足条件的头不足 `6~7` 个，就有多少个用多少个

脚本说明：

- `select_heads_auto.py`
  从单个 `head_scan_core_layers.json` 自动得到该 `layer_scale` 的 `selected_heads`
- `layer_trace_with_unknown_summary.py`
  对单个 `layer_scale` 跑 trace，并额外输出 `unknown rate` / `unknown rate delta`
- `summarize_layer_scale_sweep.py`
  汇总所有 `layer_scale` 的 `train trace` 结果
- `summarize_ablate_sweep.py`
  汇总所有 `layer_scale` 的 `val ablate` 结果

默认 sweep：

- `layer_scale = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0`

## 运行

```bash
cd /root/logit_lens/exp/multi_layer_scale
bash run_layer_scale_sweep.sh
```

完整闭环是：

```bash
train layer_trace -> train head_scan -> train auto_select -> val ablate
```

输出默认保存在：

```bash
/root/logit_lens/exp/multi_layer_scale/outputs
```

关键汇总文件：

- `outputs/layer_scale_sweep/layer_scale_sweep_train_summary.json`
- `outputs/layer_scale_sweep/layer_scale_sweep_train_summary.csv`
- `outputs/layer_scale_sweep/layer_scale_sweep_train_layers.csv`
- `outputs/layer_scale_sweep/layer_scale_sweep_val_ablate_summary.json`
- `outputs/layer_scale_sweep/layer_scale_sweep_val_ablate_summary.csv`

每个 `layer_scale` 的中间结果会放在：

- `outputs/layer_scale_sweep/layer_scale_0p0/`
- `outputs/layer_scale_sweep/layer_scale_0p2/`
- `outputs/layer_scale_sweep/layer_scale_0p4/`
- `outputs/layer_scale_sweep/layer_scale_0p6/`
- `outputs/layer_scale_sweep/layer_scale_0p8/`
- `outputs/layer_scale_sweep/layer_scale_1p0/`

每个子目录下都有：

- `trace_train/`
- `headscan_train/`
- `select_train/`
- `ablate/`
