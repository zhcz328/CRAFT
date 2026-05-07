# image_ablation_tau

这个目录用于给 `/root/logit_lens/Slake_vqa/image_conflict` 做一层外部阈值筛选，思路和 `/root/logit_lens/ablation_tau` 一样，但这里专门针对 `selected_heads_core_layers.json`。

## 当前规则

- 从 `head_scan_core_layers.json` 里真实出现过的阈值集合中随机采样
- 复用原项目 `select_heads_merged_unique_layers.py` 的筛选逻辑生成候选 head 组
- 去掉空集合和与当前主配置重复的 head 集合
- 可选调用原项目 `ablate_head.py`
- 指标使用：
  - `C2U = base_ctx_pred == gold -> ab_ctx_pred == unknown`
  - `U2O = base_ctx_pred == unknown -> ab_ctx_pred == other`
  - `UR = ab_ctx_pred == unknown` 的总体比例
- 接受规则：
  - `C2U` 比当前低
  - `U2O` 比当前高
  - `UR` 比当前高
  - 三项里满足任意两项即可保留

## 脚本

- `metrics_utils.py`
  - 计算 `C2U / U2O / UR`
  - 提供“任意两项满足”的比较规则

- `run_threshold_sweep.py`
  - 生成候选阈值
  - 落盘 `candidate_xx/selected_heads.json`
  - 可选运行 ablation
  - 最多保留 3 组满足规则的候选

## 支持 target

- `hulumed4b_image_conflict_val`
- `internvl35_4b_image_conflict_val`

兼容别名：

- `image_conflict_hulumed4b_val`
- `image_conflict_internvl35_4b_val`

## 用法

只生成当前指标和候选阈值：

```bash
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target hulumed4b_image_conflict_val
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target internvl35_4b_image_conflict_val
```

正式跑 ablation，并筛出最多 3 组满足规则的候选：

```bash
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target hulumed4b_image_conflict_val --run-ablation --limit 3
python /root/logit_lens/image_ablation_tau/run_threshold_sweep.py --target internvl35_4b_image_conflict_val --run-ablation --limit 3
```
