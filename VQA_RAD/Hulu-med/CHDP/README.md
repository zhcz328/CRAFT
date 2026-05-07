# CHDP

Conflict-Hijack Dynamic Probe 的最小可执行版本，按实验计划拆成 4 个阶段：

1. `prepare_manifest.py`
   根据现有 `train/val/test` CSV 和 `preds.jsonl`，筛出满足
   `NC == gold` 且 `IC == gold/conflict` 的干净样本，并生成成对 `NC/IC` 清单。

2. `dump_features.py`
   对每个样本分别跑 `NC` 和 `IC`，导出原子特征：
   `h_l`、`A_img/A_ctx/A_q`、`s_gold/s_conf`。

3. `build_dataset.py`
   从原子特征构造层级输入特征：
   `d_h`、`d_delta`、`A_img^IC`、`A_ctx^IC`、`A_ctx^IC-A_img^IC`、`M_gc^NC`、`M_gc^IC`。

4. `train_probe.py`
   训练主模型 `MLP + BiGRU + classifier`，以及文档里的 baseline / ablation 配置。

5. `plot_trajectories.py`
   输出单样本轨迹图和 Resist vs Hijack 平均轨迹图。

## 推荐命令

### 1. 生成训练清单

```bash
python VQA_RAD/Hulu-med/CHDP/prepare_manifest.py \
  --position before_question \
  --out_dir CHDP/data
```

默认会读取：

- `data/nc_cc_both_correct_rerun_tmp_train.csv`
- `data/nc_cc_both_correct_rerun_tmp_val.csv`
- 与 `position` 对应的 `eval-results.../preds.jsonl`

生成：

- `CHDP/data/train.jsonl`
- `CHDP/data/val.jsonl`

### 2. 导出原子特征

```bash
python VQA_RAD/Hulu-med/CHDP/dump_features.py \
  --manifest VQA_RAD/Hulu-med/CHDP/data/train.jsonl \
  --model /path/to/Hulu-Med-4B \
  --out VQA_RAD/Hulu-med/CHDP/features/train_atomic.pt

python VQA_RAD/Hulu-med/CHDP/dump_features.py \
  --manifest VQA_RAD/Hulu-med/CHDP/data/val.jsonl \
  --model /path/to/Hulu-Med-4B \
  --out VQA_RAD/Hulu-med/CHDP/features/val_atomic.pt
```

### 3. 构造 CHDP 层级特征

```bash
python VQA_RAD/Hulu-med/CHDP/build_dataset.py \
  --atomic_bundle VQA_RAD/Hulu-med/CHDP/features/train_atomic.pt \
  --out VQA_RAD/Hulu-med/CHDP/features/train_dataset.pt

python VQA_RAD/Hulu-med/CHDP/build_dataset.py \
  --atomic_bundle VQA_RAD/Hulu-med/CHDP/features/val_atomic.pt \
  --out VQA_RAD/Hulu-med/CHDP/features/val_dataset.pt
```

### 4. 训练主模型

```bash
python VQA_RAD/Hulu-med/CHDP/train_probe.py \
  --train_bundle VQA_RAD/Hulu-med/CHDP/features/train_dataset.pt \
  --val_bundle VQA_RAD/Hulu-med/CHDP/features/val_dataset.pt \
  --model_type bigru \
  --feature_set chdp_minimal \
  --out_dir VQA_RAD/Hulu-med/CHDP/results/chdp_bigru
```

### 5. 跑 baseline / ablation

单层线性 probe：

```bash
python VQA_RAD/Hulu-med/CHDP/train_probe.py \
  --train_bundle VQA_RAD/Hulu-med/CHDP/features/train_dataset.pt \
  --val_bundle VQA_RAD/Hulu-med/CHDP/features/val_dataset.pt \
  --model_type single_layer_linear \
  --feature_set chdp_minimal \
  --out_dir VQA_RAD/Hulu-med/CHDP/results/single_layer_linear
```

全层 flatten + MLP：

```bash
python VQA_RAD/Hulu-med/CHDP/train_probe.py \
  --train_bundle VQA_RAD/Hulu-med/CHDP/features/train_dataset.pt \
  --val_bundle VQA_RAD/Hulu-med/CHDP/features/val_dataset.pt \
  --model_type flat_mlp \
  --feature_set chdp_minimal \
  --out_dir VQA_RAD/Hulu-med/CHDP/results/flat_mlp
```

只用 attention 特征：

```bash
python VQA_RAD/Hulu-med/CHDP/train_probe.py \
  --train_bundle VQA_RAD/Hulu-med/CHDP/features/train_dataset.pt \
  --val_bundle VQA_RAD/Hulu-med/CHDP/features/val_dataset.pt \
  --model_type bigru \
  --feature_set attention_only \
  --out_dir VQA_RAD/Hulu-med/CHDP/results/attention_only
```

其他内置特征组：

- `score_only`
- `divergence_only`
- `no_divergence`
- `no_attention`
- `no_score`
- `chdp_plus_gap`

### 6. 画轨迹图

```bash
python VQA_RAD/Hulu-med/CHDP/plot_trajectories.py \
  --atomic_bundle VQA_RAD/Hulu-med/CHDP/features/val_atomic.pt \
  --out_dir VQA_RAD/Hulu-med/CHDP/plots/val
```

## 输出文件

`dump_features.py` 的 `.pt` 文件包含：

- `nc_hidden_states`, `ic_hidden_states`
- `nc_attn_img`, `nc_attn_ctx`, `nc_attn_q`
- `ic_attn_img`, `ic_attn_ctx`, `ic_attn_q`
- `nc_score_gold`, `nc_score_conflict`
- `ic_score_gold`, `ic_score_conflict`

`build_dataset.py` 的 `.pt` 文件包含：

- `features`: `[N, L, D]`
- `feature_names`
- `feature_groups`
- `labels`

## 当前实现约定

1. 目标位置固定为 prompt 最后一个 token，也就是 question end / assistant start。
2. `s_gold / s_conf` 目前使用答案的首 token logit。
3. 注意力聚合量使用“目标位置对各区域 token 的 head-mean attention mass 总和”。
4. `d_delta(0)` 使用第一层相对零向量的差值，后续层按相邻层差分计算。
