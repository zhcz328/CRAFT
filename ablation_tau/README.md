# ablation_tau

这个目录放“不同阈值再选三组，但不允许强过当前主配置”的外层脚本和结果。

## 已有脚本

- `metrics_utils.py`
  - 统一计算 `CFR / C2W / W2C`
  - 兼容：
    - VQA `ablation.json` 格式
    - Qwen ConflictMedQA 旧版 `jsonl` 格式，并自动对齐 baseline `conflict_positions_val.jsonl`

- `run_threshold_sweep.py`
  - 复用原项目的 `select_heads` 逻辑生成不同阈值候选
  - 阈值不是按“离当前最近”挑，而是从 head-scan 里真实出现过的阈值集合中随机采样
  - 只保留“至少能选到 1 个头”的阈值组合，并去掉头集合重复的候选
  - 保存每组候选的 `selected_heads.json`
  - 可选调用原项目 `ablate_head` 脚本
  - 用 `(更低 CFR, 更低 C2W, 更高 W2C)` 的字典序判断某组是否“强过当前配置”
  - 如果强过当前配置，则标记为拒绝，不保留

## 当前已生成

- `hulumed_before_question_val/current_metrics.json`
- `internvl35_4b_text_conflict_before_question_val/current_metrics.json`
- `qwen_conflictmedqa_before_question_val/current_metrics.json`
- `llama32_3b_conflictmedqa_before_question_val/current_metrics.json`
- 各 target 下的 `manifest.json`
- 各 candidate 下的 `selected_heads.json` 和 `status.json`

当前这些 candidate 还处于 `pending_ablation`，因为还没正式跑新的 ablation。

## 用法

只生成当前指标和候选阈值：

```bash
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target hulumed_before_question_val
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target internvl35_4b_text_conflict_before_question_val
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target qwen_conflictmedqa_before_question_val
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target llama32_3b_conflictmedqa_before_question_val
```

兼容旧别名：

```bash
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target text_conflict_before_question_val
```

正式跑候选阈值的 ablation，并自动筛出最多 3 组“不强过当前配置”的备选：

```bash
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target hulumed_before_question_val --run-ablation --limit 3
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target internvl35_4b_text_conflict_before_question_val --run-ablation --limit 3
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target qwen_conflictmedqa_before_question_val --run-ablation --limit 3
python /root/logit_lens/ablation_tau/run_threshold_sweep.py --target llama32_3b_conflictmedqa_before_question_val --run-ablation --limit 3
```
