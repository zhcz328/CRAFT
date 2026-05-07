import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(json_path: Path, source: str):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get(source)
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"JSON 中未找到有效的 `{source}` 列表。")
    return rows


def plot_grouped_bar(rows, output_path: Path, title_suffix: str):
    head_labels = [f"L{item['layer']}_H{item['head']}" for item in rows]
    if "coarse_mean_abs_effect_reduction" in rows[0]:
        effect_key = "coarse_mean_abs_effect_reduction"
        base_key = "coarse_mean_abs_base_change"
    else:
        effect_key = "mean_abs_effect_reduction"
        base_key = "mean_abs_base_change"

    effect_reduction = [item[effect_key] for item in rows]
    # 按需求将 base_change 指标缩放为原值的 1/10 后再绘图
    base_change = [item[base_key] / 10 for item in rows]

    x = np.arange(len(head_labels))
    width = 0.38

    plt.figure(figsize=(20, 8))
    plt.bar(
        x - width / 2,
        effect_reduction,
        width=width,
        label="mean_abs_effect_reduction",
        color="#4E79A7",
    )
    plt.bar(
        x + width / 2,
        base_change,
        width=width,
        label="mean_abs_base_change (/10)",
        color="#F28E2B",
    )

    plt.xticks(x, head_labels, rotation=45, ha="right")
    plt.xlabel("Attention Heads")
    plt.ylabel("Metric Value")
    plt.title(f"Top 20 Heads ({title_suffix})")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    default_json = (
        "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
        "result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json"
    )
    default_output = (
        "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/"
        "pic/head_scan/top20_head_metrics_bar.png"
    )

    parser = argparse.ArgumentParser(description="绘制 coarse top20/rerank top20 双指标柱状图")
    parser.add_argument("--json_path", type=str, default=default_json, help="输入 JSON 路径")
    parser.add_argument(
        "--source",
        type=str,
        default="top20",
        choices=["top20", "coarse_top20", "coarse_rerank_top20"],
        help="选择绘图数据来源",
    )
    parser.add_argument("--output_path", type=str, default=default_output, help="输出图片路径")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    output_path = Path(args.output_path)

    rows = load_rows(json_path, args.source)
    plot_grouped_bar(rows, output_path, title_suffix=args.source)
    print(f"图像已保存到: {output_path}")


if __name__ == "__main__":
    main()
