import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_top20(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    top20 = data.get("top20")
    if not isinstance(top20, list) or not top20:
        raise ValueError("JSON 中未找到有效的 top20 列表。")
    return top20


def plot_grouped_bar(top20, output_path: Path):
    head_labels = [f"L{item['layer']}_H{item['head']}" for item in top20]
    effect_reduction = [item["mean_abs_effect_reduction"] for item in top20]
    base_change = [item["mean_abs_base_change"] for item in top20]

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
        label="mean_abs_base_change",
        color="#F28E2B",
    )

    plt.xticks(x, head_labels, rotation=45, ha="right")
    plt.xlabel("Attention Heads")
    plt.ylabel("Metric Value")
    plt.title("Top 20 Heads: mean_abs_effect_reduction vs mean_abs_base_change")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    default_json = (
        "C:/Users/29818/Desktop/logit_lens/VQA_RAD/Hulu-med/"
        "result_train_hulumed4b_before_question/"
        "headscan_vqarad_mm_hulumed4b_before_question/"
        "head_scan_merged_unique_layers.json"
    )
    default_output = (
        "C:/Users/29818/Desktop/logit_lens/VQA_RAD/Hulu-med/"
        "pic/head_scan_metric/top20_head_metrics_bar.png"
    )

    parser = argparse.ArgumentParser(description="绘制 top20 头部双指标柱状图")
    parser.add_argument("--json_path", type=str, default=default_json, help="输入 JSON 路径")
    parser.add_argument("--output_path", type=str, default=default_output, help="输出图片路径")
    return parser.parse_args()


def main():
    args = parse_args()
    json_path = Path(args.json_path)
    output_path = Path(args.output_path)

    top20 = load_top20(json_path)
    plot_grouped_bar(top20, output_path)
    print(f"图像已保存到: {output_path}")


if __name__ == "__main__":
    main()
