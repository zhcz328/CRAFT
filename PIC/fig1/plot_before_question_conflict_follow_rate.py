#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path('/root/logit_lens/PIC/fig1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# model_name, summary_path, extraction_mode
SOURCES = [
    (
        'Qwen3-4B',
        Path('/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_all_positions/summary.json'),
        'summary_by_position',
    ),
    (
        'InternVL3.5-4B',
        Path('/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/eval-results_vqarad_all/_root_autodl-tmp_InternVL3_5-4B/before_question/summary.json'),
        'direct_before_question',
    ),
    (
        'Hulu-Med-4B',
        Path('/root/logit_lens/VQA_RAD/Hulu-med/eval-results_vqarad_hulumed4b_all/_archive_zengjiaqi_Medical_LLM_Hulu-Med-4B/before_question/summary.json'),
        'direct_before_question',
    ),
    (
        'Llama3.2-3B',
        Path('/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result_all_positions/summary_all.json'),
        'summary_by_position',
    ),
]


def extract_rate_percent(payload: dict, mode: str) -> float:
    if mode == 'summary_by_position':
        # conflictmedqa: decimal in [0,1]
        val = payload['summary_by_position']['before_question']['follow_conflict_rate']
        return float(val) * 100.0

    # vqa_rad style: percent in [0,100]
    for key in (
        'conflict_follow_evidence_rate',
        'follow_conflict_rate_given_sup_correct',
        'follow_conflict_rate_given_nc_correct',
    ):
        if key in payload:
            return float(payload[key])

    raise KeyError('Cannot find conflict follow-rate field in summary json.')


def main() -> None:
    labels = []
    values = []

    for name, path, mode in SOURCES:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        labels.append(name)
        values.append(extract_rate_percent(payload, mode))

    # Save extracted values
    values_json = OUTPUT_DIR / 'before_question_conflict_follow_rates.json'
    with values_json.open('w', encoding='utf-8') as f:
        json.dump({label: round(val, 4) for label, val in zip(labels, values)}, f, ensure_ascii=False, indent=2)

    # Plot
    plt.figure(figsize=(10, 6), dpi=150)
    bars = plt.bar(labels, values, color='#b10000', edgecolor='#8a0000', width=0.58)

    plt.title('Model Performance Under Conflict', fontsize=18)
    plt.ylabel('Conflict Follow-Rate (%)', fontsize=16)
    plt.ylim(0, max(100.0, max(values) + 10))

    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        plt.text(x, y + 1.0, f'{v:.1f}%', ha='center', va='bottom', fontsize=15)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Style similar to sample
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()

    out_png = OUTPUT_DIR / 'fig1_before_question_conflict_follow_rate.png'
    out_pdf = OUTPUT_DIR / 'fig1_before_question_conflict_follow_rate.pdf'
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print('Saved:', out_png)
    print('Saved:', out_pdf)
    print('Values JSON:', values_json)


if __name__ == '__main__':
    main()
