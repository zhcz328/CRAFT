from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


@dataclass(frozen=True)
class ModelTraceSpec:
    label: str
    color: str
    position_to_trace: Dict[str, Path]


POSITION_ORDER = ["prefix", "before_question", "before_answer"]
POSITION_TITLE = {
    "prefix": "Prefix",
    "before_question": "Before Question",
    "before_answer": "Before Answer",
}

MODEL_SPECS: List[ModelTraceSpec] = [
    ModelTraceSpec(
        label="InternVL3.5-4B",
        color="#7aa6ff",
        position_to_trace={
            "prefix": REPO_ROOT / "Slake_vqa" / "text_conflict" / "internvl35_4b" / "result_prefix_slake" / "trace_conflict.json",
            "before_question": REPO_ROOT / "Slake_vqa" / "text_conflict" / "internvl35_4b" / "result_before_question_slake" / "trace_conflict.json",
            "before_answer": REPO_ROOT / "Slake_vqa" / "text_conflict" / "internvl35_4b" / "result_before_answer_slake" / "trace_conflict.json",
        },
    ),
    ModelTraceSpec(
        label="Qwen3.5-4B",
        color="#ff8f70",
        position_to_trace={
            "prefix": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen35_4b" / "result_prefix_slake" / "trace_conflict.json",
            "before_question": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen35_4b" / "result_before_question_slake" / "trace_conflict.json",
            "before_answer": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen35_4b" / "result_before_answer_slake" / "trace_conflict.json",
        },
    ),
    ModelTraceSpec(
        label="Qwen3-VL-4B",
        color="#7ed7a6",
        position_to_trace={
            "prefix": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen3vl_4b" / "result_prefix_slake" / "trace_conflict.json",
            "before_question": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen3vl_4b" / "result_before_question_slake" / "trace_conflict.json",
            "before_answer": REPO_ROOT / "Slake_vqa" / "text_conflict" / "qwen3vl_4b" / "result_before_answer_slake" / "trace_conflict.json",
        },
    ),
    ModelTraceSpec(
        label="Hulu-Med-4B",
        color="#d77cf6",
        position_to_trace={
            "prefix": REPO_ROOT / "Slake_vqa" / "Hulu-med" / "text_conflict" / "result_slake_hulumed4b_prefix" / "trace_conflict.json",
            "before_question": REPO_ROOT / "Slake_vqa" / "Hulu-med" / "text_conflict" / "result_slake_hulumed4b_before_question" / "trace_conflict.json",
            "before_answer": REPO_ROOT / "Slake_vqa" / "Hulu-med" / "text_conflict" / "result_slake_hulumed4b_before_answer" / "trace_conflict.json",
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SLAKE layer-trace curves for multiple models across prefix/before_question/before_answer."
    )
    parser.add_argument("--metric", default="follow_conflict", help="Metric name inside trace JSON layer_scores.")
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "slake_layer_trace_models_paper_style.svg"),
        help="Output SVG path.",
    )
    parser.add_argument("--linewidth", type=float, default=1.7, help="Curve line width.")
    parser.add_argument(
        "--y-label",
        default="Follow-conflict score",
        help="Display label for the y-axis.",
    )
    return parser.parse_args()


def load_metric(trace_path: Path, metric: str) -> List[float]:
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    with trace_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    layer_scores = payload.get("layer_scores")
    if not isinstance(layer_scores, dict):
        raise KeyError(f"'layer_scores' missing in {trace_path}")
    if metric not in layer_scores:
        available = ", ".join(sorted(layer_scores))
        raise KeyError(f"Metric '{metric}' not found in {trace_path}. Available: {available}")

    scores = layer_scores[metric]
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"Metric '{metric}' is empty in {trace_path}")
    return [float(v) for v in scores]


def load_all_series(metric: str) -> Dict[str, List[Tuple[ModelTraceSpec, List[float]]]]:
    all_series: Dict[str, List[Tuple[ModelTraceSpec, List[float]]]] = {}
    for position in POSITION_ORDER:
        all_series[position] = [(spec, load_metric(spec.position_to_trace[position], metric)) for spec in MODEL_SPECS]
    return all_series


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{value:.0f}"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def compute_range(values: List[float], pad_ratio: float = 0.08) -> Tuple[float, float]:
    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        pad = 1.0 if math.isclose(v_min, 0.0) else abs(v_min) * 0.1
        return v_min - pad, v_max + pad
    pad = (v_max - v_min) * pad_ratio
    return v_min - pad, v_max + pad


def scale_x(idx: int, x0: float, width: float, max_idx: int) -> float:
    if max_idx <= 0:
        return x0
    return x0 + (idx / max_idx) * width


def scale_y(value: float, y0: float, height: float, y_min: float, y_max: float) -> float:
    if math.isclose(y_min, y_max):
        return y0 + height / 2.0
    return y0 + height - ((value - y_min) / (y_max - y_min)) * height


def draw_curve(points: List[Tuple[float, float]], color: str, linewidth: float, opacity: float = 0.95) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-opacity="{opacity}" '
        f'stroke-width="{linewidth}" points="{point_text}" '
        'stroke-linejoin="round" stroke-linecap="round" />'
    )


def build_svg(
    all_series: Dict[str, List[Tuple[ModelTraceSpec, List[float]]]],
    metric_name: str,
    y_label: str,
    output_path: Path,
    linewidth: float,
) -> None:
    panel_width = 330
    panel_height = 272
    left_margin = 92
    right_margin = 44
    top_margin = 92
    bottom_margin = 62
    panel_gap = 62
    width = left_margin + right_margin + len(POSITION_ORDER) * panel_width + (len(POSITION_ORDER) - 1) * panel_gap
    height = top_margin + panel_height + bottom_margin

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style>',
        'text { font-family: Arial, Helvetica, sans-serif; fill: #222; }',
        '.panel-title { font-size: 18px; font-weight: 500; }',
        '.axis-label { font-size: 13px; }',
        '.tick { font-size: 11px; fill: #444; }',
        '.legend { font-size: 11px; }',
        '.small-note { font-size: 10px; fill: #444; }',
        '</style>',
    ]

    for panel_idx, position in enumerate(POSITION_ORDER):
        position_series = all_series[position]
        panel_values = [value for _, scores in position_series for value in scores]
        y_min, y_max = compute_range(panel_values)
        max_layer = max(len(scores) - 1 for _, scores in position_series)

        x0 = left_margin + panel_idx * (panel_width + panel_gap)
        y0 = top_margin
        x1 = x0 + panel_width
        y1 = y0 + panel_height

        svg.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 - 20}" text-anchor="middle" class="panel-title">{xml_escape(POSITION_TITLE[position])}</text>')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" fill="#f7f7f7" stroke="#333" stroke-width="1.2" />')

        y_tick_count = 5
        for tick_idx in range(y_tick_count + 1):
            value = y_min + (tick_idx / y_tick_count) * (y_max - y_min)
            y = scale_y(value, y0, panel_height, y_min, y_max)
            svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#dcdcdc" stroke-width="0.9" />')
            svg.append(f'<text x="{x0 - 10}" y="{y + 3.5:.2f}" text-anchor="end" class="tick">{xml_escape(format_tick(value))}</text>')

        x_ticks = [0, 5, 10, 15, 20, 25, 30, 35]
        x_ticks = [tick for tick in x_ticks if tick <= max_layer]
        if x_ticks and x_ticks[-1] != max_layer:
            x_ticks.append(max_layer)

        for tick in x_ticks:
            x = scale_x(tick, x0, panel_width, max_layer)
            svg.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="#ececec" stroke-width="0.9" />')
            svg.append(f'<text x="{x:.2f}" y="{y1 + 18}" text-anchor="middle" class="tick">{tick}</text>')

        legend_box_x = x0 + 10
        legend_box_y = y0 + 10
        legend_box_w = 118
        legend_box_h = 26
        svg.append(f'<rect x="{legend_box_x}" y="{legend_box_y}" width="{legend_box_w}" height="{legend_box_h}" fill="white" stroke="#999" stroke-width="0.8" />')
        svg.append(f'<line x1="{legend_box_x + 10}" y1="{legend_box_y + 13}" x2="{legend_box_x + 42}" y2="{legend_box_y + 13}" stroke="#666" stroke-width="1.4" />')
        svg.append(f'<text x="{legend_box_x + 52}" y="{legend_box_y + 17}" class="legend">Layer trace</text>')

        for spec, scores in position_series:
            points = [
                (scale_x(layer_idx, x0, panel_width, max_layer), scale_y(value, y0, panel_height, y_min, y_max))
                for layer_idx, value in enumerate(scores)
            ]
            svg.append(draw_curve(points, spec.color, linewidth))

        svg.append(f'<text x="{x0 + panel_width / 2:.1f}" y="{height - 18}" text-anchor="middle" class="axis-label">Layer</text>')
        if panel_idx == 0:
            svg.append(f'<text x="{x0 - 58}" y="{y0 + panel_height / 2:.1f}" text-anchor="middle" class="axis-label" transform="rotate(-90 {x0 - 58} {y0 + panel_height / 2:.1f})">{xml_escape(y_label)}</text>')

    legend_y = 30
    legend_x = left_margin + 6
    legend_step = 170
    for spec in MODEL_SPECS:
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{spec.color}" stroke-width="2.1" />')
        svg.append(f'<text x="{legend_x + 34}" y="{legend_y + 4}" class="legend">{xml_escape(spec.label)}</text>')
        legend_x += legend_step

    svg.append('</svg>')
    output_path.write_text('\n'.join(svg), encoding='utf-8')


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != '.svg':
        output_path = output_path.with_suffix('.svg')

    all_series = load_all_series(args.metric)
    build_svg(
        all_series=all_series,
        metric_name=args.metric,
        y_label=args.y_label,
        output_path=output_path,
        linewidth=args.linewidth,
    )
    print(f'Saved figure to: {output_path}')


if __name__ == '__main__':
    main()
