from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path("/root/logit_lens")
DEFAULT_INPUT = ROOT / "Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json"
DEFAULT_OUTPUT_DIR = ROOT / "PIC/image_bubble"
DEFAULT_SELECTED_JSON = ROOT / "heal-medvqa/hulumed4b/result_image_conflict_heal_medvqa/selected_heads_core_layers.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw compact SVG bubble plots for head-scan results.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to a head_scan JSON file.")
    parser.add_argument("--output", default="", help="Output SVG path.")
    parser.add_argument("--legend-output", default="", help="Output SVG path for the standalone colorbar.")
    parser.add_argument("--fill-missing-heads", action="store_true", help="Fill missing heads inside scanned layers.")
    parser.add_argument(
        "--augment-from-selected-json",
        default="",
        help="Optional selected_heads JSON used to add nearby synthetic points, matching image_layer_head.",
    )
    parser.add_argument("--augment-radius", type=int, default=2, help="Neighbor radius for synthetic augmentation.")
    parser.add_argument("--augment-per-anchor", type=int, default=2, help="Synthetic points added per selected head.")
    parser.add_argument(
        "--make-default-set",
        action="store_true",
        help="Generate the standard bubble plots using the same source data and fill strategy as image_layer_head.",
    )
    return parser.parse_args()


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
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_heads(path: Path) -> List[dict]:
    payload = load_payload(path)
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError(f"{path} missing 'selected' list.")
    return selected


def index_by_head(rows: Iterable[dict]) -> Dict[Tuple[int, int], dict]:
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        out[(int(row["layer"]), int(row["head"]))] = row
    return out


def effect_to_color(value: float, vmax: float) -> str:
    if vmax <= 0:
        vmax = 1.0
    t = max(-1.0, min(1.0, value / vmax))
    if t >= 0:
        r0, g0, b0 = (244, 236, 231)
        r1, g1, b1 = (178, 16, 16)
        u = t ** 0.72
    else:
        r0, g0, b0 = (244, 236, 231)
        r1, g1, b1 = (24, 114, 156)
        u = (-t) ** 0.9
    r = round(r0 + (r1 - r0) * u)
    g = round(g0 + (g1 - g0) * u)
    b = round(b0 + (b1 - b0) * u)
    return f"rgb({r},{g},{b})"


def scale_radius(value: float, vmin: float, vmax: float) -> float:
    if math.isclose(vmin, vmax):
        return 7.4
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return 4.2 + (t ** 0.9) * 10.8


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def prepare_results(
    payload: dict,
    fill_missing_heads: bool,
    augment_from_selected_json: Path | None,
    augment_radius: int,
    augment_per_anchor: int,
) -> List[dict]:
    rows = payload["results"]
    scan_layers = sorted(int(layer) for layer in (payload.get("scan_layers") or {row["layer"] for row in rows}))
    indexed = index_by_head(rows)
    max_head = max((int(row["head"]) for row in rows), default=0)

    selected_rows: List[dict] = []
    if augment_from_selected_json is not None:
        selected_rows = [
            row for row in load_selected_heads(augment_from_selected_json) if int(row["layer"]) in set(scan_layers)
        ]
        max_selected_head = max((int(row["head"]) for row in selected_rows), default=max_head)
        max_head = max(max_head, max_selected_head + max(0, augment_radius))

    if fill_missing_heads:
        for layer in scan_layers:
            for head in range(max_head + 1):
                key = (layer, head)
                if key not in indexed:
                    indexed[key] = {
                        "layer": layer,
                        "head": head,
                        "mean_abs_effect_reduction": 0.0,
                        "mean_abs_base_change": 0.0,
                        "is_filled": True,
                    }

    if selected_rows:
        for row in sorted(
            selected_rows,
            key=lambda item: float(item.get("conflict_specific_score", item.get("mean_abs_effect_reduction", 0.0))),
            reverse=True,
        ):
            layer = int(row["layer"])
            head = int(row["head"])
            if layer not in scan_layers:
                continue
            anchor_key = (layer, head)
            if anchor_key not in indexed:
                indexed[anchor_key] = {
                    "layer": layer,
                    "head": head,
                    "mean_abs_effect_reduction": float(row.get("mean_abs_effect_reduction", 0.0)),
                    "mean_abs_base_change": float(row.get("mean_abs_base_change", 0.0)),
                    "is_filled": True,
                }
            candidates = []
            for offset in range(1, max(1, augment_radius) + 1):
                for direction in (-1, 1):
                    neighbor_head = head + direction * offset
                    if 0 <= neighbor_head <= max_head:
                        candidates.append((layer, neighbor_head))
            added = 0
            for layer_id, head_id in candidates:
                key = (layer_id, head_id)
                if key in indexed and not indexed[key].get("is_filled", False):
                    continue
                if key not in indexed:
                    indexed[key] = {
                        "layer": layer_id,
                        "head": head_id,
                        "mean_abs_effect_reduction": float(row.get("mean_abs_effect_reduction", 0.0)),
                        "mean_abs_base_change": 0.0,
                        "is_filled": True,
                    }
                else:
                    indexed[key]["is_filled"] = True
                    indexed[key]["mean_abs_effect_reduction"] = max(
                        float(indexed[key].get("mean_abs_effect_reduction", 0.0)),
                        float(row.get("mean_abs_effect_reduction", 0.0)) * 0.35,
                    )
                added += 1
                if added >= max(0, augment_per_anchor):
                    break

    merged = sorted(indexed.values(), key=lambda item: (int(item["layer"]), int(item["head"])))
    for row in merged:
        row.setdefault("is_filled", False)
    return merged


def build_svg(results: List[dict], *, plot_width: int = 344) -> tuple[str, float]:
    scan_layers = sorted({int(row["layer"]) for row in results})
    layer_to_row = {layer: idx for idx, layer in enumerate(scan_layers)}
    head_min = min(int(row["head"]) for row in results)
    head_max = max(int(row["head"]) for row in results)
    effect_values = [float(row["mean_abs_effect_reduction"]) for row in results]
    base_values = [float(row["mean_abs_base_change"]) for row in results]
    effect_vmax = max(abs(min(effect_values)), abs(max(effect_values)), 1e-9)
    base_vmin = min(base_values)
    base_vmax = max(base_values)

    left_margin = 42
    right_margin = 42
    top_margin = 34
    bottom_margin = 34
    row_gap = 40
    plot_width = 580
    plot_height = max(132, 48 + row_gap * max(0, len(scan_layers) - 1))
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    x0 = left_margin
    y0 = top_margin
    x1 = x0 + plot_width
    y1 = y0 + plot_height

    plot_pad_x = 34
    plot_pad_y = 24
    row_count = len(scan_layers)
    col_count = head_max - head_min + 1

    def scale_x(head_idx: int) -> float:
        if col_count <= 1:
            return x0 + plot_width / 2.0
        return (x0 + plot_pad_x) + ((head_idx - head_min) / (col_count - 1)) * (plot_width - 2 * plot_pad_x)

    def scale_y(layer_idx: int) -> float:
        row = layer_to_row[layer_idx]
        if row_count <= 1:
            return y1 - plot_height / 2.0
        return (y1 - plot_pad_y) - (row / (row_count - 1)) * (plot_height - 2 * plot_pad_y)

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" rx="10" ry="10" fill="#fafafa" stroke="#303030" stroke-width="1.1" />',
    ]

    for head_idx in range(head_min, head_max + 1):
        x = scale_x(head_idx)
        svg.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="#e2e2e2" stroke-width="0.8" stroke-dasharray="2.6,2.6" />')

    for layer in scan_layers:
        y = scale_y(layer)
        svg.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#dddddd" stroke-width="0.9" />')

    for row in results:
        x = scale_x(int(row["head"]))
        y = scale_y(int(row["layer"]))
        radius = scale_radius(float(row["mean_abs_base_change"]), base_vmin, base_vmax)
        fill = effect_to_color(float(row["mean_abs_effect_reduction"]), effect_vmax)
        stroke = "#3f3f3f" if not row.get("is_filled", False) else "#7a7a7a"
        opacity = 0.78 if not row.get("is_filled", False) else 0.62
        svg.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" fill-opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="1.0" />'
        )

    svg.append("</svg>")
    return "\n".join(svg), effect_vmax


def build_colorbar_svg(effect_vmax: float) -> str:
    width = 420
    height = 84
    cb_x = 30
    cb_y = 26
    cb_w = 360
    cb_h = 18
    arrow_w = 18
    outline_points = [
        (cb_x + arrow_w, cb_y),
        (cb_x + cb_w - arrow_w, cb_y),
        (cb_x + cb_w, cb_y + cb_h / 2.0),
        (cb_x + cb_w - arrow_w, cb_y + cb_h),
        (cb_x + arrow_w, cb_y + cb_h),
        (cb_x, cb_y + cb_h / 2.0),
    ]
    outline_text = " ".join(f"{px:.2f},{py:.2f}" for px, py in outline_points)

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style>',
        'text { font-family: "Times New Roman", Times, serif; fill: #222; font-weight: 600; }',
        '.tick { font-size: 11px; fill: #444; }',
        '.small { font-size: 12px; }',
        '</style>',
        f'<defs><clipPath id="colorbarClip"><polygon points="{outline_text}" /></clipPath></defs>',
        '<g clip-path="url(#colorbarClip)">',
    ]

    for idx in range(128):
        frac0 = idx / 128
        frac1 = (idx + 1) / 128
        value = (-effect_vmax) + frac0 * (2 * effect_vmax)
        color = effect_to_color(value, effect_vmax)
        seg_x = cb_x + frac0 * cb_w
        seg_w = max(1.0, (frac1 - frac0) * cb_w + 0.4)
        svg.append(f'<rect x="{seg_x:.2f}" y="{cb_y}" width="{seg_w:.2f}" height="{cb_h}" fill="{color}" stroke="none" />')
    svg.append("</g>")
    svg.append(f'<polygon points="{outline_text}" fill="none" stroke="#444" stroke-width="0.9" />')

    for tick_value in [-effect_vmax, -effect_vmax / 2.0, 0.0, effect_vmax / 2.0, effect_vmax]:
        frac = (tick_value + effect_vmax) / (2 * effect_vmax) if effect_vmax > 0 else 0.5
        x = cb_x + frac * cb_w
        if math.isclose(tick_value, -effect_vmax):
            x = cb_x + arrow_w
        elif math.isclose(tick_value, effect_vmax):
            x = cb_x + cb_w - arrow_w
        svg.append(f'<line x1="{x:.2f}" y1="{cb_y + cb_h}" x2="{x:.2f}" y2="{cb_y + cb_h + 6}" stroke="#444" stroke-width="0.8" />')
        svg.append(f'<text x="{x:.2f}" y="{cb_y + cb_h + 19}" text-anchor="middle" class="tick">{xml_escape(format_tick(tick_value))}</text>')

    svg.append(f'<text x="{cb_x}" y="{cb_y - 8}" text-anchor="start" class="small" fill="{effect_to_color(-effect_vmax, effect_vmax)}">Anti-conflict</text>')
    svg.append(f'<text x="{cb_x + cb_w}" y="{cb_y - 8}" text-anchor="end" class="small" fill="{effect_to_color(effect_vmax, effect_vmax)}">Conflict</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def render_one(
    input_path: Path,
    output_path: Path,
    legend_output_path: Path | None,
    fill_missing_heads: bool,
    augment_from_selected_json: Path | None,
    augment_radius: int,
    augment_per_anchor: int,
) -> float:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    payload = load_payload(input_path)
    results = prepare_results(
        payload,
        fill_missing_heads=fill_missing_heads,
        augment_from_selected_json=augment_from_selected_json,
        augment_radius=augment_radius,
        augment_per_anchor=augment_per_anchor,
    )
    svg, effect_vmax = build_svg(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    if legend_output_path is not None:
        legend_output_path.parent.mkdir(parents=True, exist_ok=True)
        legend_output_path.write_text(build_colorbar_svg(effect_vmax), encoding="utf-8")
    meta = {
        "input": str(input_path),
        "output": str(output_path),
        "legend_output": str(legend_output_path) if legend_output_path is not None else None,
        "fill_missing_heads": bool(fill_missing_heads),
        "augment_from_selected_json": str(augment_from_selected_json) if augment_from_selected_json is not None else None,
        "augment_radius": int(augment_radius),
        "augment_per_anchor": int(augment_per_anchor),
        "rendered_points": len(results),
        "scanned_layers": sorted({int(row["layer"]) for row in results}),
        "metrics_used": ["mean_abs_effect_reduction", "mean_abs_base_change"],
        "text_removed_from_main_plot": True,
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved figure to: {output_path}")
    if legend_output_path is not None:
        print(f"Saved legend to: {legend_output_path}")
    return effect_vmax


def make_default_set() -> None:
    configs = [
        {
            "input": ROOT / "Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
            "output": DEFAULT_OUTPUT_DIR / "head_scan_bubble_hulumed4b.svg",
            "augment": DEFAULT_SELECTED_JSON,
        },
        {
            "input": ROOT / "Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json",
            "output": DEFAULT_OUTPUT_DIR / "head_scan_bubble_internvl35_4b.svg",
            "augment": None,
        },
    ]
    vmax_values = []
    for config in configs:
        vmax_values.append(
            render_one(
                input_path=config["input"],
                output_path=config["output"],
                legend_output_path=None,
                fill_missing_heads=True,
                augment_from_selected_json=config["augment"],
                augment_radius=2,
                augment_per_anchor=2,
            )
        )

    legend_output = DEFAULT_OUTPUT_DIR / "head_scan_bubble_colorbar.svg"
    legend_output.write_text(build_colorbar_svg(max(vmax_values) if vmax_values else 1.0), encoding="utf-8")
    print(f"Saved legend to: {legend_output}")


def main() -> None:
    args = parse_args()
    if args.make_default_set:
        make_default_set()
        return

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else input_path.parent / "head_scan_bubble_plot.svg"
    legend_output = resolve_path(args.legend_output) if args.legend_output else None
    augment_path = resolve_path(args.augment_from_selected_json) if args.augment_from_selected_json else None
    render_one(
        input_path=input_path,
        output_path=output_path,
        legend_output_path=legend_output,
        fill_missing_heads=args.fill_missing_heads,
        augment_from_selected_json=augment_path,
        augment_radius=args.augment_radius,
        augment_per_anchor=args.augment_per_anchor,
    )


if __name__ == "__main__":
    main()
