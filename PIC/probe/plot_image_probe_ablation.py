#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path("/root/logit_lens")
OUT_DIR = ROOT / "PIC" / "probe" / "image_probe"
PROBE_ROOT = Path("/root/autodl-tmp/image_conflict/probe/hulumed4b")
PROJECT_ROOT = ROOT / "Slake_vqa" / "image_conflict"
MODEL_PATH = Path("/root/autodl-tmp/Hulu-Med-4B")
MANIFEST_PATH = PROBE_ROOT / "data" / "val_image_conflict.jsonl"
PROBE_A_SUMMARY = PROBE_ROOT / "results" / "image_conflict" / "conflict_linear" / "summary.json"
PROBE_A_DIR = PROBE_ROOT / "results" / "image_conflict" / "conflict_linear"
PROBE_B_SUMMARY = PROBE_ROOT / "results" / "image_conflict" / "follow_conflict_linear" / "summary.json"
PROBE_B_DIR = PROBE_ROOT / "results" / "image_conflict" / "follow_conflict_linear"
SELECTED_HEADS_PATH = (
    ROOT
    / "Slake_vqa"
    / "image_conflict"
    / "hulumed4b"
    / "result_image_conflict_slake"
    / "selected_heads_core_layers.json"
)
CACHE_PROBE_A = OUT_DIR / "hulumed4b_image_probe_a_ablation_auroc.json"
CACHE_PROBE_B = OUT_DIR / "hulumed4b_image_probe_b_ablation_macro_f1.json"
SERIES_MARKER_SIZE = 11.0
DRAMATIC_POINT_MARKER_SIZE = SERIES_MARKER_SIZE

SERIES_STYLES = {
    "probe_a": {
        "color": "#F2A900",
        "fill": "#FCE8B2",
        "marker": "o",
        "label": "Probe-A AUROC",
    },
    "probe_b": {
        "color": "#0E6EA8",
        "fill": "#D5E8F5",
        "marker": "s",
        "label": "Probe-B Macro-F1",
    },
    "probe_a_post": {
        "color": "#FF5A4F",
        "fill": "#F9D6D2",
        "marker": "D",
        "label": "Probe-A AUROC (post-ablation)",
    },
    "probe_b_post": {
        "color": "#663399",
        "fill": "#DDD3F5",
        "marker": "^",
        "label": "Probe-B Macro-F1 (post-ablation)",
    },
}


def load_project_module(unique_name: str, module_path: Path, extra_paths: list[Path]):
    old_sys_path = list(sys.path)
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    try:
        spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = old_sys_path


def load_modules():
    probe_root = PROJECT_ROOT / "probe"
    return {
        "score": load_project_module(
            "slake_image_probe_score",
            probe_root / "score_probe_on_manifest.py",
            [probe_root, PROJECT_ROOT],
        ),
        "train": load_project_module(
            "slake_image_probe_train",
            probe_root / "train_probe.py",
            [probe_root, PROJECT_ROOT],
        ),
        "ablate": load_project_module(
            "slake_image_ablate",
            PROJECT_ROOT / "ablate_head.py",
            [PROJECT_ROOT],
        ),
    }


def read_summary_curve(summary_path: Path, metric_name: str) -> list[tuple[int, float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [
        (int(item["layer_idx"]) + 1, float(item["val_metrics"][metric_name]))
        for item in payload["metrics_by_layer"]
    ]


def filter_probe_a_rows(read_jsonl, manifest_path: Path) -> list[dict]:
    rows = read_jsonl(str(manifest_path))
    return [row for row in rows if int(row.get("conflict_target", -1)) in (0, 1)]


def filter_probe_b_rows(read_jsonl, manifest_path: Path) -> list[dict]:
    rows = read_jsonl(str(manifest_path))
    return [
        row
        for row in rows
        if row.get("prompt_type") == "ic" and int(row.get("follow_target", -1)) in (0, 1)
    ]


def compute_metrics_from_logits(compute_metrics, logits_by_layer: dict[int, list[torch.Tensor]], labels: list[int]):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    curve = []
    per_layer_metrics = {}
    for layer_idx in sorted(logits_by_layer):
        layer_logits = torch.cat(logits_by_layer[layer_idx], dim=0).cpu()
        metrics = compute_metrics(layer_logits, labels_tensor)
        per_layer_metrics[layer_idx] = metrics
        curve.append((layer_idx + 1, metrics))
    return curve, per_layer_metrics


def load_mm_model_with_fallback(load_mm_model, model_path: Path, torch_dtype, device_map: str):
    try:
        if device_map == "auto":
            model = load_mm_model(str(model_path), device_map=None, torch_dtype=torch_dtype)
        else:
            model = load_mm_model(str(model_path), device_map=device_map, torch_dtype=torch_dtype)
    except ValueError as exc:
        if "requires `accelerate`" not in str(exc):
            raise
        model = load_mm_model(str(model_path), device_map=None, torch_dtype=torch_dtype)
    if torch.cuda.is_available():
        first_param = next(model.parameters())
        if first_param.device.type == "cpu":
            model = model.to("cuda:0")
    return model


def install_hulumed_compatible_hooks(ablate_module, model, layer_to_heads: dict[int, list[int]], keep_mode: str):
    layer2attn, _ = ablate_module._find_self_attn_modules(model)
    n_heads, _ = ablate_module._get_num_heads_and_hidden(model)
    if n_heads is None:
        raise RuntimeError("Missing num_attention_heads/n_head in HuluMed config.")

    handles = []
    for layer_idx, heads in layer_to_heads.items():
        if layer_idx not in layer2attn:
            continue
        heads_to_mask = sorted(set(int(head) for head in heads))

        def hook(module, args, kwargs, heads_to_mask=heads_to_mask):
            kwargs = {} if kwargs is None else kwargs
            attn_mask = None
            from_kwargs = False
            if "attention_mask" in kwargs:
                attn_mask = kwargs["attention_mask"]
                from_kwargs = True
            elif len(args) >= 2:
                attn_mask = args[1]

            if attn_mask is None or attn_mask.dim() != 4:
                return

            bsz = attn_mask.shape[0]
            q_len = attn_mask.shape[-2]
            k_len = attn_mask.shape[-1]
            device = attn_mask.device
            if torch.is_floating_point(attn_mask):
                work_dtype = attn_mask.dtype
            else:
                try:
                    work_dtype = next(module.parameters()).dtype
                except StopIteration:
                    work_dtype = torch.float32
            neg = torch.finfo(work_dtype).min

            if attn_mask.shape[1] == 1:
                expanded = attn_mask.to(work_dtype).expand(bsz, n_heads, q_len, k_len).clone()
            elif attn_mask.shape[1] == n_heads:
                expanded = attn_mask.to(work_dtype).clone()
            else:
                return

            for head_idx in heads_to_mask:
                if head_idx < 0 or head_idx >= n_heads:
                    continue
                expanded[:, head_idx, :, :] = neg
                if keep_mode == "self":
                    diag_len = min(q_len, k_len)
                    arange = torch.arange(diag_len, device=device)
                    expanded[:, head_idx, arange, arange] = 0.0
                elif keep_mode == "bos":
                    expanded[:, head_idx, :, 0] = 0.0
                else:
                    raise ValueError(f"Unsupported keep_mode: {keep_mode}")

            if from_kwargs:
                kwargs["attention_mask"] = expanded
                return args, kwargs

            args = list(args)
            if len(args) >= 2:
                args[1] = expanded
                return tuple(args), kwargs

        handles.append(layer2attn[layer_idx].register_forward_pre_hook(hook, with_kwargs=True))
    return handles


def compute_ablation_curve(
    *,
    cache_path: Path,
    probe_dir: Path,
    rows: list[dict],
    label_key: str,
    metric_key: str,
    modules: dict,
    batch_size: int = 2,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    image_size: int = 672,
) -> dict[str, object]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    probe_specs = modules["score"].load_probe_specs(probe_dir)
    layer_indices = sorted(probe_specs.keys())
    logits_by_layer = {layer_idx: [] for layer_idx in layer_indices}
    labels: list[int] = []

    modules["score"].ensure_video_import_compat()
    processor = modules["score"].load_processor_with_compat(str(MODEL_PATH))
    model = load_mm_model_with_fallback(
        modules["score"].load_mm_model,
        MODEL_PATH,
        modules["score"].resolve_dtype(dtype),
        device_map,
    )
    model.eval()
    selected_pairs = modules["ablate"].parse_selected_heads(str(SELECTED_HEADS_PATH))
    layer_to_heads = modules["ablate"].pack_layer_to_heads(selected_pairs)
    hook_handles = install_hulumed_compatible_hooks(modules["ablate"], model, layer_to_heads, keep_mode="self")

    try:
        first_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        for batch_rows in modules["score"].chunked(rows, batch_size):
            batch_inputs = modules["score"].build_batch_inputs(
                processor,
                batch_rows,
                PROJECT_ROOT,
                image_size,
            )
            batch_inputs.pop("token_type_ids", None)
            batch_inputs = modules["score"].move_to_device(batch_inputs, first_device, model_dtype)
            positions = batch_inputs["attention_mask"].sum(dim=1) - 1

            with torch.no_grad():
                out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

            hidden_states = out.hidden_states[1:]
            for layer_idx in layer_indices:
                hs_tensor = hidden_states[layer_idx]
                batch_index = torch.arange(len(batch_rows), device=hs_tensor.device)
                gathered = hs_tensor[batch_index, positions.to(hs_tensor.device)].cpu()
                probe_logits = modules["score"].apply_probe_batch(gathered, probe_specs[layer_idx], "logit")
                logits_by_layer[layer_idx].append(probe_logits)
            labels.extend(int(row[label_key]) for row in batch_rows)
    finally:
        for handle in hook_handles:
            handle.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    curve_with_metrics, per_layer_metrics = compute_metrics_from_logits(
        modules["train"].compute_metrics,
        logits_by_layer,
        labels,
    )
    payload_curve = [(layer, float(metrics[metric_key])) for layer, metrics in curve_with_metrics]
    payload = {
        "curve": payload_curve,
        "metric_key": metric_key,
        "label_key": label_key,
        "manifest": str(MANIFEST_PATH),
        "selected_heads": str(SELECTED_HEADS_PATH),
        "probe_dir": str(probe_dir),
        "model_path": str(MODEL_PATH),
        "n_samples": len(labels),
        "per_layer_metrics": per_layer_metrics,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def compute_y_limits(curves: list[list[tuple[int, float]]]) -> tuple[float, float]:
    values = [value for curve in curves for _, value in curve]
    min_value = min(values)
    max_value = max(values)
    lower = max(0.0, min_value - 0.02)
    upper = min(1.02, max_value + 0.015)
    return lower, upper


def style_axis(ax, max_layer: int, y_limits: tuple[float, float]) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(1, max_layer)
    ax.set_ylim(*y_limits)
    xticks = [tick for tick in (1, 5, 10, 15, 20, 25, 30, 35) if 1 <= tick <= max_layer]
    if max_layer not in xticks and all(abs(max_layer - tick) > 1 for tick in xticks):
        xticks.append(max_layer)
    ax.set_xticks(sorted(xticks))
    ax.tick_params(axis="both", labelbottom=True, labelleft=True, labelsize=18, length=0, width=1.2, colors="#3A424C")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, axis="y", color="#E8EDF2", linewidth=1.0, alpha=0.95)
    ax.grid(True, axis="x", color="#F2F5F8", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_color("#5E6975")


def draw_series(ax, x: list[int], y: list[float], style: dict[str, str], *, fill_floor: float, markevery):
    ax.fill_between(x, y, fill_floor, color=style["fill"], alpha=0.36, zorder=1)
    ax.plot(
        x,
        y,
        color=style["color"],
        linewidth=2.0,
        marker=style["marker"],
        markersize=SERIES_MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=2.0,
        markeredgecolor=style["color"],
        markevery=markevery,
        label=style["label"],
        zorder=3,
    )
    highlight_dramatic_point(ax, x, y, style)


def highlight_dramatic_point(ax, x: list[int], y: list[float], style: dict[str, str]) -> None:
    if len(x) < 2 or len(y) < 2:
        return

    deltas = [abs(curr - prev) for prev, curr in zip(y[:-1], y[1:])]
    point_index = max(range(1, len(y)), key=lambda idx: deltas[idx - 1])
    ax.plot(
        [x[point_index]],
        [y[point_index]],
        linestyle="None",
        marker=style["marker"],
        markersize=DRAMATIC_POINT_MARKER_SIZE,
        markerfacecolor="white",
        markeredgewidth=2.6,
        markeredgecolor=style["color"],
        zorder=4,
    )


def export_legend(handles, labels, out_prefix: Path) -> None:
    legend_fig, legend_ax = plt.subplots(figsize=(6.2, 1.25))
    legend_fig.patch.set_facecolor("#FFFFFF")
    legend_ax.axis("off")
    legend = legend_ax.legend(
        handles,
        labels,
        ncol=2,
        loc="center",
        frameon=True,
        fontsize=11,
        facecolor="#FFFFFF",
        edgecolor="#D7DEE7",
        fancybox=True,
        shadow=False,
        framealpha=1.0,
        columnspacing=1.6,
        handlelength=2.8,
    )
    legend.get_frame().set_linewidth(1.0)
    legend_fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.08)
    legend_fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(legend_fig)


def write_csv(curves: dict[str, list[tuple[int, float]]]) -> None:
    csv_path = OUT_DIR / "hulumed4b_image_probe_ablation.csv"
    rows = []
    for series_name, curve in curves.items():
        for layer, value in curve:
            rows.append({"series": series_name, "layer": layer, "value": value})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "layer", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_sources() -> None:
    lines = [
        "# Image Probe Ablation Sources",
        "",
        f"- Probe-A summary (before): `{PROBE_A_SUMMARY}`",
        f"- Probe-B summary (before): `{PROBE_B_SUMMARY}`",
        f"- Probe-A post-ablation cache: `{CACHE_PROBE_A}`",
        f"- Probe-B post-ablation cache: `{CACHE_PROBE_B}`",
        f"- Manifest: `{MANIFEST_PATH}`",
        f"- Selected heads: `{SELECTED_HEADS_PATH}`",
    ]
    (OUT_DIR / "data_sources.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )

    modules = load_modules()
    probe_a_before = read_summary_curve(PROBE_A_SUMMARY, "auroc")
    probe_b_before = read_summary_curve(PROBE_B_SUMMARY, "macro_f1")
    probe_a_rows = filter_probe_a_rows(modules["score"].read_jsonl, MANIFEST_PATH)
    probe_b_rows = filter_probe_b_rows(modules["score"].read_jsonl, MANIFEST_PATH)
    probe_a_after_payload = compute_ablation_curve(
        cache_path=CACHE_PROBE_A,
        probe_dir=PROBE_A_DIR,
        rows=probe_a_rows,
        label_key="conflict_target",
        metric_key="auroc",
        modules=modules,
    )
    probe_b_after_payload = compute_ablation_curve(
        cache_path=CACHE_PROBE_B,
        probe_dir=PROBE_B_DIR,
        rows=probe_b_rows,
        label_key="follow_target",
        metric_key="macro_f1",
        modules=modules,
    )

    curves = {
        "probe_a_before": probe_a_before,
        "probe_b_before": probe_b_before,
        "probe_a_after": [(int(layer), float(value)) for layer, value in probe_a_after_payload["curve"]],
        "probe_b_after": [(int(layer), float(value)) for layer, value in probe_b_after_payload["curve"]],
    }

    max_layer = max(layer for curve in curves.values() for layer, _ in curve)
    y_limits = compute_y_limits(list(curves.values()))

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)
    marker_stride = max(1, len(curves["probe_a_before"]) // 8)

    draw_series(
        ax,
        [layer for layer, _ in curves["probe_a_before"]],
        [value for _, value in curves["probe_a_before"]],
        SERIES_STYLES["probe_a"],
        fill_floor=y_limits[0],
        markevery=(0, marker_stride),
    )
    draw_series(
        ax,
        [layer for layer, _ in curves["probe_b_before"]],
        [value for _, value in curves["probe_b_before"]],
        SERIES_STYLES["probe_b"],
        fill_floor=y_limits[0],
        markevery=(1, marker_stride),
    )
    draw_series(
        ax,
        [layer for layer, _ in curves["probe_a_after"]],
        [value for _, value in curves["probe_a_after"]],
        SERIES_STYLES["probe_a_post"],
        fill_floor=y_limits[0],
        markevery=(2, marker_stride),
    )
    draw_series(
        ax,
        [layer for layer, _ in curves["probe_b_after"]],
        [value for _, value in curves["probe_b_after"]],
        SERIES_STYLES["probe_b_post"],
        fill_floor=y_limits[0],
        markevery=(3, marker_stride),
    )

    style_axis(ax, max_layer, y_limits)
    handles, labels = ax.get_legend_handles_labels()
    export_legend(handles, labels, OUT_DIR / "hulumed4b_image_probe_ablation_legend")

    out_prefix = OUT_DIR / "hulumed4b_image_probe_ablation"
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_csv(curves)
    write_sources()


if __name__ == "__main__":
    main()
