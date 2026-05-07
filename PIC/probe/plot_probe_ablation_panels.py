#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_DIR = Path("/root/logit_lens/PIC/probe")
OUT_DIR = DATA_DIR / "text_probe"
SERIES_MARKER_SIZE = 11.0
DRAMATIC_POINT_MARKER_SIZE = SERIES_MARKER_SIZE
LOCAL_HELPER_MODULES = ("common", "extract_features", "model_utils", "resume_utils")

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


@dataclass(frozen=True)
class ModelConfig:
    slug: str
    display_name: str
    family: str
    model_kind: str
    project_root: Path
    probe_a_summary: Path
    probe_a_dir: Path
    probe_b_summary: Path
    probe_b_dir: Path
    manifest_path: Path
    selected_heads_path: Path
    model_path: Path


MODEL_CONFIGS = [
    ModelConfig(
        slug="qwen3_4b",
        display_name="Qwen3-4B",
        family="conflictmedqa",
        model_kind="text",
        project_root=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp"),
        probe_a_summary=Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/conflict_linear_before_question/summary.json"
        ),
        probe_a_dir=Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/conflict_linear_before_question"
        ),
        probe_b_summary=Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/follow_linear_before_question/summary.json"
        ),
        probe_b_dir=Path(
            "/root/autodl-tmp/probe/conflictmedqa/qwen3-4b/"
            "before_question/results/follow_linear_before_question"
        ),
        manifest_path=Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/probe/data/"
            "val_pair_stratified_before_question.jsonl"
        ),
        selected_heads_path=Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/result_train/before_question/"
            "headscan_rounds_top50_inf/head_groups.json"
        ),
        model_path=Path("/root/autodl-tmp/qwen3-4B"),
    ),
    ModelConfig(
        slug="llama32_3b",
        display_name="Llama3.2-3B",
        family="conflictmedqa",
        model_kind="text",
        project_root=Path("/root/logit_lens/conflictmedqa/Qwen3-4B_exp"),
        probe_a_summary=Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/conflict_linear/summary.json"
        ),
        probe_a_dir=Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/conflict_linear"
        ),
        probe_b_summary=Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/follow_linear/summary.json"
        ),
        probe_b_dir=Path(
            "/root/autodl-tmp/probe/conflictmedqa/llama3.2-3b/"
            "before_question/follow_linear"
        ),
        manifest_path=Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/probe/data/"
            "val_pair_stratified_before_question.jsonl"
        ),
        selected_heads_path=Path(
            "/root/logit_lens/conflictmedqa/Qwen3-4B_exp/llama32_3b/result/"
            "before_question/headscan_rounds_top30_inf/selected_heads.json"
        ),
        model_path=Path("/root/autodl-tmp/Llama-3.2-3B-Instruct"),
    ),
    ModelConfig(
        slug="internvl35_4b",
        display_name="InternVL3.5-4B",
        family="vqa_rad",
        model_kind="multimodal",
        project_root=Path("/root/logit_lens/VQA_RAD/text_conflict"),
        probe_a_summary=Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/conflict_linear/summary.json"
        ),
        probe_a_dir=Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/conflict_linear"
        ),
        probe_b_summary=Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/follow_conflict_linear/summary.json"
        ),
        probe_b_dir=Path(
            "/root/autodl-tmp/probe/internvl35_4b/results/"
            "before_question/follow_conflict_linear"
        ),
        manifest_path=Path("/root/autodl-tmp/probe/internvl35_4b/data/val_before_question.jsonl"),
        selected_heads_path=Path(
            "/root/logit_lens/VQA_RAD/text_conflict/internvl35_4b/"
            "result_before_question_vqarad/selected_heads_merged_unique_layers.json"
        ),
        model_path=Path("/root/autodl-tmp/InternVL3_5-4B"),
    ),
    ModelConfig(
        slug="hulumed4b",
        display_name="HuluMed-4B",
        family="vqa_rad",
        model_kind="multimodal",
        project_root=Path("/root/logit_lens/VQA_RAD/Hulu-med"),
        probe_a_summary=Path(
            "/root/autodl-tmp/Hulumed/probe_vqa_rad/results/"
            "conflict_linear_before_question/summary.json"
        ),
        probe_a_dir=Path(
            "/root/autodl-tmp/Hulumed/probe_vqa_rad/results/"
            "conflict_linear_before_question"
        ),
        probe_b_summary=Path(
            "/root/autodl-tmp/Hulumed/probe_vqa_rad/results/"
            "follow_linear_before_question/summary.json"
        ),
        probe_b_dir=Path(
            "/root/autodl-tmp/Hulumed/probe_vqa_rad/results/"
            "follow_linear_before_question"
        ),
        manifest_path=Path("/root/logit_lens/VQA_RAD/Hulu-med/probe/data/val_before_question.jsonl"),
        selected_heads_path=Path(
            "/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
            "headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json"
        ),
        model_path=Path("/root/autodl-tmp/Hulu-Med-4B"),
    ),
]


def load_project_module(unique_name: str, module_path: Path, extra_paths: list[Path]):
    old_sys_path = list(sys.path)
    for helper_name in LOCAL_HELPER_MODULES:
        sys.modules.pop(helper_name, None)
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


def load_text_modules(project_root: Path):
    probe_root = project_root / "probe"
    return {
        "common": load_project_module(
            f"{project_root.name}_text_common",
            probe_root / "common.py",
            [probe_root],
        ),
        "score": load_project_module(
            f"{project_root.name}_text_probe_score",
            probe_root / "score_probe_on_manifest.py",
            [probe_root],
        ),
        "train": load_project_module(
            f"{project_root.name}_text_probe_train",
            probe_root / "train_probe.py",
            [probe_root],
        ),
        "ablate": load_project_module(
            f"{project_root.name}_text_ablate",
            project_root / "ablate_head_inf.py",
            [project_root],
        ),
    }


def load_mm_modules(project_root: Path):
    probe_root = project_root / "probe"
    return {
        "score": load_project_module(
            f"{project_root.name}_mm_probe_score",
            probe_root / "score_probe_on_manifest.py",
            [probe_root, project_root],
        ),
        "train": load_project_module(
            f"{project_root.name}_mm_probe_train",
            probe_root / "train_probe.py",
            [probe_root, project_root],
        ),
        "ablate": load_project_module(
            f"{project_root.name}_mm_ablate",
            project_root / "ablate_head.py",
            [project_root],
        ),
    }


def read_summary_curve(summary_path: Path, metric_name: str) -> list[tuple[int, float]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [
        (int(item["layer_idx"]) + 1, float(item["val_metrics"][metric_name]))
        for item in payload["metrics_by_layer"]
    ]


def filter_follow_rows(read_jsonl, manifest_path: Path) -> list[dict]:
    rows = read_jsonl(str(manifest_path))
    return [
        row
        for row in rows
        if row.get("prompt_type") == "conflict" and int(row.get("follow_target", -1)) in (0, 1)
    ]


def compute_metrics_from_logits(compute_metrics, logits_by_layer: dict[int, list[torch.Tensor]], labels: list[int]):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    curve = []
    per_layer_metrics = {}
    for layer_idx in sorted(logits_by_layer):
        layer_logits = torch.cat(logits_by_layer[layer_idx], dim=0).cpu()
        metrics = compute_metrics(layer_logits, labels_tensor)
        per_layer_metrics[layer_idx] = metrics
        curve.append((layer_idx + 1, float(metrics["macro_f1"])))
    return curve, per_layer_metrics


def filter_probe_rows(read_jsonl, manifest_path: Path) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(str(manifest_path))
    probe_a_rows = [row for row in rows if int(row.get("conflict_target", -1)) in (0, 1)]
    probe_b_rows = [
        row
        for row in rows
        if row.get("prompt_type") == "conflict" and int(row.get("follow_target", -1)) in (0, 1)
    ]
    return probe_a_rows, probe_b_rows


def filter_probe_a_rows(read_jsonl, manifest_path: Path) -> list[dict]:
    rows = read_jsonl(str(manifest_path))
    return [row for row in rows if int(row.get("conflict_target", -1)) in (0, 1)]


def load_text_model_with_fallback(model_path: Path, torch_dtype, device_map: str):
    try:
        return AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
    except ValueError as exc:
        if "requires `accelerate`" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        if torch.cuda.is_available():
            model = model.to("cuda:0")
        return model


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


def evaluate_text_probe_a_ablation(
    config: ModelConfig,
    batch_size: int,
    dtype: str,
    device_map: str,
) -> dict[str, object]:
    modules = load_text_modules(config.project_root)
    rows = filter_probe_a_rows(modules["common"].read_jsonl, config.manifest_path)
    probe_a_specs = modules["score"].load_probe_specs(config.probe_a_dir)
    layer_indices = sorted(probe_a_specs.keys())
    probe_a_logits_by_layer = {layer_idx: [] for layer_idx in layer_indices}
    probe_a_labels: list[int] = []

    tokenizer = modules["common"].ensure_padding_token(
        AutoTokenizer.from_pretrained(str(config.model_path), trust_remote_code=True)
    )
    model = load_text_model_with_fallback(
        config.model_path,
        modules["score"].resolve_dtype(dtype),
        device_map,
    )
    model.eval()
    layer_to_heads, _ = modules["ablate"].load_ablation_heads(str(config.selected_heads_path))
    if config.slug == "hulumed4b":
        hook_handles = install_hulumed_compatible_hooks(
            modules["ablate"],
            model,
            layer_to_heads,
            keep_mode="self",
        )
    else:
        hook_handles = modules["ablate"].install_head_mask_hooks(model, layer_to_heads, keep_mode="self")

    try:
        first_device = next(model.parameters()).device
        for batch_rows in modules["score"].chunked(rows, batch_size):
            prompts = [
                modules["common"].wrap_as_chat(tokenizer, row["prompt_text"], enable_thinking=False)
                for row in batch_rows
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True)
            enc = {k: v.to(first_device) for k, v in enc.items()}
            positions = enc["attention_mask"].sum(dim=1) - 1

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)

            hidden_states = out.hidden_states[1:]
            batch_index = torch.arange(len(batch_rows), device=first_device)
            for layer_idx in layer_indices:
                hs_tensor = hidden_states[layer_idx]
                gathered = hs_tensor[batch_index, positions.to(hs_tensor.device)].cpu()
                probe_a_logits = modules["score"].apply_probe_batch(gathered, probe_a_specs[layer_idx], "logit")
                probe_a_logits_by_layer[layer_idx].append(probe_a_logits)
            probe_a_labels.extend(int(row["conflict_target"]) for row in batch_rows)
    finally:
        modules["ablate"].remove_hooks(hook_handles)
        del model
        torch.cuda.empty_cache()

    probe_a_curve, probe_a_metrics = compute_metrics_from_logits(
        modules["train"].compute_metrics,
        probe_a_logits_by_layer,
        probe_a_labels,
    )
    return {
        "probe_a_curve": probe_a_curve,
        "probe_a_per_layer_metrics": probe_a_metrics,
        "probe_a_n_samples": len(probe_a_labels),
        "manifest": str(config.manifest_path),
        "selected_heads": str(config.selected_heads_path),
        "probe_a_dir": str(config.probe_a_dir),
        "model_path": str(config.model_path),
    }


def evaluate_mm_probe_a_ablation(
    config: ModelConfig,
    batch_size: int,
    dtype: str,
    device_map: str,
    image_size: int,
) -> dict[str, object]:
    modules = load_mm_modules(config.project_root)
    rows = filter_probe_a_rows(modules["score"].read_jsonl, config.manifest_path)
    probe_a_specs = modules["score"].load_probe_specs(config.probe_a_dir)
    layer_indices = sorted(probe_a_specs.keys())
    probe_a_logits_by_layer = {layer_idx: [] for layer_idx in layer_indices}
    probe_a_labels: list[int] = []

    modules["score"].ensure_video_import_compat()
    if hasattr(modules["score"], "load_processor_with_compat"):
        processor = modules["score"].load_processor_with_compat(str(config.model_path))
    else:
        processor = modules["score"].AutoProcessor.from_pretrained(str(config.model_path), trust_remote_code=True)
    model = load_mm_model_with_fallback(
        modules["score"].load_mm_model,
        config.model_path,
        modules["score"].resolve_dtype(dtype),
        device_map,
    )
    model.eval()
    selected_pairs = modules["ablate"].parse_selected_heads(str(config.selected_heads_path))
    layer_to_heads = modules["ablate"].pack_layer_to_heads(selected_pairs)
    if config.slug == "hulumed4b":
        hook_handles = install_hulumed_compatible_hooks(
            modules["ablate"],
            model,
            layer_to_heads,
            keep_mode="self",
        )
    else:
        hook_handles = modules["ablate"].install_head_mask_hooks(model, layer_to_heads, keep_mode="self")

    try:
        first_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        for batch_rows in modules["score"].chunked(rows, batch_size):
            batch_inputs = modules["score"].build_batch_inputs(
                processor,
                batch_rows,
                config.project_root,
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
                probe_a_logits = modules["score"].apply_probe_batch(gathered, probe_a_specs[layer_idx], "logit")
                probe_a_logits_by_layer[layer_idx].append(probe_a_logits)
            probe_a_labels.extend(int(row["conflict_target"]) for row in batch_rows)
    finally:
        for handle in hook_handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()

    probe_a_curve, probe_a_metrics = compute_metrics_from_logits(
        modules["train"].compute_metrics,
        probe_a_logits_by_layer,
        probe_a_labels,
    )
    return {
        "probe_a_curve": probe_a_curve,
        "probe_a_per_layer_metrics": probe_a_metrics,
        "probe_a_n_samples": len(probe_a_labels),
        "manifest": str(config.manifest_path),
        "selected_heads": str(config.selected_heads_path),
        "probe_a_dir": str(config.probe_a_dir),
        "model_path": str(config.model_path),
    }


def load_or_compute_probe_a_ablation_curve(
    config: ModelConfig,
    batch_size_text: int,
    batch_size_mm: int,
    dtype: str,
    device_map: str,
    image_size: int,
    force: bool,
) -> dict[str, object]:
    cache_path = DATA_DIR / f"{config.slug}_probe_a_ablation_auroc.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    legacy_combined_cache = DATA_DIR / f"{config.slug}_probe_ablation_curves.json"
    if legacy_combined_cache.exists() and not force:
        legacy_payload = json.loads(legacy_combined_cache.read_text(encoding="utf-8"))
        payload = {
            "probe_a_curve": legacy_payload["probe_a_curve"],
            "probe_a_per_layer_metrics": legacy_payload.get("probe_a_per_layer_metrics", {}),
            "probe_a_n_samples": legacy_payload.get("probe_a_n_samples"),
            "manifest": legacy_payload.get("manifest", str(config.manifest_path)),
            "selected_heads": legacy_payload.get("selected_heads", str(config.selected_heads_path)),
            "probe_a_dir": legacy_payload.get("probe_a_dir", str(config.probe_a_dir)),
            "model_path": legacy_payload.get("model_path", str(config.model_path)),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    if config.model_kind == "text":
        payload = evaluate_text_probe_a_ablation(config, batch_size_text, dtype, device_map)
    else:
        payload = evaluate_mm_probe_a_ablation(config, batch_size_mm, dtype, device_map, image_size)

    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_probe_b_ablation_curve(config: ModelConfig) -> dict[str, object]:
    legacy_cache = DATA_DIR / f"{config.slug}_probe_b_ablation_macro_f1.json"
    if not legacy_cache.exists():
        raise FileNotFoundError(
            f"Missing legacy Probe-B ablation cache for {config.display_name}: {legacy_cache}"
        )
    return json.loads(legacy_cache.read_text(encoding="utf-8"))


def compute_y_limits(curves: list[list[tuple[int, float]]]) -> tuple[float, float]:
    values = [value for curve in curves for _, value in curve]
    min_value = min(values)
    max_value = max(values)
    lower = max(0.35, min_value - 0.04)
    upper = min(1.02, max_value + 0.03)
    return lower, upper


def style_axis(ax, max_layer: int, ylabel: str, show_baseline: bool, y_limits: tuple[float, float]) -> None:
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
    if show_baseline:
        ax.axhline(0.5, color="#B9C0C8", linestyle=(0, (4, 3)), linewidth=1.2, zorder=0)
    ax.grid(True, axis="y", color="#E8EDF2", linewidth=1.0, alpha=0.95)
    ax.grid(True, axis="x", color="#F2F5F8", linewidth=0.8, alpha=0.8)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_color("#5E6975")


def draw_series(
    ax,
    x: list[int],
    y: list[float],
    style: dict[str, str],
    *,
    fill_floor: float,
    linestyle: str = "-",
    markevery: int | tuple[int, int] | None = None,
) -> None:
    ax.fill_between(x, y, fill_floor, color=style["fill"], alpha=0.36, zorder=1)
    ax.plot(
        x,
        y,
        color=style["color"],
        linewidth=2.0,
        linestyle=linestyle,
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


def export_legend_figure(config: ModelConfig, handles, labels) -> None:
    legend_fig, legend_ax = plt.subplots(figsize=(5.8, 1.25))
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
    legend_prefix = OUT_DIR / f"{config.slug}_probe_ablation_legend"
    legend_fig.savefig(legend_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.08)
    legend_fig.savefig(legend_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(legend_fig)


def plot_model_panel(
    config: ModelConfig,
    probe_a_curve: list[tuple[int, float]],
    probe_b_curve: list[tuple[int, float]],
    ablated_probe_a_curve: list[tuple[int, float]],
    ablated_probe_b_curve: list[tuple[int, float]],
) -> None:
    max_layer = max(max(layer for layer, _ in probe_a_curve), max(layer for layer, _ in probe_b_curve))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.96)

    x_a = [layer for layer, _ in probe_a_curve]
    y_a = [value for _, value in probe_a_curve]
    x_b = [layer for layer, _ in probe_b_curve]
    y_b = [value for _, value in probe_b_curve]
    x_aa = [layer for layer, _ in ablated_probe_a_curve]
    y_aa = [value for _, value in ablated_probe_a_curve]
    x_ab = [layer for layer, _ in ablated_probe_b_curve]
    y_ab = [value for _, value in ablated_probe_b_curve]
    y_limits = compute_y_limits(
        [probe_a_curve, probe_b_curve, ablated_probe_a_curve, ablated_probe_b_curve]
    )

    marker_stride = max(1, len(x_a) // 8)
    draw_series(ax, x_a, y_a, SERIES_STYLES["probe_a"], fill_floor=y_limits[0], markevery=(0, marker_stride))
    draw_series(ax, x_b, y_b, SERIES_STYLES["probe_b"], fill_floor=y_limits[0], markevery=(1, marker_stride))
    draw_series(
        ax,
        x_aa,
        y_aa,
        SERIES_STYLES["probe_a_post"],
        fill_floor=y_limits[0],
        markevery=(2, marker_stride),
    )
    draw_series(
        ax,
        x_ab,
        y_ab,
        SERIES_STYLES["probe_b_post"],
        fill_floor=y_limits[0],
        markevery=(3, marker_stride),
    )
    style_axis(ax, max_layer, "Probe-A AUROC / Probe-B Macro-F1", show_baseline=False, y_limits=y_limits)
    handles, labels = ax.get_legend_handles_labels()
    export_legend_figure(config, handles, labels)

    out_prefix = OUT_DIR / f"{config.slug}_probe_ablation_panel"
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_model_csv(
    config: ModelConfig,
    probe_a_curve: list[tuple[int, float]],
    probe_b_curve: list[tuple[int, float]],
    ablated_probe_a_curve: list[tuple[int, float]],
    ablated_probe_b_curve: list[tuple[int, float]],
) -> None:
    rows = []
    for series_name, source_path, curve in (
        ("probe_a_auroc", config.probe_a_summary, probe_a_curve),
        ("probe_b_macro_f1", config.probe_b_summary, probe_b_curve),
        ("probe_a_auroc_post_ablation", DATA_DIR / f"{config.slug}_probe_a_ablation_auroc.json", ablated_probe_a_curve),
        ("probe_b_macro_f1_post_ablation", DATA_DIR / f"{config.slug}_probe_b_ablation_macro_f1.json", ablated_probe_b_curve),
    ):
        for layer, value in curve:
            rows.append(
                {
                    "model": config.display_name,
                    "series": series_name,
                    "layer": layer,
                    "value": value,
                    "source": str(source_path),
                }
            )
    csv_path = OUT_DIR / f"{config.slug}_probe_ablation_panel.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "series", "layer", "value", "source"])
        writer.writeheader()
        writer.writerows(rows)


def write_sources_md(source_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Probe Ablation Panel Sources",
        "",
        "Each panel in this directory uses:",
        "",
        "- Main plot: Probe-A validation AUROC and Probe-B validation Macro-F1 from the saved `summary.json` files, plus their post-ablation curves recomputed after installing the model-specific ablation hooks.",
        "- Legend: exported separately as its own figure for layout flexibility.",
        "",
        "For HuluMed, the solid curves intentionally use the `probe_vqa_rad` artifacts so the original and post-ablation curves are aligned to the same VQA-RAD ablation setup.",
        "",
        "| Model | Probe-A summary | Probe-B summary | Probe-B val manifest | Selected heads | Ablated Probe-B cache |",
        "|---|---|---|---|---|---|",
    ]
    for row in source_rows:
        lines.append(
        f"| {row['model']} | `{row['probe_a_summary']}` | `{row['probe_b_summary']}` | "
            f"`{row['manifest']}` | `{row['selected_heads']}` | `{row['ablation_cache']}` |"
        )
    (OUT_DIR / "data_sources.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot four probe-ablation panels with reused probe and ablation logic.")
    parser.add_argument("--force_recompute", action="store_true", help="Recompute the post-ablation Probe-B curves even if cached JSON exists.")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated subset of model slugs to run. Available: "
        + ",".join(config.slug for config in MODEL_CONFIGS),
    )
    parser.add_argument("--batch_size_text", type=int, default=4)
    parser.add_argument("--batch_size_mm", type=int, default=2)
    parser.add_argument("--dtype_text", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dtype_mm", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--image_size", type=int, default=672)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_rows = []
    requested = {item.strip() for item in args.models.split(",") if item.strip()}
    selected_configs = [config for config in MODEL_CONFIGS if not requested or config.slug in requested]
    if requested and not selected_configs:
        raise ValueError(f"No matching models found for --models={args.models!r}")

    for config in selected_configs:
        probe_a_curve = read_summary_curve(config.probe_a_summary, "auroc")
        probe_b_curve = read_summary_curve(config.probe_b_summary, "macro_f1")
        probe_a_ablation_payload = load_or_compute_probe_a_ablation_curve(
            config=config,
            batch_size_text=args.batch_size_text,
            batch_size_mm=args.batch_size_mm,
            dtype=args.dtype_text if config.model_kind == "text" else args.dtype_mm,
            device_map=args.device_map,
            image_size=args.image_size,
            force=args.force_recompute,
        )
        probe_b_ablation_payload = load_probe_b_ablation_curve(config)
        ablated_probe_a_curve = [
            (int(layer), float(value))
            for layer, value in probe_a_ablation_payload["probe_a_curve"]
        ]
        ablated_probe_b_curve = [
            (int(layer), float(value))
            for layer, value in probe_b_ablation_payload["curve"]
        ]
        plot_model_panel(config, probe_a_curve, probe_b_curve, ablated_probe_a_curve, ablated_probe_b_curve)
        write_model_csv(config, probe_a_curve, probe_b_curve, ablated_probe_a_curve, ablated_probe_b_curve)
        source_rows.append(
            {
                "model": config.display_name,
                "probe_a_summary": str(config.probe_a_summary),
                "probe_b_summary": str(config.probe_b_summary),
                "manifest": str(config.manifest_path),
                "selected_heads": str(config.selected_heads_path),
                "ablation_cache": (
                    f"{DATA_DIR / f'{config.slug}_probe_a_ablation_auroc.json'} ; "
                    f"{DATA_DIR / f'{config.slug}_probe_b_ablation_macro_f1.json'}"
                ),
            }
        )

    write_sources_md(source_rows)


if __name__ == "__main__":
    main()
