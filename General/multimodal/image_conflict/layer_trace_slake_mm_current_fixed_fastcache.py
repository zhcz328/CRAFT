import os
import json
import argparse
import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import Any
from typing import Any as TypingAny

from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir
from analysis_image_conflict import (
    SYSTEM_PROMPT,
    build_ic_prompt as shared_build_ic_prompt,
    build_nc_prompt as shared_build_nc_prompt,
    get_answer_candidates_from_row,
    load_nc_ic_images,
    normalize_position,
    normalize_trace_mode,
    score_map_to_label,
)


def load_mm_model(model_name, device_map=None, torch_dtype=None, trust_remote_code=True, attn_implementation=None, quantization_config=None):
    return shared_load_mm_model(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )


def ensure_video_import_compat() -> None:
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = TypingAny
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = TypingAny


def has_non_video_image_inputs(batch) -> bool:
    for k, v in batch.items():
        key = str(k).lower()
        if "video" in key:
            continue
        if ("image" in key) or ("pixel_values" in key):
            if torch.is_tensor(v):
                if v.numel() > 0:
                    return True
            elif isinstance(v, (list, tuple)):
                if len(v) > 0:
                    return True
            elif v is not None:
                return True
    return False


def move_to_device(obj, device, float_dtype=None):
    if torch.is_tensor(obj):
        if float_dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=float_dtype)
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device, float_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device, float_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device, float_dtype) for v in obj)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except Exception:
            return obj
    return obj


def infer_vision_input_dtype(model):
    m = model.module if hasattr(model, "module") else model
    try:
        core = m.get_model() if hasattr(m, "get_model") else m
        if hasattr(core, "get_vision_encoder"):
            vision = core.get_vision_encoder()
            for p in vision.parameters():
                if p.is_floating_point():
                    return p.dtype
    except Exception:
        pass
    for p in m.parameters():
        if p.is_floating_point():
            return p.dtype
    return None


def build_processor_inputs_mm(processor, text: str, image: Image.Image):
    attempts = [
        {"text": text, "images": image},
        {"text": text, "images": [image]},
        {"text": [text], "images": image},
        {"text": [text], "images": [image]},
    ]
    last_err = None
    for kw in attempts:
        try:
            out = processor(**kw, padding=True, return_tensors="pt")
            if has_non_video_image_inputs(out):
                return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Failed to build multimodal processor inputs with explicit image injection. "
        f"Last error: {repr(last_err)}"
    )


def _find_plateau_start(
    scores: List[float],
    plateau_min_score_frac: float,
    plateau_eps: float,
    plateau_min_len: int,
) -> Optional[int]:
    if not scores:
        return None
    max_score = max(scores)
    if max_score <= 0:
        return None

    thr = plateau_min_score_frac * max_score
    n = len(scores)
    for s in range(0, n - plateau_min_len + 1):
        if scores[s] < thr:
            continue
        ok = True
        for i in range(s, s + plateau_min_len - 1):
            if abs(scores[i + 1] - scores[i]) > plateau_eps:
                ok = False
                break
        if ok:
            return s
    return None


def build_scan_plan(
    scores: List[float],
    tail_k: int = 4,
    preplateau_width: int = 7,
    plateau_min_score_frac: float = 0.8,
    plateau_eps: float = 0.6,
    plateau_min_len: int = 6,
    topk_fallback: int = 8,
    rise_k: int = 6,
    rise_min_layer: int = 0,
    rise_max_layer: int = -1,
):
    n_layers = len(scores)
    if n_layers == 0:
        return {
            "method": "empty_scores",
            "plateau_start": None,
            "round0_rise": [],
            "round1_preplateau": [],
            "round1_topk_fallback": [],
            "round2_tail": [],
            "positive_layers": [],
            "core_layers": [],
            "neighbor_layers": [],
            "merged_unique_layers": [],
            "tail_k": tail_k,
            "preplateau_width": preplateau_width,
            "plateau_min_score_frac": plateau_min_score_frac,
            "plateau_eps": plateau_eps,
            "plateau_min_len": plateau_min_len,
            "topk_fallback": topk_fallback,
            "rise_k": rise_k,
            "rise_min_layer": rise_min_layer,
            "rise_max_layer": rise_max_layer,
        }

    positive_layers = [i for i, s in enumerate(scores) if s > 0]
    ranked_layers = sorted(range(n_layers), key=lambda i: scores[i], reverse=True)

    if positive_layers:
        core_ranked = [i for i in ranked_layers if scores[i] > 0]
        core_layers = sorted(core_ranked[: min(max(1, topk_fallback), len(core_ranked))])
        method = "positive_topk"
    else:
        core_layers = sorted(ranked_layers[: min(max(1, topk_fallback), n_layers)])
        method = "topk_nonpositive_fallback"

    neighbor_radius = 1
    neighbor_layers = sorted(
        {
            j
            for i in core_layers
            for j in range(max(0, i - neighbor_radius), min(n_layers, i + neighbor_radius + 1))
            if j not in core_layers
        }
    )
    merged_layers = sorted(set(core_layers) | set(neighbor_layers))

    return {
        "method": method,
        "plateau_start": None,
        "round0_rise": [],
        "round1_preplateau": [],
        "round1_topk_fallback": core_layers,
        "round2_tail": neighbor_layers,
        "positive_layers": sorted(positive_layers),
        "core_layers": core_layers,
        "neighbor_layers": neighbor_layers,
        "merged_unique_layers": merged_layers,
        "tail_k": tail_k,
        "preplateau_width": preplateau_width,
        "plateau_min_score_frac": plateau_min_score_frac,
        "plateau_eps": plateau_eps,
        "plateau_min_len": plateau_min_len,
        "topk_fallback": topk_fallback,
        "rise_k": rise_k,
        "rise_min_layer": rise_min_layer,
        "rise_max_layer": rise_max_layer,
    }


def parse_weights(weights_str: str, n: int) -> List[float]:
    if not weights_str:
        return [1.0 / n] * n
    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    if len(parts) != n:
        raise SystemExit(f"--scan_plan_weights expects {n} comma-separated weights, got {len(parts)}")
    w = [float(x) for x in parts]
    s = sum(w)
    if s == 0:
        raise SystemExit("--scan_plan_weights sum to 0")
    return [x / s for x in w]


def combine_scores(layer_score_mean: Dict[str, List[float]], metrics: List[str], weights: List[float]) -> List[float]:
    n_layers = len(next(iter(layer_score_mean.values())))
    combined = [0.0] * n_layers
    for m, w in zip(metrics, weights):
        s = layer_score_mean[m]
        if len(s) != n_layers:
            raise RuntimeError("Metric score length mismatch")
        for i in range(n_layers):
            combined[i] += w * s[i]
    return combined
def norm_text(x: str) -> str:
    return str(x).strip().lower()


def resize_image_max_side(image: Image.Image, max_side: int = 672) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def resolve_image_path(row, image_root: str) -> Optional[Path]:
    root = Path(image_root)
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        raw = str(row["image_path"]).strip()
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        candidates.append(p)
        candidates.append(root / p)
        candidates.append(root / p.name)

    if "img_id" in row and pd.notna(row["img_id"]):
        candidates.append(root / str(row["img_id"]).strip())

    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    return None


def _get_layers(model):
    m = model
    if hasattr(m, "module"):
        m = m.module

    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("model", "h"),
    ]

    for path in candidates:
        cur = m
        ok = True
        for part in path:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return cur

    tried = [".".join(path) for path in candidates]
    raise AttributeError(
        f"Cannot locate decoder layers from model type: {type(m)}. Tried paths: {tried}"
    )


def load_model_and_processor(model_name: str, device: str, dtype: str):
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    ensure_video_import_compat()
    processor = load_processor_with_compat(model_name)
    model = load_mm_model(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def build_inputs_mm(processor, prompt: str, image: Image.Image, device):
    msgs = [
        {
        "role": "system",
        "content": SYSTEM_PROMPT,
            },
        {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    base_inputs = build_processor_inputs_mm(processor, text, image)
    if not has_non_video_image_inputs(base_inputs):
        keys = sorted(list(base_inputs.keys()))
        raise RuntimeError(f"No image-related inputs found. keys={keys}")
    base_inputs.pop("token_type_ids", None)
    base_inputs = move_to_device(base_inputs, device)
    return base_inputs


@torch.inference_mode()
def build_prompt_cache_from_inputs(model, base_inputs, need_hidden_states: bool = False):
    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=True, output_hidden_states=need_hidden_states)

    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()

    ret = {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
    }
    if need_hidden_states:
        ret["hidden_states"] = out.hidden_states
    return ret


@torch.inference_mode()
def logprob_from_cache_mm(model, processor, cache_pack, continuation: str) -> float:
    tok = processor.tokenizer
    cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    if cont_ids.numel() == 0:
        return float("-inf")

    attn_prompt = cache_pack["attn_prompt"]
    past_key_values = cache_pack["past_key_values"]
    prompt_last_logits = cache_pack["prompt_last_logits"]

    cont_len = int(cont_ids.shape[1])

    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)
    total = float(log_probs0[0, int(cont_ids[0, 0])])

    if cont_len == 1:
        return total

    inp = cont_ids[:, :-1]
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)

    out = model(
        input_ids=inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
    )
    log_probs = torch.log_softmax(out.logits, dim=-1)

    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total += float(log_probs[0, i - 1, tok_id])

    return total


@torch.inference_mode()
def score_answer_candidates_from_cache(model, processor, cache_pack, gold: str, wrong: str, unknown: str):
    score_map = {}
    for option in [gold, wrong, unknown]:
        option = str(option).strip().lower()
        if not option or option in score_map:
            continue
        score_map[option] = logprob_from_cache_mm(model, processor, cache_pack, " " + option)

    pred_answer, pred_label = score_map_to_label(score_map, gold, wrong, unknown)
    gold_lp = float(score_map.get(gold, float("-inf")))
    wrong_lp = float(score_map.get(wrong, float("-inf")))
    unknown_lp = float(score_map.get(unknown, float("-inf")))
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "unknown_lp": unknown_lp,
        "candidate_scores": score_map,
        "pred": pred_answer,
        "pred_label": pred_label,
    }


class HiddenPatcher:
    def __init__(self, model, layer_idx: int, source_tail: torch.Tensor, patch_k: int):
        self.model = model
        self.layer_idx = layer_idx
        self.source_tail = source_tail
        self.patch_k = patch_k
        self.handle = None

    def __enter__(self):
        layer = _get_layers(self.model)[self.layer_idx]

        def hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                rest = out[1:]
            else:
                hs = out
                rest = None
            k = min(self.patch_k, hs.shape[1], self.source_tail.shape[1])
            if k <= 0:
                return out
            hs = hs.clone()
            hs[:, -k:, :] = self.source_tail[:, -k:, :].to(hs.device, hs.dtype)
            if rest is None:
                return hs
            return (hs,) + rest

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


class LayerDeltaScaler:
    def __init__(self, model, layer_idx: int, scale: float):
        self.model = model
        self.layer_idx = layer_idx
        self.scale = float(scale)
        self.handle = None

    def __enter__(self):
        layer = _get_layers(self.model)[self.layer_idx]

        def hook(module, inp, out):
            if not inp:
                return out
            hs_in = inp[0]
            if not torch.is_tensor(hs_in):
                return out

            if isinstance(out, tuple):
                hs_out = out[0]
                rest = out[1:]
                out_kind = "tuple"
            elif isinstance(out, list):
                hs_out = out[0]
                rest = out[1:]
                out_kind = "list"
            else:
                hs_out = out
                rest = None
                out_kind = "tensor"

            if not torch.is_tensor(hs_out):
                return out

            hs_in = hs_in.to(device=hs_out.device, dtype=hs_out.dtype)
            hs_scaled = hs_in + self.scale * (hs_out - hs_in)

            if out_kind == "tensor":
                return hs_scaled
            if out_kind == "tuple":
                return (hs_scaled,) + tuple(rest)
            return [hs_scaled] + list(rest)

        self.handle = layer.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()


@torch.inference_mode()
def build_layer_scaled_prompt_cache_from_inputs(model, base_inputs, layer_idx: int, layer_scale: float):
    with LayerDeltaScaler(model, layer_idx, layer_scale):
        return build_prompt_cache_from_inputs(model, base_inputs, need_hidden_states=False)

def metric_from_scores(metric: str, scores: dict, gold: str, wrong: str, trace_mode: str) -> float:
    if metric == "gold_wrong_margin":
        return float(scores["gold_lp"] - scores["wrong_lp"])

    if metric == "follow_conflict":
        return float(scores["unknown_lp"] - scores["gold_lp"])

    ctx_lp = float(scores["gold_lp"] if trace_mode == "cc" else scores["unknown_lp"])
    other_lp = float(max(scores["wrong_lp"], scores["unknown_lp"])) if trace_mode == "cc" else float(max(scores["gold_lp"], scores["wrong_lp"]))

    if metric == "follow_context":
        return ctx_lp - other_lp

    if metric == "context_flip":
        return other_lp - ctx_lp

    raise ValueError(
        f"Unknown metric: {metric}. Allowed: ['gold_wrong_margin', 'follow_conflict', 'follow_context', 'context_flip']"
    )


def nc_gold_margin(scores: dict) -> float:
    return float(scores["gold_lp"] - max(scores["wrong_lp"], scores["unknown_lp"]))


def is_hallucination_sample(nc_scores: dict, ctx_scores: dict) -> bool:
    return str(nc_scores.get("pred_label")) == "gold" and str(ctx_scores.get("pred_label")) != "unknown"


def plot_scores(layer_scores: Dict[str, List[float]], out_prefix: str):
    parent = Path(out_prefix).parent
    os.makedirs(parent if str(parent) else ".", exist_ok=True)
    for metric, vals in layer_scores.items():
        plt.figure(figsize=(8, 4))
        xs = list(range(len(vals)))
        plt.plot(xs, vals, marker='o')
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.title(f"Layer trace: {metric}")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{metric}.png", dpi=200)
        plt.close()


def resume_key_for_trace_row(
    idx: int,
    img_path: str,
    question: str,
    gold: str,
    wrong: str,
    position: str,
    trace_mode: str,
    sample_filter: str,
) -> str:
    return f"{idx}\t{img_path}\t{question}\t{gold}\t{wrong}\t{position}\t{trace_mode}\t{sample_filter}"


def accumulate_trace_record(layer_deltas: Dict[str, List[float]], rec: Dict[str, Any], metrics: List[str]) -> None:
    gold = rec["gold"]
    wrong = rec["wrong"]
    trace_mode = rec["trace_mode"]
    base_ctx_scores = rec["ctx"]
    base_nc_scores = rec["nc"]
    base_nc_margin = nc_gold_margin(base_nc_scores)
    for layer_idx_str, patched in rec["layers"].items():
        layer_idx = int(layer_idx_str)
        if "ctx" not in patched or "nc" not in patched:
            continue
        ctx_scores = patched["ctx"]
        nc_scores = patched["nc"]
        ctx_follow_context_gain = (
            metric_from_scores("follow_context", ctx_scores, gold, wrong, trace_mode)
            - metric_from_scores("follow_context", base_ctx_scores, gold, wrong, trace_mode)
        )
        nc_margin_damage = max(0.0, base_nc_margin - nc_gold_margin(nc_scores))
        if "follow_context" not in metrics:
            layer_deltas["follow_context"][layer_idx] += float(ctx_follow_context_gain)
        layer_deltas["nc_gold_margin_damage"][layer_idx] += float(nc_margin_damage)
        layer_deltas["hallucination_relief"][layer_idx] += float(ctx_follow_context_gain - nc_margin_damage)
        for m in metrics:
            layer_deltas[m][layer_idx] += float(
                metric_from_scores(m, ctx_scores, gold, wrong, trace_mode)
                - metric_from_scores(m, base_ctx_scores, gold, wrong, trace_mode)
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["conflict"])
    ap.add_argument("--position", type=str, default="image_conflict")
    ap.add_argument("--layer_scale", type=float, default=0.0)
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--max_image_side", type=int, default=672)
    ap.add_argument("--sample_filter", type=str, default="none", choices=["none", "hallucination"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", type=str, default="follow_context")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot_prefix", type=str, default="")
    ap.add_argument("--scan_plan_out", type=str, default="")
    ap.add_argument("--scan_plan_metric", type=str, default="")
    ap.add_argument("--scan_plan_metrics", type=str, default="")
    ap.add_argument("--scan_plan_weights", type=str, default="")

    ap.add_argument("--tail_k", type=int, default=4)
    ap.add_argument("--preplateau_width", type=int, default=13)
    ap.add_argument("--plateau_min_score_frac", type=float, default=0.65)
    ap.add_argument("--plateau_eps", type=float, default=0.6)
    ap.add_argument("--plateau_min_len", type=int, default=6)
    ap.add_argument("--topk_fallback", type=int, default=8)
    ap.add_argument("--rise_k", type=int, default=6)
    ap.add_argument("--rise_min_layer", type=int, default=0)
    ap.add_argument("--rise_max_layer", type=int, default=35)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out = prefix_model_relative_path(args.out, model_name=args.model_name, model=args.model)
    args.position = normalize_position(args.position)
    args.trace_mode = normalize_trace_mode(args.trace_mode)
    if args.mask_scale <= 0:
        raise SystemExit("--mask_scale must be positive")
    if args.plot_prefix:
        args.plot_prefix = prefix_model_relative_path(args.plot_prefix, model_name=args.model_name, model=args.model)
    if args.scan_plan_out:
        args.scan_plan_out = prefix_model_relative_path(args.scan_plan_out, model_name=args.model_name, model=args.model)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    allowed_metrics = {"gold_wrong_margin", "follow_conflict", "follow_context", "context_flip"}
    for m in metrics:
        if m not in allowed_metrics:
            raise SystemExit(f"Unknown metric '{m}'. Allowed: {sorted(allowed_metrics)}")

    df = pd.read_csv(args.data_csv)

    if "gold" not in df.columns:
        raise ValueError("CSV must contain a 'gold' column.")
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")

    model, processor = load_model_and_processor(args.model, args.device, args.dtype)
    layers = _get_layers(model)
    n_layers = len(layers)

    layer_deltas = {m: [0.0 for _ in range(n_layers)] for m in metrics}
    for extra_metric in ("follow_context", "nc_gold_margin_damage", "hallucination_relief"):
        layer_deltas.setdefault(extra_metric, [0.0 for _ in range(n_layers)])
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=Path(__file__).resolve().parent,
            task_name="layer_trace",
            model_name=args.model_name,
            model=args.model,
            position=args.position,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="layer_trace",
        data_csv=args.data_csv,
        out=args.out,
        position=args.position,
        trace_mode=args.trace_mode,
        sample_filter=args.sample_filter,
    )
    records = tracker.read_records("records.jsonl")
    for rec in records:
        accumulate_trace_record(layer_deltas, rec, metrics)
    used = len(records)
    skipped = int(tracker.state.get("skipped", 0))

    rows = df.to_dict("records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    for row_idx, row in tqdm(enumerate(rows), total=len(rows), desc="Tracing"):
        gold, wrong, unknown, _answer_candidates = get_answer_candidates_from_row(row)
        question = str(row["question"])
        row_key = resume_key_for_trace_row(
            row_idx,
            str(row.get("image_path", "")),
            question,
            gold,
            wrong,
            args.position,
            args.trace_mode,
            args.sample_filter,
        )
        if tracker.is_done(row_key):
            continue
        if not gold or not wrong or gold == wrong:
            skipped += 1
            tracker.mark_done(row_key, {"status": "skip_invalid"})
            tracker.update(skipped=skipped, used=used)
            continue

        try:
            img_path, det_path, mask_path, target_labels, nc_image, ic_image = load_nc_ic_images(
                row,
                args.image_root,
                max_side=args.max_image_side,
                mask_scale=args.mask_scale,
            )
        except Exception:
            skipped += 1
            tracker.mark_done(row_key, {"status": "missing_image"})
            tracker.update(skipped=skipped, used=used)
            continue

        nc_prompt = shared_build_nc_prompt(question)
        ctx_prompt = shared_build_ic_prompt(question)

        nc_inputs = build_inputs_mm(processor, nc_prompt, nc_image, model.device)
        ctx_inputs = build_inputs_mm(processor, ctx_prompt, ic_image, model.device)

        nc_cache = build_prompt_cache_from_inputs(model, nc_inputs, need_hidden_states=False)
        ctx_cache = build_prompt_cache_from_inputs(model, ctx_inputs, need_hidden_states=False)

        nc_scores = score_answer_candidates_from_cache(model, processor, nc_cache, gold, wrong, unknown)
        ctx_scores = score_answer_candidates_from_cache(model, processor, ctx_cache, gold, wrong, unknown)
        if args.sample_filter == "hallucination" and not is_hallucination_sample(nc_scores, ctx_scores):
            skipped += 1
            tracker.mark_done(row_key, {"status": "skip_not_hallucination"})
            tracker.update(skipped=skipped, used=used)
            continue

        rec = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "unknown": unknown,
            "image_path": str(img_path),
            "detection_path": str(det_path),
            "mask_path": str(mask_path),
            "ic_target_labels": target_labels,
            "trace_mode": args.trace_mode,
            "mask_scale": float(args.mask_scale),
            "nc": dict(nc_scores),
            "ctx": dict(ctx_scores),
            "layers": {},
        }

        for layer_idx in range(n_layers):
            nc_layer_cache = build_layer_scaled_prompt_cache_from_inputs(
                model, nc_inputs, layer_idx, args.layer_scale
            )
            ctx_layer_cache = build_layer_scaled_prompt_cache_from_inputs(
                model, ctx_inputs, layer_idx, args.layer_scale
            )
            rec["layers"][str(layer_idx)] = {
                "nc": score_answer_candidates_from_cache(model, processor, nc_layer_cache, gold, wrong, unknown),
                "ctx": score_answer_candidates_from_cache(model, processor, ctx_layer_cache, gold, wrong, unknown),
            }
        records.append(rec)
        tracker.append_record("records.jsonl", rec)
        tracker.mark_done(row_key, {"status": "ok"})
        accumulate_trace_record(layer_deltas, rec, metrics)
        used += 1
        tracker.update(skipped=skipped, used=used)

    if used > 0:
        for m in layer_deltas:
            layer_deltas[m] = [x / used for x in layer_deltas[m]]
        layer_score_mean = layer_deltas
    else:
        layer_score_mean = layer_deltas

    out = {
        "config": vars(args),
        "used": used,
        "skipped": skipped,
        "layer_scores": layer_deltas,
        "records": records,
    }

    out_parent = Path(args.out).parent
    os.makedirs(out_parent if str(out_parent) else ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if args.plot_prefix:
        plot_scores(layer_deltas, args.plot_prefix)

    # if args.scan_plan_out:
    #     plan = {
    #         "trace_mode": args.trace_mode,
    #         "suggested_top_layers": {
    #             m: sorted(range(n_layers), key=lambda i: layer_deltas[m][i], reverse=True)[: min(20, n_layers)]
    #             for m in metrics
    #         },
    #         "layer_scores": layer_deltas,
    #     }
    #     plan_parent = Path(args.scan_plan_out).parent
    #     os.makedirs(plan_parent if str(plan_parent) else ".", exist_ok=True)
    #     with open(args.scan_plan_out, "w", encoding="utf-8") as f:
    #         json.dump(plan, f, ensure_ascii=False, indent=2)
        # ---- scan plan generation ----
    if args.scan_plan_out:
        if args.scan_plan_metric and args.scan_plan_metrics:
            raise SystemExit("Use either --scan_plan_metric or --scan_plan_metrics, not both.")

        if args.scan_plan_metric:
            plan_metric = args.scan_plan_metric.strip()
            if plan_metric not in layer_score_mean:
                raise SystemExit(f"--scan_plan_metric '{plan_metric}' not in computed metrics {sorted(layer_score_mean)}")
            plan_scores = layer_score_mean[plan_metric]
            plan_info = {"mode": "single_metric", "metric": plan_metric, "weights": {plan_metric: 1.0}}

        elif args.scan_plan_metrics:
            plan_metrics = [p.strip() for p in args.scan_plan_metrics.split(",") if p.strip()]
            for pm in plan_metrics:
                if pm not in layer_score_mean:
                    raise SystemExit(f"--scan_plan_metrics includes '{pm}' not in computed metrics {sorted(layer_score_mean)}")
            w = parse_weights(args.scan_plan_weights, len(plan_metrics))
            plan_scores = combine_scores(layer_score_mean, plan_metrics, w)
            plan_info = {
                "mode": "weighted_metrics",
                "metrics": plan_metrics,
                "weights": {m: float(wi) for m, wi in zip(plan_metrics, w)},
            }

        else:
            pm = "hallucination_relief"
            plan_scores = layer_score_mean[pm]
            plan_info = {"mode": "default_first_metric", "metric": pm, "weights": {pm: 1.0}}

        scan_plan = build_scan_plan(
            plan_scores,
            tail_k=args.tail_k,
            preplateau_width=args.preplateau_width,
            plateau_min_score_frac=args.plateau_min_score_frac,
            plateau_eps=args.plateau_eps,
            plateau_min_len=args.plateau_min_len,
            topk_fallback=args.topk_fallback,
            rise_k=args.rise_k,
            rise_min_layer=args.rise_min_layer,
            rise_max_layer=args.rise_max_layer,
        )

        scan_plan_out = {
            "n_samples": used,
            "n_layers": n_layers,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "mask_scale": float(args.mask_scale),
            "layer_scale": args.layer_scale,
            "plan_scores_source": plan_info,
            "scan_plan": scan_plan,
        }

        sp_path = Path(args.scan_plan_out)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.write_text(json.dumps(scan_plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.finish(out=args.out, used=used, skipped=skipped)
    print("Saved to:", args.out)
    print(json.dumps({"used": used, "skipped": skipped, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

