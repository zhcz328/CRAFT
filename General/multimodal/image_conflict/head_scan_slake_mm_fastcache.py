import os
import json
import argparse
import importlib.util
import gc
import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import Any as TypingAny

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir
from analysis_image_conflict import (
    build_ic_prompt as shared_build_ic_prompt,
    build_nc_prompt as shared_build_nc_prompt,
    SYSTEM_PROMPT,
    get_answer_candidates_from_row,
    load_nc_ic_images,
    normalize_position,
    normalize_trace_mode,
    score_map_to_label,
)


def build_path_candidates(raw: str, image_root: str) -> List[Path]:
    raw = str(raw).strip()
    if not raw:
        return []

    root = Path(image_root)
    p = Path(raw)
    candidates: List[Path] = [p, root / p, root / p.name]

    win_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if win_match:
        drive = win_match.group(1).lower()
        tail = win_match.group(2).replace("\\", "/")
        candidates.append(Path(f"/mnt/{drive}/{tail}"))

    wsl_match = re.match(r"^/mnt/([A-Za-z])/(.*)$", raw)
    if wsl_match:
        drive = wsl_match.group(1).upper()
        tail = wsl_match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{tail}"))

    return candidates

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


def maybe_empty_cuda_cache(step_idx: int, every: int = 8) -> None:
    if every <= 0 or (step_idx % every != 0):
        return
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


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


# -------------------------
# data / prompt helpers
# -------------------------
def norm_text(x: str) -> str:
    return str(x).strip().lower()


def resize_image_max_side(image: Image.Image, max_side: int = 672) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def resolve_image_path(row, image_root: str) -> Optional[Path]:
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        candidates.extend(build_path_candidates(row["image_path"], image_root))

    if "img_id" in row and pd.notna(row["img_id"]):
        candidates.extend(build_path_candidates(row["img_id"], image_root))

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


# -------------------------
# model / cache helpers
# -------------------------
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
        attn_implementation="eager",
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def build_inputs_mm(processor, prompt: str, image: Image.Image, device):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    # Keep prebuilt sample inputs on CPU to avoid holding all samples on GPU.
    # They will be moved to model.device inside build_prompt_cache_from_inputs().
    return base_inputs


@torch.inference_mode()
def build_prompt_cache_from_inputs(model, base_inputs):
    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=True, output_hidden_states=False)

    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()

    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
    }


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

    # 鍏抽敭锛歝ontinuation 鐨勭涓€涓?token 蹇呴』鐢?prompt 鏈€鍚庝竴涓綅缃殑 logits 璁″垎
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
        output_hidden_states=False,
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
    return {
        "gold_lp": float(score_map.get(gold, float("-inf"))),
        "wrong_lp": float(score_map.get(wrong, float("-inf"))),
        "unknown_lp": float(score_map.get(unknown, float("-inf"))),
        "candidate_scores": score_map,
        "pred": pred_answer,
        "pred_label": pred_label,
    }


# -------------------------
# metrics
# -------------------------
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


def parse_metric_weights(metrics_str: str, weights_str: str) -> Tuple[List[str], List[float]]:
    metrics = [m.strip() for m in metrics_str.split(",") if m.strip()]
    allowed = {"gold_wrong_margin", "follow_conflict", "follow_context", "context_flip"}
    if not metrics:
        raise SystemExit("--metrics cannot be empty")
    for m in metrics:
        if m not in allowed:
            raise SystemExit(f"Unknown metric '{m}'. Allowed: {sorted(allowed)}")

    if not weights_str:
        weights = [1.0 / len(metrics)] * len(metrics)
        return metrics, weights

    parts = [p.strip() for p in weights_str.split(",") if p.strip()]
    if len(parts) != len(metrics):
        raise SystemExit(f"--weights expects {len(metrics)} values, got {len(parts)}")
    weights = [float(x) for x in parts]
    s = sum(weights)
    if s == 0:
        raise SystemExit("--weights sum to 0")
    weights = [w / s for w in weights]
    return metrics, weights


def weighted_metric(metrics: List[str], weights: List[float], scores: dict, gold: str, wrong: str, trace_mode: str) -> float:
    return sum(
        w * metric_from_scores(m, scores, gold, wrong, trace_mode)
        for m, w in zip(metrics, weights)
    )


def evaluate_single_head(
    model,
    processor,
    samples,
    metrics: List[str],
    trace_mode: str,
    keep_mode: str,
    layer_idx: int,
    head_idx: int,
    save_records: bool = False,
):
    handles = install_head_mask_hooks(model, {int(layer_idx): [int(head_idx)]}, keep_mode=keep_mode)
    try:
        score_sum = 0.0
        ctx_gain_sum = 0.0
        nc_damage_sum = 0.0
        n = 0
        metric_ctx_gain = {m: 0.0 for m in metrics}
        per_head_records = []

        for sample_idx, s in enumerate(samples, start=1):
            gold = s["gold"]
            wrong = s["wrong"]
            unknown = s["unknown"]

            nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
            nc_ab_scores = score_answer_candidates_from_cache(model, processor, nc_cache_ab, gold, wrong, unknown)
            ctx_ab_scores = score_answer_candidates_from_cache(model, processor, ctx_cache_ab, gold, wrong, unknown)

            ab_nc_margin = nc_gold_margin(nc_ab_scores)
            nc_damage = max(0.0, s["base_nc_gold_margin"] - ab_nc_margin)
            ctx_follow_context = metric_from_scores("follow_context", ctx_ab_scores, gold, wrong, trace_mode)
            ctx_gain = ctx_follow_context - s["base_ctx_follow_context"]
            score_sum += ctx_gain - nc_damage
            ctx_gain_sum += ctx_gain
            nc_damage_sum += nc_damage

            for m in metrics:
                metric_ctx_gain[m] += (
                    metric_from_scores(m, ctx_ab_scores, gold, wrong, trace_mode)
                    - s["base_metric_values"][m]["ctx"]
                )

            if save_records:
                per_head_records.append({
                    "question": s["question"],
                    "gold": gold,
                    "wrong": wrong,
                    "unknown": unknown,
                    "image_path": s["image_path"],
                    "base_nc": s["nc_scores"],
                    "base_ctx": s["ctx_scores"],
                    "ablated_nc": nc_ab_scores,
                    "ablated_ctx": ctx_ab_scores,
                    "base_ctx_follow_context": s["base_ctx_follow_context"],
                    "ablated_ctx_follow_context": ctx_follow_context,
                    "ctx_follow_context_gain": ctx_gain,
                    "base_nc_gold_margin": s["base_nc_gold_margin"],
                    "ablated_nc_gold_margin": ab_nc_margin,
                    "nc_gold_margin_damage": nc_damage,
                    "hallucination_relief": ctx_gain - nc_damage,
                })
            n += 1
            maybe_empty_cuda_cache(sample_idx, every=8)
    finally:
        remove_hooks(handles)

    denom = max(n, 1)
    return {
        "mean_hallucination_relief": score_sum / denom,
        "mean_ic_follow_context_gain": ctx_gain_sum / denom,
        "mean_nc_gold_margin_damage": nc_damage_sum / denom,
        "metric_mean_ctx_gain": {m: metric_ctx_gain[m] / denom for m in metrics},
        "records": per_head_records,
        "n_samples": n,
    }


# -------------------------
# head mask hooks (ported from old head_scan)
# -------------------------
def _get_num_heads_and_hidden(model):
    cand_cfgs = []

    # 椤跺眰
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cand_cfgs.append(cfg)

        # 甯歌鐨勫妯℃€?灏佽瀛楁
        for name in [
            "text_config",
            "language_config",
            "llm_config",
            "decoder_config",
        ]:
            sub = getattr(cfg, name, None)
            if sub is not None:
                cand_cfgs.append(sub)

    # common nested modules
    for obj in [
        getattr(model, "language_model", None),
        getattr(model, "model", None),
    ]:
        if obj is not None:
            subcfg = getattr(obj, "config", None)
            if subcfg is not None:
                cand_cfgs.append(subcfg)

    # 鍘婚噸
    uniq = []
    seen = set()
    for c in cand_cfgs:
        if c is None:
            continue
        if id(c) not in seen:
            uniq.append(c)
            seen.add(id(c))

    for c in uniq:
        nheads = getattr(c, "num_attention_heads", None) or getattr(c, "n_head", None)
        hidden = (
            getattr(c, "hidden_size", None)
            or getattr(c, "n_embd", None)
            or getattr(c, "d_model", None)
        )
        if nheads is not None:
            return int(nheads), (int(hidden) if hidden is not None else None)

    return None, None


def _find_layers_for_attn(model):
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
        ("gpt_neox", "layers"),
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
    raise RuntimeError(
        f"Cannot locate transformer layers for attention modules. "
        f"Model type: {type(m)}. Tried paths: {tried}"
    )


def _find_self_attn_modules(model):
    layers = _find_layers_for_attn(model)
    layer2attn = {}
    for i, layer in enumerate(layers):
        mod = None
        for name in ["self_attn", "attn", "attention"]:
            if hasattr(layer, name):
                mod = getattr(layer, name)
                break
        if mod is not None:
            layer2attn[i] = mod

    if not layer2attn:
        raise RuntimeError(
            f"Cannot find any self-attention modules in located layers. "
            f"First layer type: {type(layers[0]) if len(layers) > 0 else 'EMPTY'}"
        )
    return layer2attn, len(layers)


def _apply_head_specific_mask(attn_mask, n_heads: int, heads_to_mask, keep_mode: str = "self"):
    if attn_mask is None:
        return None
    if not torch.is_tensor(attn_mask):
        return attn_mask
    if attn_mask.dim() != 4:
        return attn_mask

    B, Hm, T, S = attn_mask.shape

    if Hm == 1:
        mask = attn_mask.expand(B, n_heads, T, S).clone()
    elif Hm == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask

    neg = torch.finfo(mask.dtype).min

    for h in heads_to_mask:
        if not (0 <= h < n_heads):
            continue
        mask[:, h, :, :] = neg
        if keep_mode == "bos":
            mask[:, h, :, 0] = 0
        else:
            diag_len = min(T, S)
            idx = torch.arange(diag_len, device=mask.device)
            mask[:, h, idx, idx] = 0

    return mask


def install_head_mask_hooks(model, layer2heads: Dict[int, List[int]], keep_mode: str = "self"):
    layer2attn, _ = _find_self_attn_modules(model)
    n_heads_cfg, _ = _get_num_heads_and_hidden(model)
    if n_heads_cfg is None or n_heads_cfg <= 0:
        raise RuntimeError("Cannot read num_attention_heads from model.config.")

    handles = []
    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(h) for h in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, args, kwargs):
                attn_mask = None
                if kwargs is not None and "attention_mask" in kwargs:
                    attn_mask = kwargs["attention_mask"]
                elif len(args) >= 2:
                    attn_mask = args[1]

                new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)
                if new_mask is attn_mask:
                    return args, kwargs

                if kwargs is not None and "attention_mask" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = new_mask
                    return args, kwargs

                args = list(args)
                if len(args) >= 2:
                    args[1] = new_mask
                return tuple(args), kwargs
            return pre_hook

        try:
            handle = attn_mod.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=True)
        except TypeError:
            def make_pre_hook_no_kwargs(heads_local):
                def pre_hook_no_kwargs(module, inputs):
                    args = list(inputs)
                    if len(args) < 2:
                        return inputs
                    attn_mask = args[1]
                    new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)
                    args[1] = new_mask
                    return tuple(args)
                return pre_hook_no_kwargs
            handle = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# -------------------------
# scan plan loader
# -------------------------
def _dedup_sorted_layers(layers):
    return sorted({int(x) for x in (layers or [])})


def load_plan_layers(
    plan_path: str,
    round_name: Optional[str] = None,
    plan_layer_key: str = "merged_unique_layers",
):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    rounds = plan.get("rounds")
    using_round_schema = isinstance(rounds, list) and bool(rounds)
    if using_round_schema:
        round_layers = [
            (r.get("name", f"round{idx}"), _dedup_sorted_layers(r.get("layers", [])))
            for idx, r in enumerate(rounds)
        ]
        round_layers = [(n, ls) for (n, ls) in round_layers if ls]
        if not round_layers:
            raise ValueError("rounds contains no layer lists.")
    else:
        sp = plan.get("scan_plan")
        if not isinstance(sp, dict):
            raise ValueError("plan file missing 'rounds' (old schema) and missing 'scan_plan' (new schema).")

        selected_layers = _dedup_sorted_layers(sp.get(plan_layer_key, []))
        if not selected_layers:
            available = sorted(k for k, v in sp.items() if isinstance(v, list) and v)
            raise ValueError(
                f"scan_plan missing or empty '{plan_layer_key}'. Available non-empty layer lists: {available}"
            )

        # Uniformly wrap new-schema layers as a pseudo round.
        round_layers = [(plan_layer_key, selected_layers)]

    if round_name is None:
        return plan, round_layers

    if not using_round_schema:
        if round_name in {plan_layer_key, "merged_unique_layers"}:
            return plan, round_layers
        raise ValueError(
            f"round '{round_name}' not found for new scan_plan schema. "
            f"Use --plan_layer_key to choose from the available layer lists instead."
        )

    for n, ls in round_layers:
        if n == round_name:
            return plan, [(n, ls)]

    raise ValueError(f"round '{round_name}' not found. Available: {[n for n, _ in round_layers]}")


def resume_key_for_head(round_name: str, layer_idx: int, head_idx: int) -> str:
    return f"{round_name}:layer={int(layer_idx)}:head={int(head_idx)}"


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
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--sample_filter", type=str, default="none", choices=["none", "hallucination"])

    ap.add_argument("--metrics", type=str, default="follow_context")
    ap.add_argument("--weights", type=str, default="")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan", type=str, default="")
    ap.add_argument("--round", default='merged_unique_layers', help="optional group name; for old rounds schema")
    ap.add_argument("--plan_layer_key", type=str, default="merged_unique_layers",
                    help="for new scan_plan schema, choose which layer list to scan (e.g. merged_unique_layers/core_layers/positive_layers)")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--plan_out_dir", type=str, default="result/headscan_slake_mm")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save_records", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.plan_out_dir = prefix_model_relative_path(args.plan_out_dir, model_name=args.model_name, model=args.model)
    args.position = normalize_position(args.position)
    args.trace_mode = normalize_trace_mode(args.trace_mode)
    if args.mask_scale <= 0:
        raise SystemExit("--mask_scale must be positive")

    metrics, weights = parse_metric_weights(args.metrics, args.weights)

    df = pd.read_csv(args.data_csv)
    if "gold" not in df.columns:
        raise ValueError("CSV must contain a 'gold' column.")
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")

    model, processor = load_model_and_processor(args.model, args.device, args.dtype)
    layers = _get_layers(model)
    n_layers = len(layers)

    n_heads, _ = _get_num_heads_and_hidden(model)
    if n_heads is None or n_heads <= 0:
        raise RuntimeError("Missing num_attention_heads/n_head in config or nested text config")

    rows = df.to_dict("records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    samples = []
    skipped = 0

    for row_idx, row in enumerate(tqdm(rows, desc="Build samples"), start=1):
        gold, wrong, unknown, _answer_candidates = get_answer_candidates_from_row(row)
        question = str(row["question"])
        if not gold or not wrong or gold == wrong:
            skipped += 1
            continue

        try:
            img_path, det_path, mask_path, target_labels, nc_image, ic_image = load_nc_ic_images(
                row,
                args.image_root,
                max_side=672,
                mask_scale=args.mask_scale,
            )
        except Exception:
            skipped += 1
            continue

        nc_prompt = shared_build_nc_prompt(question)
        ctx_prompt = shared_build_ic_prompt(question)

        nc_inputs = build_inputs_mm(processor, nc_prompt, nc_image, model.device)
        ctx_inputs = build_inputs_mm(processor, ctx_prompt, ic_image, model.device)

        nc_cache = build_prompt_cache_from_inputs(model, nc_inputs)
        ctx_cache = build_prompt_cache_from_inputs(model, ctx_inputs)
        nc_scores = score_answer_candidates_from_cache(model, processor, nc_cache, gold, wrong, unknown)
        ctx_scores = score_answer_candidates_from_cache(model, processor, ctx_cache, gold, wrong, unknown)
        if args.sample_filter == "hallucination" and not is_hallucination_sample(nc_scores, ctx_scores):
            skipped += 1
            continue

        sample = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "unknown": unknown,
            "image_path": str(img_path),
            "detection_path": str(det_path),
            "mask_path": str(mask_path),
            "ic_target_labels": target_labels,
            "nc_inputs": nc_inputs,
            "ctx_inputs": ctx_inputs,
            "nc_scores": nc_scores,
            "ctx_scores": ctx_scores,
            "base_nc_gold_margin": nc_gold_margin(nc_scores),
            "base_ctx_follow_context": metric_from_scores("follow_context", ctx_scores, gold, wrong, args.trace_mode),
            "base_metric_values": {
                m: {
                    "nc": metric_from_scores(m, nc_scores, gold, wrong, args.trace_mode),
                    "ctx": metric_from_scores(m, ctx_scores, gold, wrong, args.trace_mode),
                }
                for m in metrics
            },
        }
        samples.append(sample)
        maybe_empty_cuda_cache(row_idx, every=8)

    if not samples:
        raise SystemExit("No usable examples after filtering.")

    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            Path(__file__).resolve().parent,
            task_name="head_scan",
            model_name=args.model_name,
            model=args.model,
            position=args.position,
        ),
        enabled=args.resume,
    )
    tracker.start(
        data_csv=args.data_csv,
        image_root=args.image_root,
        model=args.model,
        model_name=args.model_name,
        trace_mode=args.trace_mode,
        position=args.position,
        keep_mode=args.keep_mode,
        plan=args.plan,
        plan_layer_key=args.plan_layer_key,
        plan_out_dir=args.plan_out_dir,
        out=args.out,
        save_records=bool(args.save_records),
        sample_filter=args.sample_filter,
        n_samples=len(samples),
        skipped=skipped,
    )

    def run_round(scan_layers: List[int], out_path: str, round_name: str):
        results = tracker.read_records(f"{round_name}_results.jsonl") if args.resume else []
        records = tracker.read_records(f"{round_name}_records.jsonl") if (args.resume and args.save_records) else []
        existing_heads = {(int(item["layer"]), int(item["head"])) for item in results}
        for layer_idx in scan_layers:
            for head_idx in tqdm(range(n_heads), desc=f"Scan L{layer_idx}", unit="head"):
                head_key = resume_key_for_head(round_name, layer_idx, head_idx)
                if tracker.is_done(head_key) or (int(layer_idx), int(head_idx)) in existing_heads:
                    continue
                stats = evaluate_single_head(
                    model=model,
                    processor=processor,
                    samples=samples,
                    metrics=metrics,
                    trace_mode=args.trace_mode,
                    keep_mode=args.keep_mode,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    save_records=bool(args.save_records),
                )

                item = {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "mean_abs_effect_reduction": stats["mean_hallucination_relief"],
                    "mean_abs_base_change": stats["mean_nc_gold_margin_damage"],
                    "mean_ic_follow_context_gain": stats["mean_ic_follow_context_gain"],
                    "mean_nc_gold_margin_damage": stats["mean_nc_gold_margin_damage"],
                    "metric_mean_abs_effect_reduction": dict(stats["metric_mean_ctx_gain"]),
                    "metric_mean_abs_base_change": {"nc_gold_margin": stats["mean_nc_gold_margin_damage"]},
                }
                results.append(item)
                existing_heads.add((int(layer_idx), int(head_idx)))
                tracker.append_record(f"{round_name}_results.jsonl", item)
                tracker.mark_done(
                    head_key,
                    {
                        "round": round_name,
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                    },
                )

                if args.save_records:
                    record_item = {
                        "layer": int(layer_idx),
                        "head": int(head_idx),
                        "records": stats["records"],
                    }
                    records.append(record_item)
                    tracker.append_record(f"{round_name}_records.jsonl", record_item)

                tracker.update(
                    current_round=round_name,
                    current_output=out_path,
                    completed_heads=len(results),
                    total_heads=sum(len(range(n_heads)) for _ in scan_layers),
                )
                maybe_empty_cuda_cache(len(results), every=1)

        results.sort(key=lambda r: (-(r["mean_abs_effect_reduction"]), r["mean_abs_base_change"], r["layer"], r["head"]))
        out = {
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "model": args.model,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "keep_mode": args.keep_mode,
            "mask_scale": float(args.mask_scale),
            "sample_filter": args.sample_filter,
            "metrics": metrics,
            "weights": weights,
            "n_samples": len(samples),
            "skipped": skipped,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "round": round_name,
            "scan_layers": scan_layers,
            "top20": results[:20],
            "results": results,
        }
        if args.save_records:
            out["records"] = records

        out_parent = Path(out_path).parent
        os.makedirs(out_parent if str(out_parent) else ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    if args.plan:
        _, round_layers = load_plan_layers(args.plan, args.round or None, args.plan_layer_key)
        Path(args.plan_out_dir).mkdir(parents=True, exist_ok=True)

        all_rounds = []
        for rname, scan_layers in round_layers:
            out_path = str(Path(args.plan_out_dir) / f"head_scan_{rname}.json")
            run_round(scan_layers, out_path, rname)
            all_rounds.append({"round": rname, "result_path": out_path, "scan_layers": scan_layers})

        summary_path = str(Path(args.plan_out_dir) / "head_scan_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "data_csv": args.data_csv,
                "image_root": args.image_root,
                "model": args.model,
                "trace_mode": args.trace_mode,
                "position": args.position,
                "keep_mode": args.keep_mode,
                "metrics": metrics,
                "weights": weights,
                "limit": args.limit,
                "plan": args.plan,
                "plan_layer_key": args.plan_layer_key,
                "rounds_ran": [x["round"] for x in all_rounds],
                "round_outputs": all_rounds,
                "n_samples": len(samples),
                "skipped": skipped,
                "saved": summary_path,
            }, f, ensure_ascii=False, indent=2)
        print(f"[OK] Saved per-round results under: {args.plan_out_dir}")
        print(f"[OK] Saved summary: {summary_path}")
        tracker.finish(saved=summary_path, rounds_ran=[x["round"] for x in all_rounds], skipped=skipped, n_samples=len(samples))
        return

    if args.layers:
        scan_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
        out_path = args.out or "head_scan_manual.json"
        run_round(scan_layers, out_path, "manual")
        print(f"[OK] Saved: {out_path}")
        tracker.finish(saved=out_path, rounds_ran=["manual"], skipped=skipped, n_samples=len(samples))
        return

    raise SystemExit("Either provide --plan or --layers.")


if __name__ == "__main__":
    main()
