#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import copy
import random
import argparse
import importlib.util
import re
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from typing import Any as TypingAny

import pandas as pd
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat as shared_load_processor_with_compat,
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
    normalize_position as normalize_image_conflict_position,
    normalize_trace_mode,
    score_map_to_label,
)
from filter_fine_grained import score_candidate_options_cached
from mask_utils import build_masked_image


# =========================================================
# 基础工具
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def str2dtype(x: str):
    x = x.lower()
    if x in ("fp16", "float16", "half"):
        return torch.float16
    if x in ("bf16", "bfloat16"):
        return torch.bfloat16
    if x in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {x}")


def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


def to_device(obj, device, float_dtype=None):
    if torch.is_tensor(obj):
        if float_dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=float_dtype)
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device, float_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device, float_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device, float_dtype) for v in obj)
    # transformers BatchEncoding/BatchFeature expose .to(device)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except Exception:
            pass
    return obj


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if len(xs) > 0 else 0.0


def maybe_empty_cuda_cache(step_idx: int, every: int = 8) -> None:
    if every <= 0 or step_idx % every != 0:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def resize_image_max_side(image: Image.Image, max_side: int = 672) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def ensure_video_import_compat() -> None:
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = TypingAny
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = TypingAny


def _clear_hf_dynamic_modules_for_hulumed() -> None:
    # A failed first import can leave partially initialized dynamic modules behind.
    for mod_name in list(sys.modules):
        lower_name = mod_name.lower()
        if not mod_name.startswith("transformers_modules."):
            continue
        if "hulumed" in lower_name or "hulu_med" in lower_name:
            sys.modules.pop(mod_name, None)


def load_processor_with_compat(model_name: str):
    return shared_load_processor_with_compat(model_name)


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


def model_prefers_cache_object(model) -> bool:
    candidates = [_get_underlying_model(model)]
    base_model = getattr(candidates[0], "base_model", None)
    if base_model is not None:
        candidates.append(base_model)
    language_model = getattr(candidates[0], "language_model", None)
    if language_model is not None:
        candidates.append(language_model)

    for cand in candidates:
        cfg = getattr(cand, "config", None)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if model_type.startswith("qwen3"):
            return True
    return False


def normalize_past_key_values_for_model(model, past_key_values):
    if past_key_values is None:
        return None
    if not isinstance(past_key_values, tuple):
        return past_key_values
    if not model_prefers_cache_object(model):
        return past_key_values

    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(past_key_values)
    except Exception:
        return past_key_values


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


# =========================================================
# 模型结构相关
# =========================================================

def _get_underlying_model(model):
    return model.module if hasattr(model, "module") else model


def _get_num_heads_and_hidden(model):
    cand_cfgs = []

    m = _get_underlying_model(model)
    cfg = getattr(m, "config", None)
    if cfg is not None:
        cand_cfgs.append(cfg)
        for name in ["text_config", "language_config", "llm_config", "decoder_config"]:
            sub = getattr(cfg, name, None)
            if sub is not None:
                cand_cfgs.append(sub)

    for obj in [
        getattr(m, "language_model", None),
        getattr(m, "model", None),
    ]:
        if obj is not None:
            subcfg = getattr(obj, "config", None)
            if subcfg is not None:
                cand_cfgs.append(subcfg)

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
    m = _get_underlying_model(model)

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
            f"Cannot find any self-attention modules. "
            f"First layer type: {type(layers[0]) if len(layers) > 0 else 'EMPTY'}"
        )
    return layer2attn, len(layers)


# =========================================================
# head mask hook
# =========================================================

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


def install_head_mask_hooks(model, layer_to_heads: Dict[int, List[int]], keep_mode="self"):
    layer2attn, _ = _find_self_attn_modules(model)
    n_heads_cfg, _ = _get_num_heads_and_hidden(model)
    if n_heads_cfg is None or n_heads_cfg <= 0:
        raise RuntimeError("Cannot read num_attention_heads from model.config.")

    handles = []
    for layer_idx, heads in layer_to_heads.items():
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


def remove_handles(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


# =========================================================
# 数据 / prompt
# =========================================================

def infer_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower_map:
            return lower_map[x.lower()]
    if required:
        raise ValueError(f"Cannot find required column from candidates={candidates}, available={list(df.columns)}")
    return None


def resolve_image_path(image_root: str, img_rel: str) -> str:
    for candidate in build_path_candidates(img_rel, image_root):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Image not found: {img_rel}")


def invert_yesno(x: str) -> str:
    x = normalize_text(x).lower()
    if x == "yes":
        return "no"
    if x == "no":
        return "yes"
    return x


def build_context_text(row: pd.Series) -> str:
    # 尽量兼容常见列名
    for c in ["supporting_context", "context", "evidence", "support", "support_text", "caption"]:
        if c in row and normalize_text(row[c]):
            return normalize_text(row[c])
    return ""


def normalize_position(position: str) -> str:
    return normalize_image_conflict_position(position)


def make_prompts(question: str):
    question = normalize_text(question)
    nc_prompt = shared_build_nc_prompt(question)
    ctx_prompt = shared_build_ic_prompt(question)
    return nc_prompt, ctx_prompt


def build_messages_multimodal(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]


@torch.inference_mode()
def score_answer_candidates_eval_style(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    gold: str,
    wrong: str,
    unknown: str,
    trace_mode: str,
):
    messages = build_messages_multimodal(image, prompt)
    score_map = score_candidate_options_cached(model, processor, messages, [gold, wrong, unknown])

    gold_lp = float(score_map.get(gold, float("-inf")))
    wrong_lp = float(score_map.get(wrong, float("-inf")))
    unknown_lp = float(score_map.get(unknown, float("-inf")))
    if trace_mode == "cc":
        ctx_target = gold
        ctx_lp = gold_lp
        other_lp = max(wrong_lp, unknown_lp)
    elif trace_mode == "conflict":
        ctx_target = unknown
        ctx_lp = unknown_lp
        other_lp = max(gold_lp, wrong_lp)
    else:
        raise ValueError(f"Unsupported trace_mode: {trace_mode}")
    pred_answer, pred_choice = score_map_to_label(score_map, gold, wrong, unknown)
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "unknown_lp": unknown_lp,
        "candidate_scores": score_map,
        "ctx_target": ctx_target,
        "ctx_lp": float(ctx_lp),
        "other_lp": float(other_lp),
        "pred_answer": pred_answer,
        "pred_choice": pred_choice,
    }


def build_inputs_mm(processor, image: Image.Image, prompt: str, device: str):
    # Qwen3-VL needs multimodal chat template to align image placeholders/features.
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
    return to_device(base_inputs, device)


# =========================================================
# fastcache / continuation scoring
# =========================================================

@torch.no_grad()
def build_prompt_cache_from_inputs(model, base_inputs: Dict[str, Any]):
    """
    返回:
      {
        "past_key_values": ...,
        "prompt_last_logits": [1, vocab]
      }
    """
    model_device = next(model.parameters()).device
    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = to_device(base_inputs, model_device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=True, return_dict=True)
    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()
    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
    }


@torch.no_grad()
def score_continuation_with_cache(
    model,
    tokenizer,
    prompt_cache: Dict[str, Any],
    continuation: str,
    device: str,
) -> float:
    """
    continuation 的第一个 token 用 prompt_last_logits 计分；
    后续 token 用 past_key_values 接着算。
    返回整个 continuation 的 log-prob sum。
    """
    tok = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")
    cont_ids = tok["input_ids"].to(device)  # [1, T]
    if cont_ids.numel() == 0:
        return float("-inf")

    attn_prompt = prompt_cache["attn_prompt"]
    past_key_values = prompt_cache["past_key_values"]
    prompt_last_logits = prompt_cache["prompt_last_logits"]
    cont_len = int(cont_ids.shape[1])

    first_logprobs = F.log_softmax(prompt_last_logits, dim=-1)
    total_lp = float(first_logprobs[0, int(cont_ids[0, 0])])

    if cont_len == 1:
        return float(total_lp)

    rest_inp = cont_ids[:, :-1]  # Let the model predict token 2..T.
    attn_full = torch.cat([attn_prompt, torch.ones_like(rest_inp)], dim=1)
    out = model(
        input_ids=rest_inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
        output_hidden_states=False,
    )
    log_probs = F.log_softmax(out.logits, dim=-1)
    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total_lp += float(log_probs[0, i - 1, tok_id])
    return float(total_lp)


@torch.no_grad()
def score_answer_candidates_with_cache(
    model,
    tokenizer,
    prompt_cache: Dict[str, Any],
    gold: str,
    wrong: str,
    unknown: str,
    trace_mode: str,
    device: str,
):
    score_map = {}
    for option in [gold, wrong, unknown]:
        option = str(option).strip().lower()
        if not option or option in score_map:
            continue
        score_map[option] = float(score_continuation_with_cache(model, tokenizer, prompt_cache, " " + option, device))

    gold_lp = float(score_map.get(gold, float("-inf")))
    wrong_lp = float(score_map.get(wrong, float("-inf")))
    unknown_lp = float(score_map.get(unknown, float("-inf")))
    if trace_mode == "cc":
        ctx_target = gold
        ctx_lp = gold_lp
        other_lp = max(wrong_lp, unknown_lp)
    elif trace_mode == "conflict":
        ctx_target = unknown
        ctx_lp = unknown_lp
        other_lp = max(gold_lp, wrong_lp)
    else:
        raise ValueError(f"Unsupported trace_mode: {trace_mode}")
    pred_answer, pred_choice = score_map_to_label(score_map, gold, wrong, unknown)
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "unknown_lp": unknown_lp,
        "candidate_scores": score_map,
        "ctx_target": ctx_target,
        "ctx_lp": float(ctx_lp),
        "other_lp": float(other_lp),
        "pred_answer": pred_answer,
        "pred_choice": pred_choice,
    }


def is_hallucination_sample(nc_scores: Dict[str, float], ctx_scores: Dict[str, float]) -> bool:
    return pred_label(nc_scores) == "gold" and pred_label(ctx_scores) != "unknown"


def score_answer_candidates_by_mode(
    model,
    processor,
    tokenizer,
    image: Image.Image,
    prompt: str,
    gold: str,
    wrong: str,
    unknown: str,
    trace_mode: str,
    scoring_mode: str,
    device: str,
):
    if scoring_mode == "eval":
        return score_answer_candidates_eval_style(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            gold=gold,
            wrong=wrong,
            unknown=unknown,
            trace_mode=trace_mode,
        )

    if scoring_mode == "cache":
        inputs = build_inputs_mm(processor, image, prompt, device)
        prompt_cache = build_prompt_cache_from_inputs(model, inputs)
        return score_answer_candidates_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            gold=gold,
            wrong=wrong,
            unknown=unknown,
            trace_mode=trace_mode,
            device=device,
        )

    raise ValueError(f"Unsupported scoring_mode: {scoring_mode}")


# =========================================================
# metric
# =========================================================

ALLOWED_METRICS = {
    "gold_wrong_margin",
    "follow_conflict",
    "follow_context",
    "context_flip",
}


def compute_scalar_metrics(scores: Dict[str, float], trace_mode: str) -> Dict[str, float]:
    gold_lp = scores["gold_lp"]
    wrong_lp = scores["wrong_lp"]
    ctx_lp = scores["ctx_lp"]
    other_lp = scores["other_lp"]
    unknown_lp = scores["unknown_lp"]

    out = {}
    out["gold_wrong_margin"] = float(gold_lp - wrong_lp)
    out["follow_conflict"] = float(unknown_lp - gold_lp)
    out["follow_context"] = float(ctx_lp - other_lp)
    out["context_flip"] = float(other_lp - ctx_lp)
    return out


def pred_label(scores: Dict[str, float]) -> str:
    return str(scores.get("pred_choice", "other"))


def nc_gold_margin(scores: Dict[str, float]) -> float:
    return float(scores["gold_lp"] - max(scores["wrong_lp"], scores["unknown_lp"]))


# =========================================================
# 读 selected heads
# =========================================================

def parse_selected_heads(json_path: str) -> List[Tuple[int, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # Support outputs from multiple scripts:
        # - selected_heads/results/heads/topk (legacy)
        # - selected (select_heads_merged_unique_layers.py)
        # - conflict_specific/backbone (head_groups style)
        for k in ["selected", "selected_heads", "results", "heads", "topk", "conflict_specific", "backbone"]:
            if k in data and isinstance(data[k], list):
                items = data[k]
                break

    if items is None:
        raise ValueError(f"Cannot parse selected heads from: {json_path}")

    pairs = []
    for x in items:
        if not isinstance(x, dict):
            continue

        layer = None
        head = None

        for lk in ["layer", "layer_idx", "l", "L"]:
            if lk in x:
                layer = int(x[lk])
                break

        for hk in ["head", "head_idx", "h", "H"]:
            if hk in x:
                head = int(x[hk])
                break

        if layer is None and "name" in x:
            # 允许 "L24-H11"
            name = str(x["name"])
            try:
                a = name.split("-")
                layer = int(a[0].replace("L", ""))
                head = int(a[1].replace("H", ""))
            except Exception:
                pass

        if layer is None or head is None:
            continue

        pairs.append((layer, head))

    pairs = sorted(set(pairs))
    if not pairs:
        raise ValueError(f"No valid selected heads found in: {json_path}")
    return pairs


def resolve_selected_heads_path(selected_heads: str, out_json: str, mode: str) -> str:
    if selected_heads:
        return selected_heads

    out_path = Path(out_json)
    result_root = out_path.parent.parent
    return str(result_root / f"selected_heads_core_layers_{mode}.json")


def random_select_heads(model, k: int, seed: int = 0):
    rng = random.Random(seed)
    n_heads, _ = _get_num_heads_and_hidden(model)
    _, n_layers = _find_self_attn_modules(model)

    all_pairs = []
    layers = list(range(n_layers))
    for l in layers:
        for h in range(n_heads):
            all_pairs.append((l, h))
    rng.shuffle(all_pairs)
    return sorted(all_pairs[:k])


def pack_layer_to_heads(pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    m = {}
    for l, h in pairs:
        m.setdefault(int(l), []).append(int(h))
    for l in list(m.keys()):
        m[l] = sorted(set(m[l]))
    return dict(sorted(m.items(), key=lambda x: x[0]))


def resume_key_for_ablation_sample(
    sample: Dict[str, Any],
    data_csv: str,
    position: str,
    trace_mode: str,
    mask_scope: str,
    sample_filter: str,
    scoring_mode: str,
) -> str:
    data_tag = Path(data_csv).stem
    return (
        f"{data_tag}\t{sample['row_idx']}\t{sample['question']}\t{sample['gold']}\t"
        f"{sample['wrong']}\t{position}\t{trace_mode}\t{mask_scope}\t{sample_filter}\t{scoring_mode}"
    )


# =========================================================
# 样本构建
# =========================================================

def build_samples(args, df, processor, tokenizer, model, device):
    q_col = infer_col(df, ["question", "query"])
    gold_col = infer_col(df, ["answer", "gold", "gt_answer", "label"])

    samples = []
    skipped_oom = 0
    skipped_by_filter = 0

    iterable = df.iterrows()
    if args.max_examples > 0:
        iterable = list(df.head(args.max_examples).iterrows())

    for idx, row in tqdm(iterable, total=(len(df.head(args.max_examples)) if args.max_examples > 0 else len(df)), desc="Build samples"):
        question = normalize_text(row[q_col])
        gold, wrong, unknown, _answer_candidates = get_answer_candidates_from_row(row)
        if not gold:
            gold = normalize_text(row[gold_col]).lower()

        if not question or not gold or not unknown or gold == unknown:
            continue

        try:
            img_path, det_path, mask_path, target_labels, nc_image, ic_image = load_nc_ic_images(
                row,
                args.image_root,
                max_side=args.max_image_side,
                mask_scale=args.mask_scale,
            )
        except Exception:
            continue

        nc_prompt, ctx_prompt = make_prompts(question=question)

        try:
            nc_scores = score_answer_candidates_by_mode(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=nc_image,
                prompt=nc_prompt,
                gold=gold,
                wrong=wrong,
                unknown=unknown,
                trace_mode=args.trace_mode,
                scoring_mode=args.scoring_mode,
                device=device,
            )
            ctx_scores = score_answer_candidates_by_mode(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=ic_image,
                prompt=ctx_prompt,
                gold=gold,
                wrong=wrong,
                unknown=unknown,
                trace_mode=args.trace_mode,
                scoring_mode=args.scoring_mode,
                device=device,
            )
            if args.sample_filter == "hallucination" and not is_hallucination_sample(nc_scores, ctx_scores):
                skipped_by_filter += 1
                continue
            nc_metrics = compute_scalar_metrics(nc_scores, args.trace_mode)
            ctx_metrics = compute_scalar_metrics(ctx_scores, args.trace_mode)
        except Exception as e:
            if not is_cuda_oom_error(e):
                continue
            skipped_oom += 1
            maybe_empty_cuda_cache(len(samples) + skipped_oom, every=1)
            continue
        finally:
            maybe_empty_cuda_cache(len(samples) + 1)

        samples.append({
            "row_idx": int(idx),
            "img_path": img_path,
            "question": question,
            "context": "",
            "gold": gold,
            "wrong": wrong,
            "unknown": unknown,
            "nc_prompt": nc_prompt,
            "ctx_prompt": ctx_prompt,
            "detection_path": str(det_path),
            "mask_path": str(mask_path),
            "ic_target_labels": target_labels,
            "base_nc_scores": nc_scores,
            "base_ctx_scores": ctx_scores,
            "base_nc_metrics": nc_metrics,
            "base_ctx_metrics": ctx_metrics,
            "base_nc_pred": pred_label(nc_scores),
            "base_ctx_pred": pred_label(ctx_scores),
            "base_nc_gold_margin": nc_gold_margin(nc_scores),
        })

    if skipped_oom > 0:
        print(f"[WARN] Skipped {skipped_oom} samples due to CUDA OOM during baseline cache build.")
    if skipped_by_filter > 0:
        print(f"[INFO] Filtered out {skipped_by_filter} samples by sample_filter={args.sample_filter}.")

    return samples


def rebuild_sample_images(sample: Dict[str, Any], max_image_side: int, mask_scale: float):
    nc_image = resize_image_max_side(Image.open(sample["img_path"]).convert("RGB"), max_image_side)
    ic_image, _ = build_masked_image(
        source_path=sample["img_path"],
        detection_path=sample["detection_path"],
        target_labels=sample["ic_target_labels"],
        mask_path=sample["mask_path"],
        image_id=str(sample.get("img_id", "")).strip() or None,
        mask_scale=mask_scale,
    )
    ic_image = resize_image_max_side(ic_image, max_image_side)
    return nc_image, ic_image


# =========================================================
# 主实验
# =========================================================

def summarize_metric_changes(records: List[Dict[str, Any]], metrics: List[str]):
    out = {}
    for m in metrics:
        eff_fixed_nc = [r["metric_effect_reduction_fixed_nc"][m] for r in records]
        eff_scope = [r["metric_effect_reduction_mask_scope"][m] for r in records]
        base_ch = [r["metric_abs_base_change"][m] for r in records]
        ctx_ch = [r["metric_abs_ctx_change"][m] for r in records]
        out[m] = {
            "mean_abs_effect_reduction_fixed_nc": safe_mean(eff_fixed_nc),
            "mean_abs_effect_reduction_mask_scope": safe_mean(eff_scope),
            "mean_abs_base_change": safe_mean(base_ch),
            "mean_abs_ctx_change": safe_mean(ctx_ch),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--mode", type=str, default="pareto", choices=["pareto", "custom"])
    ap.add_argument("--selected_heads", type=str, default="")

    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--max_image_side", type=int, default=672)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["conflict"])
    ap.add_argument(
        "--position",
        type=str,
        default="image_conflict",
    )
    ap.add_argument("--mask_scale", type=float, default=1.0)

    ap.add_argument("--metrics", type=str, default="follow_context")
    ap.add_argument("--mask_scope", type=str, default="all", choices=["all", "ctx_only"])
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])
    ap.add_argument("--scoring_mode", type=str, default="cache", choices=["cache", "eval"])
    ap.add_argument("--sample_filter", type=str, default="hallucination", choices=["hallucination", "none"])

    ap.add_argument("--random_ablate", action="store_true")
    ap.add_argument("--random_seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_json = prefix_model_relative_path(args.out_json, model_name=args.model_name, model=args.model)
    args.selected_heads = resolve_selected_heads_path(args.selected_heads, args.out_json, args.mode)
    args.position = normalize_position(args.position)
    args.trace_mode = normalize_trace_mode(args.trace_mode)
    if args.mask_scale <= 0:
        raise ValueError("--mask_scale must be positive")

    set_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    dtype = str2dtype(args.dtype)

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    for m in metrics:
        if m not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric: {m}")

    df = pd.read_csv(args.data_csv)

    processor = load_processor_with_compat(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # 尝试两种加载方式
    model = shared_load_mm_model(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    model = model.to(device)
    model.eval()

    samples = build_samples(args, df, processor, tokenizer, model, device)
    if len(samples) == 0:
        raise RuntimeError("No valid samples were built.")
    data_tag = Path(args.data_csv).stem
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=Path(__file__).resolve().parent,
            task_name=(
                f"ablate_head_random_{data_tag}"
                if args.random_ablate
                else f"ablate_head_{data_tag}"
            ),
            model_name=args.model_name,
            model=args.model,
            position=args.position,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="ablate_head_random" if args.random_ablate else "ablate_head",
        data_csv=args.data_csv,
        out_json=args.out_json,
        mode=args.mode,
        position=args.position,
        trace_mode=args.trace_mode,
        mask_scope=args.mask_scope,
        sample_filter=args.sample_filter,
        scoring_mode=args.scoring_mode,
    )

    selected_pairs = parse_selected_heads(args.selected_heads)
    if args.random_ablate:
        selected_pairs = random_select_heads(
            model=model,
            k=len(selected_pairs),
            seed=args.random_seed,
        )

    layer_to_heads = pack_layer_to_heads(selected_pairs)

    # baseline summary
    base_effects = {m: [] for m in metrics}
    for s in samples:
        for m in metrics:
            base_effects[m].append(abs(s["base_ctx_metrics"][m] - s["base_nc_metrics"][m]))

    # 联合安装 hooks
    handles = install_head_mask_hooks(model, layer_to_heads, keep_mode=args.keep_mode)

    records = tracker.read_records("records.jsonl")
    skipped_rebuild = 0
    first_rebuild_error = ""
    try:
        for s in tqdm(samples, desc="Ablate selected heads"):
            sample_key = resume_key_for_ablation_sample(
                s,
                args.data_csv,
                args.position,
                args.trace_mode,
                args.mask_scope,
                args.sample_filter,
                args.scoring_mode,
            )
            if tracker.is_done(sample_key):
                continue
            try:
                nc_image, ctx_image = rebuild_sample_images(
                    s,
                    max_image_side=args.max_image_side,
                    mask_scale=args.mask_scale,
                )
            except Exception as e:
                skipped_rebuild += 1
                if not first_rebuild_error:
                    first_rebuild_error = repr(e)
                continue
            # mask_scope: all => nc/ctx 都重算
            # ctx_only => 只重算 ctx，nc 用 baseline
            if args.mask_scope == "all":
                ab_nc_scores = score_answer_candidates_by_mode(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    image=nc_image,
                    prompt=s["nc_prompt"],
                    gold=s["gold"],
                    wrong=s["wrong"],
                    unknown=s.get("unknown", "unknown"),
                    trace_mode=args.trace_mode,
                    scoring_mode=args.scoring_mode,
                    device=device,
                )
                ab_nc_metrics = compute_scalar_metrics(ab_nc_scores, args.trace_mode)
                ab_nc_pred = pred_label(ab_nc_scores)
            else:
                ab_nc_scores = copy.deepcopy(s["base_nc_scores"])
                ab_nc_metrics = copy.deepcopy(s["base_nc_metrics"])
                ab_nc_pred = s["base_nc_pred"]
            ab_nc_gold_margin = nc_gold_margin(ab_nc_scores)

            ab_ctx_scores = score_answer_candidates_by_mode(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                image=ctx_image,
                prompt=s["ctx_prompt"],
                gold=s["gold"],
                wrong=s["wrong"],
                unknown=s.get("unknown", "unknown"),
                trace_mode=args.trace_mode,
                scoring_mode=args.scoring_mode,
                device=device,
            )
            ab_ctx_metrics = compute_scalar_metrics(ab_ctx_scores, args.trace_mode)
            ab_ctx_pred = pred_label(ab_ctx_scores)
            ctx_follow_context_gain = float(ab_ctx_metrics["follow_context"] - s["base_ctx_metrics"]["follow_context"])
            nc_gold_margin_damage = float(max(0.0, s["base_nc_gold_margin"] - ab_nc_gold_margin))
            hallucination_relief = float(ctx_follow_context_gain - nc_gold_margin_damage)

            metric_effect_reduction_fixed_nc = {}
            metric_effect_reduction_mask_scope = {}
            metric_abs_base_change = {}
            metric_abs_ctx_change = {}

            for m in metrics:
                metric_effect_reduction_fixed_nc[m] = float(ab_ctx_metrics[m] - s["base_ctx_metrics"][m])
                metric_effect_reduction_mask_scope[m] = float(ab_ctx_metrics[m] - s["base_ctx_metrics"][m])
                metric_abs_base_change[m] = float(abs(ab_nc_metrics[m] - s["base_nc_metrics"][m]))
                metric_abs_ctx_change[m] = float(abs(ab_ctx_metrics[m] - s["base_ctx_metrics"][m]))

            rec = {
                "row_idx": s["row_idx"],
                "gold": s["gold"],
                "wrong": s["wrong"],
                "base_nc_pred": s["base_nc_pred"],
                "base_ctx_pred": s["base_ctx_pred"],
                "ab_nc_pred": ab_nc_pred,
                "ab_ctx_pred": ab_ctx_pred,

                "base_nc_scores": s["base_nc_scores"],
                "base_ctx_scores": s["base_ctx_scores"],
                "ab_nc_scores": ab_nc_scores,
                "ab_ctx_scores": ab_ctx_scores,

                "base_nc_metrics": s["base_nc_metrics"],
                "base_ctx_metrics": s["base_ctx_metrics"],
                "ab_nc_metrics": ab_nc_metrics,
                "ab_ctx_metrics": ab_ctx_metrics,
                "base_nc_gold_margin": s["base_nc_gold_margin"],
                "ab_nc_gold_margin": ab_nc_gold_margin,
                "ctx_follow_context_gain": ctx_follow_context_gain,
                "nc_gold_margin_damage": nc_gold_margin_damage,
                "hallucination_relief": hallucination_relief,

                "metric_effect_reduction_fixed_nc": metric_effect_reduction_fixed_nc,
                "metric_effect_reduction_mask_scope": metric_effect_reduction_mask_scope,
                "metric_abs_base_change": metric_abs_base_change,
                "metric_abs_ctx_change": metric_abs_ctx_change,
            }
            records.append(rec)
            tracker.append_record("records.jsonl", rec)
            tracker.mark_done(sample_key, {"row_idx": int(s["row_idx"])})
            tracker.update(completed=len(records))

            maybe_empty_cuda_cache(len(records))

    finally:
        remove_handles(handles)

    # summary
    all_relief = [r["hallucination_relief"] for r in records]
    all_ctx_gain = [r["ctx_follow_context_gain"] for r in records]
    all_nc_damage = [r["nc_gold_margin_damage"] for r in records]

    base_nc_correct = 0
    base_ctx_correct = 0
    ab_nc_correct = 0
    ab_ctx_correct = 0
    base_ctx_unknown = 0
    ab_ctx_unknown = 0

    for r in records:
        # gold 视为正确
        if r["base_nc_pred"] == "gold":
            base_nc_correct += 1
        if r["base_ctx_pred"] == "gold":
            base_ctx_correct += 1
        if r["ab_nc_pred"] == "gold":
            ab_nc_correct += 1
        if r["ab_ctx_pred"] == "gold":
            ab_ctx_correct += 1
        if r["base_ctx_pred"] == "unknown":
            base_ctx_unknown += 1
        if r["ab_ctx_pred"] == "unknown":
            ab_ctx_unknown += 1

    n = len(records)
    metric_summary = summarize_metric_changes(records, metrics)

    summary = {
        "config": {
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "model": args.model,
            "mode": args.mode,
            "selected_heads": args.selected_heads,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "mask_scale": float(args.mask_scale),
            "metrics": metrics,
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "dtype": args.dtype,
            "device": device,
            "max_examples": args.max_examples,
            "max_image_side": args.max_image_side,
            "random_ablate": bool(args.random_ablate),
            "random_seed": args.random_seed,
            "sample_filter": args.sample_filter,
            "scoring_mode": args.scoring_mode,
        },
        "selected_pairs": [{"layer": l, "head": h} for l, h in selected_pairs],
        "n_selected_heads": len(selected_pairs),
        "mode": args.mode,
        "n_samples": n,
        "sample_filter": args.sample_filter,
        "mask_scale": float(args.mask_scale),
        "mean_hallucination_relief": safe_mean(all_relief),
        "mean_ic_follow_context_gain": safe_mean(all_ctx_gain),
        "mean_nc_gold_margin_damage": safe_mean(all_nc_damage),

        "pred_nc": {
            "base_acc": float(base_nc_correct / n) if n > 0 else 0.0,
            "ab_acc": float(ab_nc_correct / n) if n > 0 else 0.0,
            "acc_delta": float((ab_nc_correct - base_nc_correct) / n) if n > 0 else 0.0,
        },
        "pred_ctx": {
            "base_acc": float(base_ctx_correct / n) if n > 0 else 0.0,
            "ab_acc": float(ab_ctx_correct / n) if n > 0 else 0.0,
            "acc_delta": float((ab_ctx_correct - base_ctx_correct) / n) if n > 0 else 0.0,
        },
        "ctx_unknown": {
            "base_rate": float(base_ctx_unknown / n) if n > 0 else 0.0,
            "ab_rate": float(ab_ctx_unknown / n) if n > 0 else 0.0,
            "rate_delta": float((ab_ctx_unknown - base_ctx_unknown) / n) if n > 0 else 0.0,
        },

        "per_metric": metric_summary,
        "records": records,
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tracker.finish(out_json=args.out_json, n_samples=n, n_selected_heads=len(selected_pairs), mode=args.mode)

    if skipped_rebuild > 0:
        print(f"[WARN] Skipped {skipped_rebuild} samples during image rebuild before ablation.")
        if first_rebuild_error:
            print(f"[WARN] First rebuild error: {first_rebuild_error}")

    print(json.dumps({
        "saved_to": args.out_json,
        "n_selected_heads": len(selected_pairs),
        "n_samples": n,
        "mean_hallucination_relief": summary["mean_hallucination_relief"],
        "mean_ic_follow_context_gain": summary["mean_ic_follow_context_gain"],
        "mean_nc_gold_margin_damage": summary["mean_nc_gold_margin_damage"],
        "ctx_unknown_rate_delta": summary["ctx_unknown"]["rate_delta"],
        "pred_nc_acc_delta": summary["pred_nc"]["acc_delta"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
