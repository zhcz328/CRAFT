from __future__ import annotations
import os
import json
import argparse
import importlib.util
import gc
import random
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

BASE_RULE = (
    "Answer the question using the image and your general world knowledge.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)
SYSTEM_PROMPT = "You are a helpful visual question answering assistant."
EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
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


def move_inputs_to_cpu(obj):
    return move_to_device(obj, "cpu")


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


def resolve_attention_backend(ablation_impl: str, requested_backend: str) -> str:
    backend = str(requested_backend or "auto").strip().lower()
    if backend not in {"auto", "eager", "sdpa"}:
        raise ValueError(f"Unsupported attention backend: {requested_backend}")
    if backend != "auto":
        return backend
    return "eager" if ablation_impl == "mask" else "sdpa"


def _slice_batch_like(obj, indices, batch_size: int):
    if torch.is_tensor(obj):
        if obj.dim() > 0 and obj.shape[0] == batch_size:
            return obj[indices]
        return obj
    if isinstance(obj, tuple):
        return tuple(_slice_batch_like(x, indices, batch_size) for x in obj)
    if isinstance(obj, list):
        return [_slice_batch_like(x, indices, batch_size) for x in obj]
    if isinstance(obj, Mapping):
        return {k: _slice_batch_like(v, indices, batch_size) for k, v in obj.items()}
    return obj


def _concat_batch_like(items):
    if not items:
        return None
    first = items[0]
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    if isinstance(first, tuple):
        return tuple(_concat_batch_like([item[idx] for item in items]) for idx in range(len(first)))
    if isinstance(first, list):
        return [_concat_batch_like([item[idx] for item in items]) for idx in range(len(first))]
    if isinstance(first, Mapping):
        return {k: _concat_batch_like([item[k] for item in items]) for k in first}
    return first


class VisionCacheController:
    def __init__(self, encoder):
        self.encoder = encoder
        self.cache = {}
        self.current_keys = None
        self.handle = None
        self.original_forward = getattr(encoder, "forward", None)

    def install(self):
        if self.original_forward is None or self.handle is not None:
            return

        controller = self
        original_forward = self.original_forward

        def wrapped_forward(*args, **kwargs):
            current_keys = controller.current_keys
            if not current_keys:
                return original_forward(*args, **kwargs)

            batch_tensor = None
            if args and torch.is_tensor(args[0]):
                batch_tensor = args[0]
            else:
                for value in kwargs.values():
                    if torch.is_tensor(value):
                        batch_tensor = value
                        break
            if batch_tensor is None or batch_tensor.dim() == 0:
                return original_forward(*args, **kwargs)

            batch_size = int(batch_tensor.shape[0])
            keys = list(current_keys) if isinstance(current_keys, (list, tuple)) else [current_keys] * batch_size
            if len(keys) != batch_size:
                return original_forward(*args, **kwargs)

            sample_outputs = [None] * batch_size
            missing_by_key = {}
            for idx, raw_key in enumerate(keys):
                if not raw_key:
                    missing_by_key.setdefault((None, idx), []).append(idx)
                    continue
                key = str(raw_key)
                if key in controller.cache:
                    sample_outputs[idx] = controller.cache[key]
                else:
                    missing_by_key.setdefault(key, []).append(idx)

            for key, idxs in missing_by_key.items():
                idx_tensor = torch.tensor(idxs, device=batch_tensor.device, dtype=torch.long)
                sub_args = tuple(_slice_batch_like(arg, idx_tensor, batch_size) for arg in args)
                sub_kwargs = {k: _slice_batch_like(v, idx_tensor, batch_size) for k, v in kwargs.items()}
                sub_out = original_forward(*sub_args, **sub_kwargs)
                for pos, sample_idx in enumerate(idxs):
                    sample_out = _slice_batch_like(sub_out, slice(pos, pos + 1), len(idxs))
                    sample_outputs[sample_idx] = sample_out
                    if key is not None:
                        controller.cache[str(keys[sample_idx])] = sample_out

            return _concat_batch_like(sample_outputs)

        self.encoder.forward = wrapped_forward
        self.handle = True

    def begin(self, cache_keys):
        self.current_keys = cache_keys

    def end(self):
        self.current_keys = None


def get_vision_cache_controller(model):
    m = model.module if hasattr(model, "module") else model
    controller = getattr(m, "_codex_vision_cache_controller", None)
    if controller is not None:
        return controller
    try:
        core = m.get_model() if hasattr(m, "get_model") else m
        if hasattr(core, "get_vision_encoder"):
            encoder = core.get_vision_encoder()
            controller = VisionCacheController(encoder)
            controller.install()
            setattr(m, "_codex_vision_cache_controller", controller)
            return controller
    except Exception:
        pass
    return None


def infer_model_type(model) -> str:
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        cfg = getattr(cand, "config", None)
        model_type = str(getattr(cfg, "model_type", "")).strip().lower()
        if model_type:
            return model_type
    return ""


def is_qwen_headscan_accel_model(model) -> bool:
    return infer_model_type(model) in {"qwen3_5", "qwen3_vl"}


def resolve_qwen_prefill_batch_size(model, requested_batch_size: int = 0) -> int:
    if requested_batch_size and requested_batch_size > 0:
        return int(requested_batch_size)
    model_type = infer_model_type(model)
    if model_type == "qwen3_vl":
        return 2
    if model_type == "qwen3_5":
        return 4
    return 1


def get_processor_pad_token_id(processor) -> int:
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        return 0
    return int(pad_token_id)


def pad_and_stack_tensors(tensors, key: str, pad_token_id: int):
    sample = tensors[0]
    if sample.ndim < 2 or any(t.shape[0] != sample.shape[0] for t in tensors):
        return torch.cat(tensors, dim=0)

    if any(t.shape[2:] != sample.shape[2:] for t in tensors):
        return torch.cat(tensors, dim=0)

    max_len = max(t.shape[1] for t in tensors)
    if all(t.shape[1] == max_len for t in tensors):
        return torch.cat(tensors, dim=0)

    if key == "attention_mask":
        pad_value = 0
    elif key in {"position_ids", "cache_position"}:
        pad_value = 0
    elif key == "labels":
        pad_value = -100
    else:
        pad_value = pad_token_id

    padded = []
    for tensor in tensors:
        if tensor.shape[1] == max_len:
            padded.append(tensor)
            continue
        pad_shape = list(tensor.shape)
        pad_shape[1] = max_len - tensor.shape[1]
        pad_tensor = torch.full(
            pad_shape,
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded.append(torch.cat([tensor, pad_tensor], dim=1))
    return torch.cat(padded, dim=0)


def collate_processor_outputs(outputs, pad_token_id: int):
    collated = {}
    for key in outputs[0]:
        values = [output[key] for output in outputs]
        if torch.is_tensor(values[0]):
            collated[key] = pad_and_stack_tensors(values, key, pad_token_id)
        else:
            collated[key] = values
    return collated


def sanitize_qwen_batched_prefill_inputs(batch_inputs):
    if "attention_mask" not in batch_inputs:
        return batch_inputs

    sanitized = dict(batch_inputs)
    attn = sanitized["attention_mask"]
    if torch.is_tensor(attn) and attn.dim() == 2:
        pos = attn.to(dtype=torch.long).cumsum(dim=1) - 1
        pos.masked_fill_(attn <= 0, 0)
        if "position_ids" in sanitized:
            pos_ids = sanitized["position_ids"]
            if torch.is_tensor(pos_ids) and pos_ids.dim() == 2 and pos_ids.shape == pos.shape:
                sanitized["position_ids"] = pos
            else:
                sanitized.pop("position_ids", None)
        sanitized.pop("cache_position", None)
    return sanitized


def tokenize_continuation_ids(processor, continuation: str) -> torch.Tensor:
    return processor.tokenizer(
        continuation,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]


def model_prefers_cache_object(model) -> bool:
    model_type = infer_model_type(model)
    return model_type.startswith("qwen3")


def get_qwen_generation_model(model):
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        if hasattr(cand, "prepare_inputs_for_generation"):
            return cand
        inner = getattr(cand, "model", None)
        if inner is not None and hasattr(inner, "prepare_inputs_for_generation"):
            return inner
    return None


def get_qwen_core_model(model):
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        inner = getattr(cand, "model", None)
        if inner is not None and hasattr(inner, "rope_deltas"):
            return inner
        if hasattr(cand, "rope_deltas"):
            return cand
    return None


def set_qwen_rope_deltas(model, rope_deltas) -> None:
    if rope_deltas is None:
        return
    core = get_qwen_core_model(model)
    if core is not None and hasattr(core, "rope_deltas"):
        core.rope_deltas = rope_deltas


def build_qwen_continuation_kwargs(model, cache_pack, continuation_ids, attention_mask, past_key_values, use_cache):
    generation_model = get_qwen_generation_model(model)
    if generation_model is None:
        return {
            "input_ids": continuation_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_hidden_states": False,
        }

    set_qwen_rope_deltas(model, cache_pack.get("rope_deltas"))
    prompt_lengths = cache_pack.get("prompt_lengths")
    if prompt_lengths is None:
        prompt_lengths = attention_mask.to(dtype=torch.long).sum(dim=1) - continuation_ids.shape[1]
    if not torch.is_tensor(prompt_lengths):
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=continuation_ids.device)
    prompt_lengths = prompt_lengths.to(device=continuation_ids.device, dtype=torch.long).view(-1)
    if prompt_lengths.numel() == 0:
        prompt_start = 0
    else:
        prompt_start = int(prompt_lengths[0].item())
        if torch.any(prompt_lengths != prompt_start):
            raise RuntimeError("Qwen batched continuation requires uniform prompt lengths within a batch.")
    cache_position = torch.arange(
        prompt_start,
        prompt_start + int(continuation_ids.shape[1]),
        device=continuation_ids.device,
        dtype=torch.long,
    )
    prepared = generation_model.prepare_inputs_for_generation(
        continuation_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        cache_position=cache_position,
        use_cache=use_cache,
    )
    prepared = dict(prepared)
    prepared.pop("token_type_ids", None)
    prepared["past_key_values"] = past_key_values
    prepared["attention_mask"] = attention_mask
    prepared["cache_position"] = cache_position
    prepared["use_cache"] = use_cache
    prepared["output_hidden_states"] = False
    prepared.setdefault("pixel_values", None)
    prepared.setdefault("pixel_values_videos", None)
    return prepared


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


def _slice_nested_cache_item(item, batch_idx: int, batch_size: int):
    if torch.is_tensor(item):
        if item.dim() > 0 and item.shape[0] == batch_size:
            return item[batch_idx : batch_idx + 1]
        return item
    if isinstance(item, tuple):
        return tuple(_slice_nested_cache_item(x, batch_idx, batch_size) for x in item)
    if isinstance(item, list):
        return [_slice_nested_cache_item(x, batch_idx, batch_size) for x in item]
    if isinstance(item, Mapping):
        return {k: _slice_nested_cache_item(v, batch_idx, batch_size) for k, v in item.items()}
    return item


def slice_past_key_values(past_key_values, batch_idx: int, batch_size: int):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "batch_select_indices") and callable(getattr(past_key_values, "batch_select_indices")):
        for candidate in ([batch_idx], torch.tensor([batch_idx], dtype=torch.long)):
            try:
                return past_key_values.batch_select_indices(candidate)
            except Exception:
                continue
    if hasattr(past_key_values, "batch_split") and callable(getattr(past_key_values, "batch_split")):
        try:
            splits = past_key_values.batch_split(batch_size, 1)
            if isinstance(splits, list) and 0 <= batch_idx < len(splits):
                return splits[batch_idx]
        except Exception:
            pass
    if hasattr(past_key_values, "to_legacy_cache") and callable(getattr(past_key_values, "to_legacy_cache")):
        try:
            past_key_values = past_key_values.to_legacy_cache()
        except Exception:
            pass
    return _slice_nested_cache_item(past_key_values, batch_idx, batch_size)


def slice_cache_pack(cache_pack, batch_idx: int):
    batch_size = int(cache_pack["prompt_last_logits"].shape[0])
    sample_cache = {
        "attn_prompt": cache_pack["attn_prompt"][batch_idx : batch_idx + 1],
        "past_key_values": slice_past_key_values(cache_pack["past_key_values"], batch_idx, batch_size),
        "prompt_last_logits": cache_pack["prompt_last_logits"][batch_idx : batch_idx + 1],
    }
    for key, value in cache_pack.items():
        if key in sample_cache:
            continue
        sample_cache[key] = _slice_nested_cache_item(value, batch_idx, batch_size)
    return sample_cache


def is_cuda_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def build_token_batch(token_batches, pad_token_id: int):
    if not token_batches:
        return None, None
    max_len = max(int(tokens.shape[1]) for tokens in token_batches)
    batch_size = len(token_batches)
    token_ids = torch.full((batch_size, max_len), pad_token_id, dtype=token_batches[0].dtype)
    token_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for idx, tokens in enumerate(token_batches):
        seq_len = int(tokens.shape[1])
        token_ids[idx, :seq_len] = tokens[0]
        token_mask[idx, :seq_len] = 1
    return token_ids, token_mask


@torch.inference_mode()
def logprob_from_cache_batch_mm(model, processor, cache_pack, continuation_ids_batch) -> List[float]:
    if not continuation_ids_batch:
        return []

    pad_token_id = get_processor_pad_token_id(processor)
    cont_ids, cont_mask = build_token_batch(continuation_ids_batch, pad_token_id)
    cont_ids = cont_ids.to(model.device)
    cont_mask = cont_mask.to(model.device)
    valid_lengths = cont_mask.sum(dim=1)
    if int(valid_lengths.max().item()) <= 0:
        return [float("-inf")] * len(continuation_ids_batch)

    attn_prompt = cache_pack["attn_prompt"]
    past_key_values = normalize_past_key_values_for_model(model, cache_pack["past_key_values"])
    prompt_last_logits = cache_pack["prompt_last_logits"]

    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)
    totals = log_probs0.gather(-1, cont_ids[:, :1]).squeeze(-1)
    totals = totals.masked_fill(valid_lengths <= 0, float("-inf"))

    cont_len = int(cont_ids.shape[1])
    if cont_len == 1:
        return [float(x) for x in totals]

    inp = cont_ids[:, :-1]
    inp_mask = cont_mask[:, :-1]
    target_ids = cont_ids[:, 1:]
    target_mask = cont_mask[:, 1:]

    if int(target_mask.sum().item()) == 0:
        return [float(x) for x in totals]

    attn_suffix = inp_mask.to(dtype=attn_prompt.dtype)
    attn_full = torch.cat([attn_prompt, attn_suffix], dim=1)
    if is_qwen_headscan_accel_model(model):
        model_kwargs = build_qwen_continuation_kwargs(
            model,
            cache_pack,
            inp,
            attn_full,
            past_key_values,
            use_cache=False,
        )
    else:
        model_kwargs = {
            "input_ids": inp,
            "attention_mask": attn_full,
            "past_key_values": past_key_values,
            "use_cache": False,
            "output_hidden_states": False,
        }
    out = model(**model_kwargs)
    log_probs = torch.log_softmax(out.logits, dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * target_mask.to(dtype=gathered.dtype)
    totals = totals + gathered.sum(dim=1)
    return [float(x) for x in totals]


def score_two_options_from_cache_batch_mm(model, processor, cache_pack, batch_samples):
    gold_scores = logprob_from_cache_batch_mm(
        model,
        processor,
        cache_pack,
        [sample["gold_token_ids"] for sample in batch_samples],
    )
    wrong_scores = logprob_from_cache_batch_mm(
        model,
        processor,
        cache_pack,
        [sample["wrong_token_ids"] for sample in batch_samples],
    )
    results = []
    for sample, gold_lp, wrong_lp in zip(batch_samples, gold_scores, wrong_scores):
        results.append(
            {
                "gold_lp": gold_lp,
                "wrong_lp": wrong_lp,
                "pred": sample["gold"] if gold_lp >= wrong_lp else sample["wrong"],
            }
        )
    return results


def build_qwen_batch_signature(sample_inputs) -> tuple:
    sig = []
    for key in sorted(sample_inputs.keys()):
        value = sample_inputs[key]
        if torch.is_tensor(value):
            sig.append((key, tuple(value.shape), str(value.dtype)))
    attn_mask = sample_inputs.get("attention_mask")
    if torch.is_tensor(attn_mask):
        sig.append(("attention_tokens", int(attn_mask.to(dtype=torch.long).sum().item())))
    return tuple(sig)


def score_samples_for_prompt_keys_qwen(
    model,
    processor,
    samples,
    prompt_keys: List[str],
    requested_batch_size: int = 0,
    progress_desc: Optional[str] = None,
    ablation_controller: Optional[HeadAblationController] = None,
):
    if not samples:
        return {key: [] for key in prompt_keys}

    pad_token_id = get_processor_pad_token_id(processor)
    target_batch_size = resolve_qwen_prefill_batch_size(model, requested_batch_size)
    all_scores = {key: [None] * len(samples) for key in prompt_keys}
    progress_bar = None
    if progress_desc:
        progress_bar = tqdm(total=len(samples) * len(prompt_keys), desc=progress_desc, unit="sample")

    try:
        for prompt_key in prompt_keys:
            buckets = {}
            for sample_idx, sample in enumerate(samples):
                sig = (
                    build_qwen_batch_signature(sample[prompt_key]),
                    int(sample["gold_token_ids"].shape[1]),
                    int(sample["wrong_token_ids"].shape[1]),
                )
                buckets.setdefault(sig, []).append(sample_idx)

            for bucket_indices in buckets.values():
                start = 0
                while start < len(bucket_indices):
                    current_batch_size = min(target_batch_size, len(bucket_indices) - start)
                    while True:
                        chunk_indices = bucket_indices[start : start + current_batch_size]
                        batch_samples = [samples[idx] for idx in chunk_indices]
                        try:
                            if ablation_controller is not None:
                                batch_head_indices = [sample.get("_codex_head_idx", -1) for sample in batch_samples]
                                if any(int(x) >= 0 for x in batch_head_indices):
                                    ablation_controller.set_batch_head_indices(batch_head_indices)
                                else:
                                    ablation_controller.clear_batch_head_indices()
                            collated_inputs = collate_processor_outputs(
                                [sample[prompt_key] for sample in batch_samples],
                                pad_token_id,
                            )
                            collated_inputs = sanitize_qwen_batched_prefill_inputs(collated_inputs)
                            cache_pack = build_prompt_cache_from_inputs(model, collated_inputs)
                            batch_scores = score_two_options_from_cache_batch_mm(
                                model,
                                processor,
                                cache_pack,
                                batch_samples,
                            )
                            for sample_idx, sample_scores in zip(chunk_indices, batch_scores):
                                all_scores[prompt_key][sample_idx] = sample_scores
                            start += current_batch_size
                            if progress_bar is not None:
                                progress_bar.update(current_batch_size)
                            break
                        except Exception as exc:
                            exc_text = str(exc).lower()
                            should_shrink = current_batch_size > 1 and (
                                is_cuda_oom_error(exc)
                                or "cache_position" in exc_text
                                or "prepare_inputs_for_generation" in exc_text
                                or isinstance(exc, TypeError)
                            )
                            if should_shrink:
                                current_batch_size = max(1, current_batch_size // 2)
                                if is_cuda_oom_error(exc) and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            raise
                        finally:
                            if ablation_controller is not None:
                                ablation_controller.clear_batch_head_indices()
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return {key: list(values) for key, values in all_scores.items()}


def attach_sample_baseline_metrics(
    sample: dict,
    nc_scores: dict,
    ctx_scores: dict,
    metrics: List[str],
    weights: List[float],
    trace_mode: str,
) -> None:
    gold = sample["gold"]
    wrong = sample["wrong"]
    base_nc_scalar = weighted_metric(metrics, weights, nc_scores, gold, wrong, trace_mode)
    base_ctx_scalar = weighted_metric(metrics, weights, ctx_scores, gold, wrong, trace_mode)
    base_effect_scalar = base_ctx_scalar - base_nc_scalar

    sample["nc_scores"] = nc_scores
    sample["ctx_scores"] = ctx_scores
    sample["base_nc_scalar"] = base_nc_scalar
    sample["base_ctx_scalar"] = base_ctx_scalar
    sample["base_effect_scalar"] = base_effect_scalar
    sample["base_metric_values"] = {
        m: {
            "nc": metric_from_scores(m, nc_scores, gold, wrong, trace_mode),
            "ctx": metric_from_scores(m, ctx_scores, gold, wrong, trace_mode),
            "effect": metric_from_scores(m, ctx_scores, gold, wrong, trace_mode)
                      - metric_from_scores(m, nc_scores, gold, wrong, trace_mode),
        }
        for m in metrics
    }


def build_head_batch_groups(pending_heads, head_batch_size: int):
    if head_batch_size <= 1:
        return [[(int(layer_idx), int(head_idx))] for layer_idx, head_idx in pending_heads]
    by_layer = {}
    for layer_idx, head_idx in pending_heads:
        by_layer.setdefault(int(layer_idx), []).append(int(head_idx))
    groups = []
    for layer_idx in sorted(by_layer):
        heads = sorted(by_layer[layer_idx])
        for start in range(0, len(heads), head_batch_size):
            chunk = heads[start : start + head_batch_size]
            groups.append([(int(layer_idx), int(head_idx)) for head_idx in chunk])
    return groups


def expand_samples_for_head_group(samples, head_group):
    expanded = []
    head_order = []
    for layer_idx, head_idx in head_group:
        head_order.append((int(layer_idx), int(head_idx)))
        for sample in samples:
            copied = dict(sample)
            copied["_codex_head_idx"] = int(head_idx)
            expanded.append(copied)
    return expanded, head_order


def summarize_head_group_scores(head_group, expanded_scores, source_samples, metrics, weights, trace_mode, include_nc: bool):
    out = {}
    sample_count = len(source_samples)
    for head_pos, (layer_idx, head_idx) in enumerate(head_group):
        start = head_pos * sample_count
        end = start + sample_count
        chunk_nc = expanded_scores.get("nc_inputs", [None] * len(expanded_scores.get("ctx_inputs", [])))[start:end]
        chunk_ctx = expanded_scores["ctx_inputs"][start:end]
        eff_red_sum = 0.0
        base_change_sum = 0.0
        metric_eff_red = {m: 0.0 for m in metrics}
        metric_base_change = {m: 0.0 for m in metrics}
        for sample_idx, sample in enumerate(source_samples):
            gold = sample["gold"]
            wrong = sample["wrong"]
            ctx_ab_scores = chunk_ctx[sample_idx]
            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, trace_mode)
            if include_nc:
                nc_ab_scores = chunk_nc[sample_idx]
                nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, trace_mode)
                base_change_sum += abs(nc_ab_scalar - sample["base_nc_scalar"])
            else:
                nc_ab_scores = None
                nc_ab_scalar = sample["base_nc_scalar"]
            effect_ab = ctx_ab_scalar - nc_ab_scalar
            eff_red_sum += abs(sample["base_effect_scalar"]) - abs(effect_ab)
            if include_nc:
                for m in metrics:
                    nc_m_base = sample["base_metric_values"][m]["nc"]
                    ctx_m_ab = metric_from_scores(m, ctx_ab_scores, gold, wrong, trace_mode)
                    nc_m_ab = metric_from_scores(m, nc_ab_scores, gold, wrong, trace_mode)
                    effect_m_ab = ctx_m_ab - nc_m_base
                    metric_eff_red[m] += abs(sample["base_metric_values"][m]["effect"]) - abs(effect_m_ab)
                    metric_base_change[m] += abs(nc_m_ab - nc_m_base)
        out[(int(layer_idx), int(head_idx))] = {
            "coarse_mean_abs_effect_reduction": eff_red_sum / max(sample_count, 1),
            "coarse_mean_abs_base_change": base_change_sum / max(sample_count, 1),
            "mean_abs_effect_reduction": eff_red_sum / max(sample_count, 1),
            "mean_abs_base_change": base_change_sum / max(sample_count, 1),
            "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(sample_count, 1) for m in metrics},
            "metric_mean_abs_base_change": {m: metric_base_change[m] / max(sample_count, 1) for m in metrics},
        }
    return out


def resolve_group_eval_batch_size(model, requested_batch_size: int, head_group_size: int = 1) -> int:
    base = resolve_qwen_prefill_batch_size(model, requested_batch_size)
    return max(int(base), int(head_group_size), 1)


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


def build_nc_prompt(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def build_ctx_prompt(question: str, evidence: str, position: str = "before_question") -> str:
    base = build_nc_prompt(question)
    ev = EVIDENCE_TMPL.format(ans=evidence)
    if position == "prefix":
        return ev + "\n" + base
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base:
            raise ValueError("base prompt missing '\\nQuestion:' marker")
        return base.replace(marker, "\n" + ev + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base:
            raise ValueError("base prompt missing '\\nAnswer:' marker")
        return base.replace(marker, "\n" + ev + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


def resize_image_max_side(image: Image.Image, max_side: int = 672) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def resolve_image_path(row, image_root: str) -> Optional[Path]:
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        candidates.extend(build_path_candidates(row["image_path"], image_root))

    if "img_name" in row and pd.notna(row["img_name"]):
        candidates.extend(build_path_candidates(row["img_name"], image_root))

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


def validate_shard_args(num_shards: int, shard_idx: int) -> None:
    if num_shards <= 0:
        raise SystemExit("--num_shards must be >= 1")
    if not (0 <= shard_idx < num_shards):
        raise SystemExit("--shard_idx must satisfy 0 <= shard_idx < num_shards")


def build_shard_token(shard_idx: int, num_shards: int) -> str:
    return f"shard{int(shard_idx):02d}of{int(num_shards):02d}"


def build_shard_file_path(path_value: str, shard_idx: int, num_shards: int) -> str:
    path = Path(path_value)
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    shard_dir = path.parent / f"{stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return str(shard_dir / f"{stem}.{build_shard_token(shard_idx, num_shards)}{suffix}")


def iter_heads_for_shard(all_heads, num_shards: int, shard_idx: int):
    for head_pos, head_item in enumerate(all_heads):
        if head_pos % num_shards == shard_idx:
            yield head_item


def build_stage_file_path(path_value: str, stage: str) -> str:
    path = Path(path_value)
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    return str(path.parent / f"{stem}.{stage}{suffix}")


def build_stage_shard_file_path(path_value: str, stage: str, shard_idx: int, num_shards: int) -> str:
    base_stage_path = build_stage_file_path(path_value, stage)
    return build_shard_file_path(base_stage_path, shard_idx, num_shards)


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


def load_model_and_processor(model_name: str, device: str, dtype: str, attn_backend: str = "eager"):
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
        attn_implementation=attn_backend,
    )
    model.eval()
    return model, processor


@torch.inference_mode()
def build_inputs_mm(processor, prompt: str, image: Image.Image, device, vision_cache_key: str = ""):
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
    if vision_cache_key:
        base_inputs["codex_vision_cache_key"] = vision_cache_key
    base_inputs = move_to_device(base_inputs, device)
    return base_inputs


@torch.inference_mode()
def build_prompt_cache_from_inputs(model, base_inputs):
    base_inputs = dict(base_inputs)
    vision_cache_keys = base_inputs.pop("codex_vision_cache_key", None)
    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)
    vision_cache_controller = get_vision_cache_controller(model)
    if vision_cache_controller is not None and vision_cache_keys:
        vision_cache_controller.begin(vision_cache_keys)
    try:
        out = model(**fixed_inputs, use_cache=True, output_hidden_states=False)
    finally:
        if vision_cache_controller is not None:
            vision_cache_controller.end()

    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    if torch.is_tensor(attn_prompt) and attn_prompt.dim() == 2:
        prompt_positions = attn_prompt.to(dtype=torch.long).sum(dim=1).clamp(min=1) - 1
    else:
        prompt_positions = torch.full(
            (input_ids_prompt.shape[0],),
            int(input_ids_prompt.shape[1]) - 1,
            dtype=torch.long,
            device=out.logits.device,
        )
    batch_indices = torch.arange(out.logits.shape[0], device=out.logits.device)
    prompt_last_logits = out.logits[batch_indices, prompt_positions, :].detach()

    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
        "prompt_lengths": prompt_positions + 1,
        "rope_deltas": getattr(out, "rope_deltas", None),
    }


@torch.inference_mode()
def logprob_from_cache_mm(model, processor, cache_pack, continuation: str = "", continuation_ids=None) -> float:
    if continuation_ids is None:
        tok = processor.tokenizer
        cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
    else:
        cont_ids = continuation_ids
    cont_ids = cont_ids.to(model.device)
    if cont_ids.numel() == 0:
        return float("-inf")

    attn_prompt = cache_pack["attn_prompt"]
    past_key_values = normalize_past_key_values_for_model(model, cache_pack["past_key_values"])
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
def score_two_options_from_cache(
    model,
    processor,
    cache_pack,
    gold: str,
    wrong: str,
    gold_token_ids=None,
    wrong_token_ids=None,
):
    gold_lp = logprob_from_cache_mm(
        model,
        processor,
        cache_pack,
        " " + gold,
        continuation_ids=gold_token_ids,
    )
    wrong_lp = logprob_from_cache_mm(
        model,
        processor,
        cache_pack,
        " " + wrong,
        continuation_ids=wrong_token_ids,
    )
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "pred": gold if gold_lp >= wrong_lp else wrong,
    }


# -------------------------
# metrics
# -------------------------
def metric_from_scores(metric: str, scores: dict, gold: str, wrong: str, trace_mode: str) -> float:
    if metric == "gold_wrong_margin":
        return float(scores["gold_lp"] - scores["wrong_lp"])

    if metric == "follow_conflict":
        return float(scores["wrong_lp"] - scores["gold_lp"])

    ctx_target = gold if trace_mode == "cc" else wrong
    other_target = wrong if ctx_target == gold else gold

    ctx_lp = float(scores["gold_lp"] if ctx_target == gold else scores["wrong_lp"])
    other_lp = float(scores["wrong_lp"] if other_target == wrong else scores["gold_lp"])

    if metric == "follow_context":
        return ctx_lp - other_lp

    if metric == "context_flip":
        return other_lp - ctx_lp

    raise ValueError(
        f"Unknown metric: {metric}. Allowed: ['gold_wrong_margin', 'follow_conflict', 'follow_context', 'context_flip']"
    )


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


def _get_attn_head_geometry(attn_mod, n_heads_cfg: int):
    num_heads = int(getattr(attn_mod, "num_heads", None) or getattr(attn_mod, "num_attention_heads", None) or n_heads_cfg)
    q_proj = getattr(attn_mod, "q_proj", None)
    v_proj = getattr(attn_mod, "v_proj", None)
    o_proj = getattr(attn_mod, "o_proj", None)
    head_dim = getattr(attn_mod, "head_dim", None)
    if head_dim is None and q_proj is not None and hasattr(q_proj, "out_features"):
        q_out = int(getattr(q_proj, "out_features"))
        if num_heads > 0 and q_out > 0 and q_out % num_heads == 0:
            head_dim = q_out // num_heads
    if head_dim is None and o_proj is not None and hasattr(o_proj, "in_features"):
        o_in = int(getattr(o_proj, "in_features"))
        if num_heads > 0 and o_in > 0 and o_in % num_heads == 0:
            head_dim = o_in // num_heads
    if head_dim is None:
        hidden_size = getattr(attn_mod, "hidden_size", None)
        if hidden_size is not None:
            hidden_size = int(hidden_size)
            if num_heads > 0 and hidden_size % num_heads == 0:
                head_dim = hidden_size // num_heads
    if head_dim is None or int(head_dim) <= 0:
        raise RuntimeError("Cannot infer attention head_dim for output ablation hook.")
    num_kv_heads = getattr(attn_mod, "num_key_value_heads", None) or getattr(attn_mod, "num_kv_heads", None)
    if num_kv_heads is None and v_proj is not None and hasattr(v_proj, "out_features"):
        v_out = int(getattr(v_proj, "out_features"))
        if v_out > 0 and v_out % int(head_dim) == 0:
            num_kv_heads = v_out // int(head_dim)
    num_kv_heads = int(num_kv_heads or num_heads)
    num_kv_groups = max(1, num_heads // max(num_kv_heads, 1))
    return num_heads, int(head_dim), num_kv_heads, num_kv_groups


class HeadAblationController:
    def __init__(self, model, layer2heads: Dict[int, List[int]], keep_mode: str = "self", impl: str = "mask"):
        self.model = model
        self.layer2heads = {int(k): sorted(set(int(h) for h in v)) for k, v in layer2heads.items()}
        self.keep_mode = keep_mode
        self.impl = impl
        self.layer2attn, _ = _find_self_attn_modules(model)
        self.n_heads_cfg, _ = _get_num_heads_and_hidden(model)
        if self.n_heads_cfg is None or self.n_heads_cfg <= 0:
            raise RuntimeError("Cannot read num_attention_heads from model.config.")
        self.batch_head_indices = None
        self.handles = []

    def set_batch_head_indices(self, head_indices):
        if head_indices is None:
            self.batch_head_indices = None
            return
        if torch.is_tensor(head_indices):
            self.batch_head_indices = head_indices.detach().to(dtype=torch.long).cpu()
        else:
            self.batch_head_indices = torch.tensor(list(head_indices), dtype=torch.long)

    def clear_batch_head_indices(self):
        self.batch_head_indices = None

    def _resolve_batch_heads(self, batch_size: int, device):
        if self.batch_head_indices is None:
            return None
        if int(self.batch_head_indices.numel()) != int(batch_size):
            return None
        return self.batch_head_indices.to(device=device, dtype=torch.long)

    def install(self):
        for layer_idx, heads in self.layer2heads.items():
            if layer_idx not in self.layer2attn:
                continue
            attn_mod = self.layer2attn[layer_idx]
            if self.impl == "output":
                self.handles.extend(self._install_output_hooks(attn_mod, heads))
            else:
                self.handles.append(self._install_mask_hook(attn_mod, heads))
        return self.handles

    def _install_mask_hook(self, attn_mod, heads):
        n_heads_cfg = self.n_heads_cfg
        keep_mode = self.keep_mode
        controller = self
        heads = sorted(set(int(h) for h in heads))

        def pre_hook(module, args, kwargs):
            kwargs = {} if kwargs is None else dict(kwargs)
            attn_mask = None
            from_kwargs = False
            if "attention_mask" in kwargs:
                attn_mask = kwargs["attention_mask"]
                from_kwargs = True
            elif len(args) >= 2:
                attn_mask = args[1]
            if attn_mask is None or not torch.is_tensor(attn_mask) or attn_mask.dim() != 4:
                return args, kwargs

            batch_heads = controller._resolve_batch_heads(int(attn_mask.shape[0]), attn_mask.device)
            if batch_heads is None:
                new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads, keep_mode=keep_mode)
            else:
                B, Hm, T, S = attn_mask.shape
                if Hm == 1:
                    new_mask = attn_mask.expand(B, n_heads_cfg, T, S).clone()
                elif Hm == n_heads_cfg:
                    new_mask = attn_mask.clone()
                else:
                    return args, kwargs
                neg = torch.finfo(new_mask.dtype).min
                for b in range(B):
                    h = int(batch_heads[b].item())
                    if not (0 <= h < n_heads_cfg):
                        continue
                    new_mask[b, h, :, :] = neg
                    if keep_mode == "bos":
                        new_mask[b, h, :, 0] = 0
                    else:
                        diag_len = min(T, S)
                        idx = torch.arange(diag_len, device=new_mask.device)
                        new_mask[b, h, idx, idx] = 0
            if from_kwargs:
                kwargs["attention_mask"] = new_mask
                return args, kwargs
            args = list(args)
            if len(args) >= 2:
                args[1] = new_mask
            return tuple(args), kwargs

        try:
            return attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
        except TypeError:
            def pre_hook_no_kwargs(module, inputs):
                args = list(inputs)
                if len(args) < 2:
                    return inputs
                attn_mask = args[1]
                args[1] = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads, keep_mode=keep_mode)
                return tuple(args)
            return attn_mod.register_forward_pre_hook(pre_hook_no_kwargs)

    def _install_output_hooks(self, attn_mod, heads):
        num_heads, head_dim, num_kv_heads, num_kv_groups = _get_attn_head_geometry(attn_mod, self.n_heads_cfg)
        keep_mode = self.keep_mode
        controller = self
        heads = sorted(set(int(h) for h in heads))
        v_proj = getattr(attn_mod, "v_proj", None)
        o_proj = getattr(attn_mod, "o_proj", None)
        if v_proj is None or o_proj is None:
            raise RuntimeError("Output ablation requires attention modules with v_proj and o_proj.")

        def v_hook(module, args, output):
            if not torch.is_tensor(output):
                return output
            batch_size = int(output.shape[0])
            seq_len = int(output.shape[1])
            attn_mod._codex_last_value_states = output.view(batch_size, seq_len, num_kv_heads, head_dim)
            return output

        def o_pre_hook(module, args):
            if not args:
                return args
            hidden = args[0]
            if not torch.is_tensor(hidden):
                return args
            values = getattr(attn_mod, "_codex_last_value_states", None)
            if values is None:
                return args
            batch_size = int(hidden.shape[0])
            seq_len = int(hidden.shape[1])
            head_outputs = hidden.view(batch_size, seq_len, num_heads, head_dim)
            batch_heads = controller._resolve_batch_heads(batch_size, head_outputs.device)
            if batch_heads is None:
                target_heads = heads
                for h in target_heads:
                    kv_h = min(num_kv_heads - 1, int(h) // num_kv_groups)
                    if keep_mode == "bos":
                        head_outputs[:, :, h, :] = values[:, :1, kv_h, :].expand(batch_size, seq_len, head_dim)
                    else:
                        head_outputs[:, :, h, :] = values[:, :seq_len, kv_h, :]
            else:
                for b in range(batch_size):
                    h = int(batch_heads[b].item())
                    if not (0 <= h < num_heads):
                        continue
                    kv_h = min(num_kv_heads - 1, h // num_kv_groups)
                    if keep_mode == "bos":
                        head_outputs[b, :, h, :] = values[b, :1, kv_h, :].expand(seq_len, head_dim)
                    else:
                        head_outputs[b, :, h, :] = values[b, :seq_len, kv_h, :]
            return (head_outputs.reshape(batch_size, seq_len, num_heads * head_dim),) + tuple(args[1:])

        return [
            v_proj.register_forward_hook(v_hook),
            o_proj.register_forward_pre_hook(o_pre_hook),
        ]


def install_head_mask_hooks(model, layer2heads: Dict[int, List[int]], keep_mode: str = "self"):
    controller = HeadAblationController(model, layer2heads, keep_mode=keep_mode, impl="mask")
    return controller.install()


def install_head_ablation_hooks(model, layer2heads: Dict[int, List[int]], keep_mode: str = "self", impl: str = "mask"):
    controller = HeadAblationController(model, layer2heads, keep_mode=keep_mode, impl=impl)
    controller.install()
    return controller


def remove_hooks(handles):
    if isinstance(handles, HeadAblationController):
        handles = handles.handles
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


def load_plan_layers(plan_path: str, round_name: Optional[str] = None):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    rounds = plan.get("rounds")
    if isinstance(rounds, list) and rounds:
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

        merged_layers = _dedup_sorted_layers(sp.get("merged_unique_layers", []))
        if not merged_layers:
            raise ValueError("scan_plan missing or empty 'merged_unique_layers'.")

        # unify new schema into a pseudo-round for downstream logic
        round_layers = [("merged_unique_layers", merged_layers)]

    if round_name is None:
        return plan, round_layers

    for n, ls in round_layers:
        if n == round_name:
            return plan, [(n, ls)]

    raise ValueError(f"round '{round_name}' not found. Available: {[n for n, _ in round_layers]}")


def resume_key_for_head(round_name: str, stage: str, layer_idx: int, head_idx: int) -> str:
    return f"{round_name}:{stage}:layer={int(layer_idx)}:head={int(head_idx)}"


def resume_results_file_name(round_name: str, stage: str) -> str:
    return f"results__{round_name}__{stage}.jsonl"


def resume_records_file_name(round_name: str, stage: str) -> str:
    return f"records__{round_name}__{stage}.jsonl"


def load_stage_resume_results(tracker: ResumeTracker, round_name: str, stage: str) -> Dict[Tuple[int, int], dict]:
    rows = tracker.read_records(resume_results_file_name(round_name, stage))
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        try:
            layer_idx = int(row["layer"])
            head_idx = int(row["head"])
        except Exception:
            continue
        out[(layer_idx, head_idx)] = row
    return out


def load_stage_resume_records(tracker: ResumeTracker, round_name: str, stage: str) -> Dict[Tuple[int, int], dict]:
    rows = tracker.read_records(resume_records_file_name(round_name, stage))
    out: Dict[Tuple[int, int], dict] = {}
    for row in rows:
        try:
            layer_idx = int(row["layer"])
            head_idx = int(row["head"])
        except Exception:
            continue
        out[(layer_idx, head_idx)] = row
    return out


def sort_stage_results(stage: str, rows: List[dict]) -> List[dict]:
    if stage in {"coarse", "full_coarse"}:
        rows.sort(key=lambda r: (-(r["coarse_mean_abs_effect_reduction"]), r["layer"], r["head"]))
        return rows
    if stage in {"coarse_rerank", "full_rerank"}:
        rows.sort(key=lambda r: (-(r["coarse_mean_abs_effect_reduction"]), r["coarse_mean_abs_base_change"], r["layer"], r["head"]))
        return rows
    if stage in {"refine", "full_refine"}:
        rows.sort(key=lambda r: (-(r["mean_abs_effect_reduction"]), r["mean_abs_base_change"], r["layer"], r["head"]))
        return rows
    rows.sort(key=lambda r: (r.get("layer", 0), r.get("head", 0)))
    return rows


def merge_head_scan_accel_shards(args, metrics: List[str], weights: List[float]) -> None:
    _, round_layers = load_plan_layers(args.plan, args.round or None)
    Path(args.plan_out_dir).mkdir(parents=True, exist_ok=True)

    all_rounds = []
    merged_n_samples = 0
    merged_skipped = 0
    for round_name, scan_layers in round_layers:
        final_out_path = Path(args.plan_out_dir) / f"head_scan_{round_name}.json"
        shard_paths = [Path(build_shard_file_path(str(final_out_path), shard_idx, args.num_shards)) for shard_idx in range(args.num_shards)]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise SystemExit("Missing shard outputs:\n" + "\n".join(missing))

        shard_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
        first_payload = shard_payloads[0]
        merged_n_samples = int(first_payload.get("n_samples", merged_n_samples))
        merged_skipped = int(first_payload.get("skipped", merged_skipped))

        results = []
        records = []
        for payload in shard_payloads:
            results.extend(payload.get("results", []))
            if args.save_records and payload.get("records"):
                records.extend(payload.get("records", []))

        results.sort(
            key=lambda r: (
                -(r["mean_abs_effect_reduction"]),
                r["mean_abs_base_change"],
                r["layer"],
                r["head"],
            )
        )

        out = {
            "data_csv": first_payload.get("data_csv", args.data_csv),
            "image_root": first_payload.get("image_root", args.image_root),
            "model": first_payload.get("model", args.model),
            "trace_mode": first_payload.get("trace_mode", args.trace_mode),
            "position": first_payload.get("position", args.position),
            "keep_mode": first_payload.get("keep_mode", args.keep_mode),
            "metrics": metrics,
            "weights": weights,
            "n_samples": merged_n_samples,
            "skipped": merged_skipped,
            "n_layers": int(first_payload.get("n_layers", 0)),
            "n_heads": int(first_payload.get("n_heads", 0)),
            "round": round_name,
            "scan_layers": scan_layers,
            "coarse": {
                "parallel_mode": "head_shard_exact",
                "num_shards": int(args.num_shards),
                "n_all_heads": len(results),
                "n_refined_heads": len(results),
            },
            "coarse_top20": [],
            "coarse_rerank_top20": [],
            "top20": results[:20],
            "results": results,
            "shards": [str(path) for path in shard_paths],
        }
        if args.save_records:
            out["records"] = records

        final_out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        all_rounds.append({"round": round_name, "result_path": str(final_out_path), "scan_layers": scan_layers})

    summary_path = Path(args.plan_out_dir) / "head_scan_summary.json"
    summary_payload = {
        "data_csv": args.data_csv,
        "image_root": args.image_root,
        "model": args.model,
        "trace_mode": args.trace_mode,
        "position": args.position,
        "keep_mode": args.keep_mode,
        "metrics": metrics,
        "weights": weights,
        "limit": args.limit,
        "coarse_limit": args.coarse_limit,
        "coarse_topk": 0,
        "coarse_with_nc": bool(args.coarse_with_nc),
        "coarse_seed": int(args.coarse_seed),
        "plan": args.plan,
        "rounds_ran": [x["round"] for x in all_rounds],
        "round_outputs": all_rounds,
        "n_samples": merged_n_samples,
        "skipped": merged_skipped,
        "parallel_mode": "head_shard_exact",
        "num_shards": int(args.num_shards),
        "saved": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Merged per-shard results under: {args.plan_out_dir}")
    print(f"[OK] Saved summary: {summary_path}")


def merge_coarse_stage(args, metrics: List[str], weights: List[float]) -> None:
    _, round_layers = load_plan_layers(args.plan, args.round or None)
    for round_name, scan_layers in round_layers:
        final_out_path = str(Path(args.plan_out_dir) / f"head_scan_{round_name}.json")
        shard_paths = [
            Path(build_stage_shard_file_path(final_out_path, "coarse", shard_idx, args.num_shards))
            for shard_idx in range(args.num_shards)
        ]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise SystemExit("Missing coarse shard outputs:\n" + "\n".join(missing))

        merged_scores = []
        base_payload = None
        for shard_path in shard_paths:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            if base_payload is None:
                base_payload = payload
            merged_scores.extend(payload.get("results", []))

        merged_scores.sort(
            key=lambda r: (-(r["coarse_mean_abs_effect_reduction"]), r["layer"], r["head"])
        )
        keep_n = min(int(args.coarse_topk), len(merged_scores))
        rerank_pool_n = min(len(merged_scores), max(keep_n, keep_n * 2))
        rerank_pool = merged_scores[:rerank_pool_n]
        merged_payload = {
            "round": round_name,
            "scan_layers": scan_layers,
            "metrics": metrics,
            "weights": weights,
            "n_samples": int(base_payload.get("n_samples", 0) if base_payload else 0),
            "skipped": int(base_payload.get("skipped", 0) if base_payload else 0),
            "coarse_topk": int(args.coarse_topk),
            "rerank_pool_n": rerank_pool_n,
            "results": merged_scores,
            "rerank_pool": rerank_pool,
            "shards": [str(path) for path in shard_paths],
        }
        merged_path = Path(build_stage_file_path(final_out_path, "coarse_merged"))
        merged_path.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved coarse merge: {merged_path}")


def merge_coarse_rerank_stage(args, metrics: List[str], weights: List[float]) -> None:
    _, round_layers = load_plan_layers(args.plan, args.round or None)
    for round_name, scan_layers in round_layers:
        final_out_path = str(Path(args.plan_out_dir) / f"head_scan_{round_name}.json")
        shard_paths = [
            Path(build_stage_shard_file_path(final_out_path, "coarse_rerank", shard_idx, args.num_shards))
            for shard_idx in range(args.num_shards)
        ]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise SystemExit("Missing coarse_rerank shard outputs:\n" + "\n".join(missing))

        merged_scores = []
        for shard_path in shard_paths:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            merged_scores.extend(payload.get("results", []))

        merged_scores.sort(
            key=lambda r: (
                -(r["coarse_mean_abs_effect_reduction"]),
                r["coarse_mean_abs_base_change"],
                r["layer"],
                r["head"],
            )
        )
        keep_n = min(int(args.coarse_topk), len(merged_scores))
        candidate_heads = [(int(x["layer"]), int(x["head"])) for x in merged_scores[:keep_n]]
        merged_payload = {
            "round": round_name,
            "scan_layers": scan_layers,
            "metrics": metrics,
            "weights": weights,
            "coarse_topk": int(args.coarse_topk),
            "results": merged_scores,
            "candidate_heads": candidate_heads,
            "shards": [str(path) for path in shard_paths],
        }
        merged_path = Path(build_stage_file_path(final_out_path, "coarse_rerank_merged"))
        merged_path.write_text(json.dumps(merged_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Saved coarse rerank merge: {merged_path}")


def merge_refine_stage(args, metrics: List[str], weights: List[float]) -> None:
    _, round_layers = load_plan_layers(args.plan, args.round or None)
    Path(args.plan_out_dir).mkdir(parents=True, exist_ok=True)

    all_rounds = []
    merged_n_samples = 0
    merged_skipped = 0
    for round_name, scan_layers in round_layers:
        final_out_path = Path(args.plan_out_dir) / f"head_scan_{round_name}.json"
        shard_paths = [
            Path(build_stage_shard_file_path(str(final_out_path), "refine", shard_idx, args.num_shards))
            for shard_idx in range(args.num_shards)
        ]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise SystemExit("Missing refine shard outputs:\n" + "\n".join(missing))

        coarse_merged_path = Path(build_stage_file_path(str(final_out_path), "coarse_merged"))
        coarse_rerank_merged_path = Path(build_stage_file_path(str(final_out_path), "coarse_rerank_merged"))
        coarse_payload = json.loads(coarse_merged_path.read_text(encoding="utf-8")) if coarse_merged_path.exists() else {}
        rerank_payload = json.loads(coarse_rerank_merged_path.read_text(encoding="utf-8")) if coarse_rerank_merged_path.exists() else {}

        shard_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
        first_payload = shard_payloads[0]
        merged_n_samples = int(first_payload.get("n_samples", merged_n_samples))
        merged_skipped = int(first_payload.get("skipped", merged_skipped))

        results = []
        records = []
        for payload in shard_payloads:
            results.extend(payload.get("results", []))
            if args.save_records and payload.get("records"):
                records.extend(payload.get("records", []))

        results.sort(
            key=lambda r: (
                -(r["mean_abs_effect_reduction"]),
                r["mean_abs_base_change"],
                r["layer"],
                r["head"],
            )
        )

        out = {
            "data_csv": first_payload.get("data_csv", args.data_csv),
            "image_root": first_payload.get("image_root", args.image_root),
            "model": first_payload.get("model", args.model),
            "trace_mode": first_payload.get("trace_mode", args.trace_mode),
            "position": first_payload.get("position", args.position),
            "keep_mode": first_payload.get("keep_mode", args.keep_mode),
            "metrics": metrics,
            "weights": weights,
            "n_samples": merged_n_samples,
            "skipped": merged_skipped,
            "n_layers": int(first_payload.get("n_layers", 0)),
            "n_heads": int(first_payload.get("n_heads", 0)),
            "round": round_name,
            "scan_layers": scan_layers,
            "coarse": {
                "parallel_mode": "global_topk_parallel",
                "num_shards": int(args.num_shards),
                "coarse_limit": args.coarse_limit,
                "coarse_with_nc": bool(args.coarse_with_nc),
                "coarse_seed": int(args.coarse_seed),
                "coarse_topk": int(args.coarse_topk),
                "n_all_heads": int(coarse_payload.get("results") and len(coarse_payload["results"]) or len(results)),
                "n_rerank_heads": int(rerank_payload.get("results") and len(rerank_payload["results"]) or 0),
                "n_refined_heads": len(results),
            },
            "coarse_top20": coarse_payload.get("results", [])[:20],
            "coarse_rerank_top20": rerank_payload.get("results", [])[:20],
            "top20": results[:20],
            "results": results,
            "shards": [str(path) for path in shard_paths],
        }
        if args.save_records:
            out["records"] = records

        final_out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        all_rounds.append({"round": round_name, "result_path": str(final_out_path), "scan_layers": scan_layers})

    summary_path = Path(args.plan_out_dir) / "head_scan_summary.json"
    summary_payload = {
        "data_csv": args.data_csv,
        "image_root": args.image_root,
        "model": args.model,
        "trace_mode": args.trace_mode,
        "position": args.position,
        "keep_mode": args.keep_mode,
        "metrics": metrics,
        "weights": weights,
        "limit": args.limit,
        "coarse_limit": args.coarse_limit,
        "coarse_topk": int(args.coarse_topk),
        "coarse_with_nc": bool(args.coarse_with_nc),
        "coarse_seed": int(args.coarse_seed),
        "plan": args.plan,
        "rounds_ran": [x["round"] for x in all_rounds],
        "round_outputs": all_rounds,
        "n_samples": merged_n_samples,
        "skipped": merged_skipped,
        "parallel_mode": "global_topk_parallel",
        "num_shards": int(args.num_shards),
        "saved": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Merged global-topk per-shard results under: {args.plan_out_dir}")
    print(f"[OK] Saved summary: {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["cc", "conflict"])
    ap.add_argument("--position", type=str, default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])

    ap.add_argument("--metrics", type=str, default="gold_wrong_margin,follow_context")
    ap.add_argument("--weights", type=str, default="")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coarse_limit", type=int, default=60, help="coarse stage sample cap (0 means use all samples)")
    ap.add_argument("--coarse_topk", type=int, default=80, help="heads kept after coarse stage (0 means keep all)")
    ap.add_argument("--coarse_with_nc", action="store_true", help="if set, coarse stage also computes nc_ab (slower)")
    ap.add_argument("--coarse_seed", type=int, default=42, help="random seed for coarse-stage subsampling")
    ap.add_argument("--qwen_prefill_batch_size", type=int, default=0, help="qwen-only micro-batch size for prompt-cache prefill; 0 uses a model-specific default")
    ap.add_argument("--head_ablation_impl", type=str, default="mask", choices=["mask", "output"], help="mask keeps the original eager attention implementation; output modifies head outputs before o_proj and can use a faster attention backend")
    ap.add_argument("--attention_backend", type=str, default="auto", choices=["auto", "eager", "sdpa"], help="attention backend to request from transformers; auto picks eager for mask ablation and sdpa for output ablation")
    ap.add_argument("--head_batch_size", type=int, default=1, help="evaluate this many heads together when they belong to the same layer; works across head-scan stages and models that support the installed ablation hook")
    ap.add_argument("--qwen_vl_vision_cache", action="store_true", help="cache vision-encoder outputs across repeated image forwards when the loaded model exposes a reusable vision encoder")
    ap.add_argument("--plan", type=str, default="")
    ap.add_argument("--round", default='merged_unique_layers', help="optional group name; for new scan_plan use 'merged_unique_layers'")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--plan_out_dir", type=str, default="result/headscan_slake_mm")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save_records", action="store_true")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--merge_shards", action="store_true")
    ap.add_argument(
        "--parallel_stage",
        type=str,
        default="full",
        choices=["full", "coarse", "merge_coarse", "coarse_rerank", "merge_coarse_rerank", "refine", "merge_refine"],
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    validate_shard_args(args.num_shards, args.shard_idx)
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.plan_out_dir = prefix_model_relative_path(args.plan_out_dir, model_name=args.model_name, model=args.model)
    shard_scope = build_shard_token(args.shard_idx, args.num_shards) if args.num_shards > 1 else ""

    metrics, weights = parse_metric_weights(args.metrics, args.weights)

    if args.parallel_stage == "merge_coarse":
        merge_coarse_stage(args, metrics, weights)
        return
    if args.parallel_stage == "merge_coarse_rerank":
        merge_coarse_rerank_stage(args, metrics, weights)
        return
    if args.parallel_stage == "merge_refine":
        merge_refine_stage(args, metrics, weights)
        return
    if args.merge_shards:
        merge_head_scan_accel_shards(args, metrics, weights)
        return

    df = pd.read_csv(args.data_csv)
    if "gold" not in df.columns:
        raise ValueError("CSV must contain a 'gold' column.")
    if "wrong" not in df.columns:
        raise ValueError("CSV must contain a 'wrong' column.")
    if "question" not in df.columns:
        raise ValueError("CSV must contain a 'question' column.")

    if args.head_ablation_impl == "output" and args.keep_mode != "self":
        print("[WARN] output head ablation currently matches the self-only path best; falling back to mask impl for keep_mode!=self")
        args.head_ablation_impl = "mask"
    attention_backend = resolve_attention_backend(args.head_ablation_impl, args.attention_backend)
    model, processor = load_model_and_processor(args.model, args.device, args.dtype, attn_backend=attention_backend)
    qwen_accel_mode = is_qwen_headscan_accel_model(model)
    qwen_prefill_batch_size = resolve_qwen_prefill_batch_size(model, args.qwen_prefill_batch_size)
    head_batch_size = max(1, int(args.head_batch_size))
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

    for row in tqdm(rows, desc="Build samples"):
        gold = norm_text(row["gold"])
        wrong = norm_text(row["wrong"])
        question = str(row["question"])
        if not gold or not wrong or gold == wrong:
            skipped += 1
            continue

        img_path = resolve_image_path(row, args.image_root)
        if img_path is None:
            skipped += 1
            continue

        image = resize_image_max_side(Image.open(img_path))
        nc_prompt = build_nc_prompt(question)
        ctx_evidence = gold if args.trace_mode == "cc" else wrong
        ctx_prompt = build_ctx_prompt(question, ctx_evidence, position=args.position)

        vision_cache_key = str(img_path) if args.qwen_vl_vision_cache else ""
        nc_inputs = build_inputs_mm(processor, nc_prompt, image, model.device, vision_cache_key=vision_cache_key)
        ctx_inputs = build_inputs_mm(processor, ctx_prompt, image, model.device, vision_cache_key=vision_cache_key)

        sample = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "image_path": str(img_path),
            "nc_inputs": move_inputs_to_cpu(nc_inputs),
            "ctx_inputs": move_inputs_to_cpu(ctx_inputs),
            "gold_token_ids": tokenize_continuation_ids(processor, " " + gold),
            "wrong_token_ids": tokenize_continuation_ids(processor, " " + wrong),
        }
        del nc_inputs
        del ctx_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if qwen_accel_mode:
            samples.append(sample)
            continue

        nc_cache = build_prompt_cache_from_inputs(model, sample["nc_inputs"])
        ctx_cache = build_prompt_cache_from_inputs(model, sample["ctx_inputs"])
        nc_scores = score_two_options_from_cache(
            model,
            processor,
            nc_cache,
            gold,
            wrong,
            gold_token_ids=sample["gold_token_ids"],
            wrong_token_ids=sample["wrong_token_ids"],
        )
        ctx_scores = score_two_options_from_cache(
            model,
            processor,
            ctx_cache,
            gold,
            wrong,
            gold_token_ids=sample["gold_token_ids"],
            wrong_token_ids=sample["wrong_token_ids"],
        )
        attach_sample_baseline_metrics(sample, nc_scores, ctx_scores, metrics, weights, args.trace_mode)
        del nc_cache
        del ctx_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        samples.append(sample)

    if not samples:
        raise SystemExit("No usable examples after filtering.")

    if qwen_accel_mode:
        baseline_scores = score_samples_for_prompt_keys_qwen(
            model,
            processor,
            samples,
            ["nc_inputs", "ctx_inputs"],
            requested_batch_size=qwen_prefill_batch_size,
            progress_desc="Baseline qwen prefill",
        )
        for sample, nc_scores, ctx_scores in zip(
            samples,
            baseline_scores["nc_inputs"],
            baseline_scores["ctx_inputs"],
        ):
            attach_sample_baseline_metrics(sample, nc_scores, ctx_scores, metrics, weights, args.trace_mode)

    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            Path(__file__).resolve().parent,
            task_name="head_scan_accel",
            model_name=args.model_name,
            model=args.model,
            position=args.position,
            scope=shard_scope,
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
        plan_out_dir=args.plan_out_dir,
        out=args.out,
        save_records=bool(args.save_records),
        n_samples=len(samples),
        skipped=skipped,
        coarse_limit=args.coarse_limit,
        coarse_topk=args.coarse_topk,
        coarse_with_nc=bool(args.coarse_with_nc),
        coarse_seed=int(args.coarse_seed),
        qwen_prefill_batch_size=int(qwen_prefill_batch_size),
        head_ablation_impl=args.head_ablation_impl,
        attention_backend=attention_backend,
        head_batch_size=head_batch_size,
        qwen_vl_vision_cache=bool(args.qwen_vl_vision_cache),
    )

    def run_round(scan_layers: List[int], out_path: str, round_name: str):
        all_heads_full = [(int(layer_idx), int(head_idx)) for layer_idx in scan_layers for head_idx in range(n_heads)]
        stage = args.parallel_stage
        coarse_samples = samples
        if args.coarse_limit and args.coarse_limit > 0:
            coarse_n = min(len(samples), int(args.coarse_limit))
            coarse_rng = random.Random(int(args.coarse_seed))
            if coarse_n < len(samples):
                coarse_indices = sorted(coarse_rng.sample(range(len(samples)), coarse_n))
                coarse_samples = [samples[i] for i in coarse_indices]
            else:
                coarse_samples = list(samples)

        if stage == "coarse":
            assigned_heads = list(iter_heads_for_shard(all_heads_full, args.num_shards, args.shard_idx))
            assigned_head_set = {(int(layer_idx), int(head_idx)) for layer_idx, head_idx in assigned_heads}
            stage_out_path = build_stage_shard_file_path(out_path, "coarse", args.shard_idx, args.num_shards)
            stage_tag = "coarse"
            coarse_scores_map = load_stage_resume_results(tracker, round_name, stage_tag)
            pending_heads = []
            for layer_idx, head_idx in assigned_heads:
                head_key = (int(layer_idx), int(head_idx))
                resume_key = resume_key_for_head(round_name, stage_tag, head_key[0], head_key[1])
                if tracker.is_done(resume_key) and head_key in coarse_scores_map:
                    continue
                pending_heads.append(head_key)
            pending_head_groups = build_head_batch_groups(pending_heads, head_batch_size)
            for head_group in tqdm(pending_head_groups, desc=f"Coarse {round_name}", unit="group"):
                layer2heads = {}
                for layer_idx, head_idx in head_group:
                    layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                try:
                    if len(head_group) > 1:
                        prompt_keys = ["ctx_inputs"]
                        if args.coarse_with_nc:
                            prompt_keys.append("nc_inputs")
                        expanded_samples, _ = expand_samples_for_head_group(coarse_samples, head_group)
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            expanded_samples,
                            prompt_keys,
                            requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                            ablation_controller=handles,
                        )
                        grouped_scores = summarize_head_group_scores(
                            head_group,
                            ablated_scores,
                            coarse_samples,
                            metrics,
                            weights,
                            args.trace_mode,
                            include_nc=bool(args.coarse_with_nc),
                        )
                        for layer_idx, head_idx in head_group:
                            item = {
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "coarse_mean_abs_effect_reduction": grouped_scores[(int(layer_idx), int(head_idx))]["coarse_mean_abs_effect_reduction"],
                            }
                            coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                            tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                            tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    elif qwen_accel_mode:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        n = 0
                        prompt_keys = ["ctx_inputs"]
                        if args.coarse_with_nc:
                            prompt_keys.append("nc_inputs")
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            coarse_samples,
                            prompt_keys,
                            requested_batch_size=qwen_prefill_batch_size,
                            ablation_controller=handles,
                        )
                        for sample_idx, s in enumerate(coarse_samples):
                            gold = s["gold"]
                            wrong = s["wrong"]
                            ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            if args.coarse_with_nc:
                                nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                                nc_ref = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            else:
                                nc_ref = s["base_nc_scalar"]
                            effect_ab = ctx_ab_scalar - nc_ref
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1)}
                        coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    else:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        n = 0
                        for s in coarse_samples:
                            gold = s["gold"]
                            wrong = s["wrong"]
                            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                            ctx_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                ctx_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            if args.coarse_with_nc:
                                nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                                nc_ab_scores = score_two_options_from_cache(
                                    model,
                                    processor,
                                    nc_cache_ab,
                                    gold,
                                    wrong,
                                    gold_token_ids=s.get("gold_token_ids"),
                                    wrong_token_ids=s.get("wrong_token_ids"),
                                )
                                nc_ref = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            else:
                                nc_ref = s["base_nc_scalar"]
                            effect_ab = ctx_ab_scalar - nc_ref
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            n += 1
                            del ctx_cache_ab
                            if args.coarse_with_nc:
                                del nc_cache_ab
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1)}
                        coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finally:
                    remove_hooks(handles)
            coarse_scores = sort_stage_results(stage_tag, [item for key, item in coarse_scores_map.items() if key in assigned_head_set])
            payload = {"data_csv": args.data_csv, "image_root": args.image_root, "model": args.model, "trace_mode": args.trace_mode, "position": args.position, "keep_mode": args.keep_mode, "metrics": metrics, "weights": weights, "n_samples": len(samples), "skipped": skipped, "n_layers": n_layers, "n_heads": n_heads, "round": round_name, "scan_layers": scan_layers, "results": coarse_scores}
            Path(stage_out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload

        if stage == "coarse_rerank":
            merged_coarse_path = Path(build_stage_file_path(out_path, "coarse_merged"))
            if not merged_coarse_path.exists():
                raise SystemExit(f"Missing merged coarse file: {merged_coarse_path}")
            merged_coarse = json.loads(merged_coarse_path.read_text(encoding="utf-8"))
            rerank_pool = [(int(x["layer"]), int(x["head"])) for x in merged_coarse.get("rerank_pool", [])]
            assigned_heads = list(iter_heads_for_shard(rerank_pool, args.num_shards, args.shard_idx))
            assigned_head_set = {(int(layer_idx), int(head_idx)) for layer_idx, head_idx in assigned_heads}
            stage_out_path = build_stage_shard_file_path(out_path, "coarse_rerank", args.shard_idx, args.num_shards)
            stage_tag = "coarse_rerank"
            rerank_scores_map = load_stage_resume_results(tracker, round_name, stage_tag)
            pending_heads = []
            for layer_idx, head_idx in assigned_heads:
                head_key = (int(layer_idx), int(head_idx))
                resume_key = resume_key_for_head(round_name, stage_tag, head_key[0], head_key[1])
                if tracker.is_done(resume_key) and head_key in rerank_scores_map:
                    continue
                pending_heads.append(head_key)
            pending_head_groups = build_head_batch_groups(pending_heads, head_batch_size)
            for head_group in tqdm(pending_head_groups, desc=f"Rerank {round_name}", unit="group"):
                layer2heads = {}
                for layer_idx, head_idx in head_group:
                    layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                try:
                    if len(head_group) > 1:
                        expanded_samples, _ = expand_samples_for_head_group(coarse_samples, head_group)
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            expanded_samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                            ablation_controller=handles,
                        )
                        grouped_scores = summarize_head_group_scores(
                            head_group,
                            ablated_scores,
                            coarse_samples,
                            metrics,
                            weights,
                            args.trace_mode,
                            include_nc=True,
                        )
                        for layer_idx, head_idx in head_group:
                            group_item = grouped_scores[(int(layer_idx), int(head_idx))]
                            item = {
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "coarse_mean_abs_effect_reduction": group_item["coarse_mean_abs_effect_reduction"],
                                "coarse_mean_abs_base_change": group_item["coarse_mean_abs_base_change"],
                            }
                            rerank_scores_map[(int(layer_idx), int(head_idx))] = item
                            tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                            tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    elif qwen_accel_mode:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            coarse_samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=qwen_prefill_batch_size,
                            ablation_controller=handles,
                        )
                        for sample_idx, s in enumerate(coarse_samples):
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                            ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1), "coarse_mean_abs_base_change": base_change_sum / max(n, 1)}
                        rerank_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    else:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        for s in coarse_samples:
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                            nc_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                nc_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                ctx_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1), "coarse_mean_abs_base_change": base_change_sum / max(n, 1)}
                        rerank_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                finally:
                    remove_hooks(handles)
            rerank_scores = sort_stage_results(stage_tag, [item for key, item in rerank_scores_map.items() if key in assigned_head_set])
            payload = {"data_csv": args.data_csv, "image_root": args.image_root, "model": args.model, "trace_mode": args.trace_mode, "position": args.position, "keep_mode": args.keep_mode, "metrics": metrics, "weights": weights, "n_samples": len(samples), "skipped": skipped, "n_layers": n_layers, "n_heads": n_heads, "round": round_name, "scan_layers": scan_layers, "results": rerank_scores}
            Path(stage_out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload

        if stage == "refine":
            merged_rerank_path = Path(build_stage_file_path(out_path, "coarse_rerank_merged"))
            merged_coarse_path = Path(build_stage_file_path(out_path, "coarse_merged"))
            if merged_rerank_path.exists():
                merged_candidates = json.loads(merged_rerank_path.read_text(encoding="utf-8")).get("candidate_heads", [])
            elif merged_coarse_path.exists():
                merged_candidates = [(int(x["layer"]), int(x["head"])) for x in json.loads(merged_coarse_path.read_text(encoding="utf-8")).get("results", [])[: int(args.coarse_topk)]]
            else:
                raise SystemExit(f"Missing candidate source: {merged_rerank_path} or {merged_coarse_path}")
            candidate_heads = [(int(x[0]), int(x[1])) if isinstance(x, list) else (int(x["layer"]), int(x["head"])) for x in merged_candidates]
            assigned_heads = list(iter_heads_for_shard(candidate_heads, args.num_shards, args.shard_idx))
            assigned_head_set = {(int(layer_idx), int(head_idx)) for layer_idx, head_idx in assigned_heads}
            stage_out_path = build_stage_shard_file_path(out_path, "refine", args.shard_idx, args.num_shards)
            stage_tag = "refine"
            results_map = load_stage_resume_results(tracker, round_name, stage_tag)
            records_map = load_stage_resume_records(tracker, round_name, stage_tag) if args.save_records else {}
            pending_heads = []
            for layer_idx, head_idx in assigned_heads:
                head_key = (int(layer_idx), int(head_idx))
                resume_key = resume_key_for_head(round_name, stage_tag, head_key[0], head_key[1])
                if tracker.is_done(resume_key) and head_key in results_map:
                    continue
                pending_heads.append(head_key)
            pending_head_groups = build_head_batch_groups(pending_heads, head_batch_size)
            for head_group in tqdm(pending_head_groups, desc=f"Refine {round_name}", unit="group"):
                layer2heads = {}
                for layer_idx, head_idx in head_group:
                    layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                try:
                    if len(head_group) > 1:
                        expanded_samples, _ = expand_samples_for_head_group(samples, head_group)
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            expanded_samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                            ablation_controller=handles,
                        )
                        grouped_scores = summarize_head_group_scores(
                            head_group,
                            ablated_scores,
                            samples,
                            metrics,
                            weights,
                            args.trace_mode,
                            include_nc=True,
                        )
                        sample_count = len(samples)
                        for head_pos, (layer_idx, head_idx) in enumerate(head_group):
                            group_item = grouped_scores[(int(layer_idx), int(head_idx))]
                            item = {
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "mean_abs_effect_reduction": group_item["mean_abs_effect_reduction"],
                                "mean_abs_base_change": group_item["mean_abs_base_change"],
                                "metric_mean_abs_effect_reduction": group_item["metric_mean_abs_effect_reduction"],
                                "metric_mean_abs_base_change": group_item["metric_mean_abs_base_change"],
                            }
                            results_map[(int(layer_idx), int(head_idx))] = item
                            tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                            if args.save_records:
                                start = head_pos * sample_count
                                end = start + sample_count
                                chunk_nc = ablated_scores["nc_inputs"][start:end]
                                chunk_ctx = ablated_scores["ctx_inputs"][start:end]
                                per_head_records = []
                                for sample_idx, s in enumerate(samples):
                                    gold = s["gold"]
                                    wrong = s["wrong"]
                                    nc_ab_scores = chunk_nc[sample_idx]
                                    ctx_ab_scores = chunk_ctx[sample_idx]
                                    ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                                    effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                                    per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                                record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                                records_map[(int(layer_idx), int(head_idx))] = record_item
                                tracker.append_record(resume_records_file_name(round_name, stage_tag), record_item)
                            tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    elif qwen_accel_mode:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        metric_eff_red = {m: 0.0 for m in metrics}
                        metric_base_change = {m: 0.0 for m in metrics}
                        per_head_records = []
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=qwen_prefill_batch_size,
                            ablation_controller=handles,
                        )
                        for sample_idx, s in enumerate(samples):
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                            ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            for m in metrics:
                                nc_m_base = s["base_metric_values"][m]["nc"]
                                ctx_m_ab = metric_from_scores(m, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_m_ab = metric_from_scores(m, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_m_ab = ctx_m_ab - nc_m_base
                                metric_eff_red[m] += abs(s["base_metric_values"][m]["effect"]) - abs(effect_m_ab)
                                metric_base_change[m] += abs(nc_m_ab - nc_m_base)
                            if args.save_records:
                                per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "mean_abs_effect_reduction": eff_red_sum / max(n, 1), "mean_abs_base_change": base_change_sum / max(n, 1), "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(n, 1) for m in metrics}, "metric_mean_abs_base_change": {m: metric_base_change[m] / max(n, 1) for m in metrics}}
                        results_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        if args.save_records:
                            record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                            records_map[(int(layer_idx), int(head_idx))] = record_item
                            tracker.append_record(resume_records_file_name(round_name, stage_tag), record_item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                    else:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        metric_eff_red = {m: 0.0 for m in metrics}
                        metric_base_change = {m: 0.0 for m in metrics}
                        per_head_records = []
                        for s in samples:
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                            nc_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                nc_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                ctx_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            for m in metrics:
                                nc_m_base = s["base_metric_values"][m]["nc"]
                                ctx_m_ab = metric_from_scores(m, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_m_ab = metric_from_scores(m, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_m_ab = ctx_m_ab - nc_m_base
                                metric_eff_red[m] += abs(s["base_metric_values"][m]["effect"]) - abs(effect_m_ab)
                                metric_base_change[m] += abs(nc_m_ab - nc_m_base)
                            if args.save_records:
                                per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                            del nc_cache_ab
                            del ctx_cache_ab
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "mean_abs_effect_reduction": eff_red_sum / max(n, 1), "mean_abs_base_change": base_change_sum / max(n, 1), "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(n, 1) for m in metrics}, "metric_mean_abs_base_change": {m: metric_base_change[m] / max(n, 1) for m in metrics}}
                        results_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, stage_tag), item)
                        if args.save_records:
                            record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                            records_map[(int(layer_idx), int(head_idx))] = record_item
                            tracker.append_record(resume_records_file_name(round_name, stage_tag), record_item)
                        tracker.mark_done(resume_key_for_head(round_name, stage_tag, layer_idx, head_idx), {"status": "ok", "stage": stage_tag, "round": round_name})
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finally:
                    remove_hooks(handles)
            results = sort_stage_results(stage_tag, [item for key, item in results_map.items() if key in assigned_head_set])
            payload = {"data_csv": args.data_csv, "image_root": args.image_root, "model": args.model, "trace_mode": args.trace_mode, "position": args.position, "keep_mode": args.keep_mode, "metrics": metrics, "weights": weights, "n_samples": len(samples), "skipped": skipped, "n_layers": n_layers, "n_heads": n_heads, "round": round_name, "scan_layers": scan_layers, "results": results}
            if args.save_records:
                payload["records"] = [records_map[k] for k in sorted(records_map) if k in assigned_head_set]
            Path(stage_out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload

        if stage == "full":
            all_heads = all_heads_full
            all_head_set = {(int(layer_idx), int(head_idx)) for layer_idx, head_idx in all_heads}
            coarse_stage_tag = "full_coarse"
            coarse_scores_map = load_stage_resume_results(tracker, round_name, coarse_stage_tag)
            pending_coarse_heads = []
            for layer_idx, head_idx in all_heads:
                head_key = (int(layer_idx), int(head_idx))
                resume_key = resume_key_for_head(round_name, coarse_stage_tag, head_key[0], head_key[1])
                if tracker.is_done(resume_key) and head_key in coarse_scores_map:
                    continue
                pending_coarse_heads.append(head_key)
            pending_coarse_groups = build_head_batch_groups(pending_coarse_heads, head_batch_size)
            for head_group in tqdm(pending_coarse_groups, desc=f"Coarse {round_name}", unit="group"):
                layer2heads = {}
                for layer_idx, head_idx in head_group:
                    layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                try:
                    if len(head_group) > 1:
                        prompt_keys = ["ctx_inputs"]
                        if args.coarse_with_nc:
                            prompt_keys.append("nc_inputs")
                        expanded_samples, _ = expand_samples_for_head_group(coarse_samples, head_group)
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            expanded_samples,
                            prompt_keys,
                            requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                            ablation_controller=handles,
                        )
                        grouped_scores = summarize_head_group_scores(
                            head_group,
                            ablated_scores,
                            coarse_samples,
                            metrics,
                            weights,
                            args.trace_mode,
                            include_nc=bool(args.coarse_with_nc),
                        )
                        for layer_idx, head_idx in head_group:
                            item = {
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "coarse_mean_abs_effect_reduction": grouped_scores[(int(layer_idx), int(head_idx))]["coarse_mean_abs_effect_reduction"],
                            }
                            coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                            tracker.append_record(resume_results_file_name(round_name, coarse_stage_tag), item)
                            tracker.mark_done(resume_key_for_head(round_name, coarse_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": coarse_stage_tag, "round": round_name})
                    elif qwen_accel_mode:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        n = 0
                        prompt_keys = ["ctx_inputs"]
                        if args.coarse_with_nc:
                            prompt_keys.append("nc_inputs")
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            coarse_samples,
                            prompt_keys,
                            requested_batch_size=qwen_prefill_batch_size,
                            ablation_controller=handles,
                        )
                        for sample_idx, s in enumerate(coarse_samples):
                            gold = s["gold"]
                            wrong = s["wrong"]
                            ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            if args.coarse_with_nc:
                                nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                                nc_ref = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            else:
                                nc_ref = s["base_nc_scalar"]
                            effect_ab = ctx_ab_scalar - nc_ref
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1)}
                        coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, coarse_stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, coarse_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": coarse_stage_tag, "round": round_name})
                    else:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        n = 0
                        for s in coarse_samples:
                            gold = s["gold"]
                            wrong = s["wrong"]
                            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                            ctx_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                ctx_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            if args.coarse_with_nc:
                                nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                                nc_ab_scores = score_two_options_from_cache(
                                    model,
                                    processor,
                                    nc_cache_ab,
                                    gold,
                                    wrong,
                                    gold_token_ids=s.get("gold_token_ids"),
                                    wrong_token_ids=s.get("wrong_token_ids"),
                                )
                                nc_ref = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            else:
                                nc_ref = s["base_nc_scalar"]
                            effect_ab = ctx_ab_scalar - nc_ref
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1)}
                        coarse_scores_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, coarse_stage_tag), item)
                        tracker.mark_done(resume_key_for_head(round_name, coarse_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": coarse_stage_tag, "round": round_name})
                finally:
                    remove_hooks(handles)
            coarse_scores = sort_stage_results(coarse_stage_tag, [item for key, item in coarse_scores_map.items() if key in all_head_set])

            coarse_rerank_scores = []
            if args.coarse_topk and args.coarse_topk > 0:
                keep_n = min(int(args.coarse_topk), len(coarse_scores))
                rerank_pool_n = min(len(coarse_scores), max(keep_n, keep_n * 2))
                rerank_pool = coarse_scores[:rerank_pool_n]
                rerank_stage_tag = "full_rerank"
                coarse_rerank_scores_map = load_stage_resume_results(tracker, round_name, rerank_stage_tag)
                pending_rerank_heads = []
                for item in rerank_pool:
                    head_key = (int(item["layer"]), int(item["head"]))
                    resume_key = resume_key_for_head(round_name, rerank_stage_tag, head_key[0], head_key[1])
                    if tracker.is_done(resume_key) and head_key in coarse_rerank_scores_map:
                        continue
                    pending_rerank_heads.append(head_key)
                pending_rerank_groups = build_head_batch_groups(pending_rerank_heads, head_batch_size)
                for head_group in tqdm(pending_rerank_groups, desc=f"Rerank {round_name}", unit="group"):
                    layer2heads = {}
                    for layer_idx, head_idx in head_group:
                        layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                    handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                    try:
                        if len(head_group) > 1:
                            expanded_samples, _ = expand_samples_for_head_group(coarse_samples, head_group)
                            ablated_scores = score_samples_for_prompt_keys_qwen(
                                model,
                                processor,
                                expanded_samples,
                                ["nc_inputs", "ctx_inputs"],
                                requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                                ablation_controller=handles,
                            )
                            grouped_scores = summarize_head_group_scores(
                                head_group,
                                ablated_scores,
                                coarse_samples,
                                metrics,
                                weights,
                                args.trace_mode,
                                include_nc=True,
                            )
                            for layer_idx, head_idx in head_group:
                                group_item = grouped_scores[(int(layer_idx), int(head_idx))]
                                rerank_item = {
                                    "layer": int(layer_idx),
                                    "head": int(head_idx),
                                    "coarse_mean_abs_effect_reduction": group_item["coarse_mean_abs_effect_reduction"],
                                    "coarse_mean_abs_base_change": group_item["coarse_mean_abs_base_change"],
                                }
                                coarse_rerank_scores_map[(int(layer_idx), int(head_idx))] = rerank_item
                                tracker.append_record(resume_results_file_name(round_name, rerank_stage_tag), rerank_item)
                                tracker.mark_done(resume_key_for_head(round_name, rerank_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": rerank_stage_tag, "round": round_name})
                        elif qwen_accel_mode:
                            layer_idx, head_idx = head_group[0]
                            layer_idx = int(layer_idx)
                            head_idx = int(head_idx)
                            eff_red_sum = 0.0
                            base_change_sum = 0.0
                            n = 0
                            ablated_scores = score_samples_for_prompt_keys_qwen(
                                model,
                                processor,
                                coarse_samples,
                                ["nc_inputs", "ctx_inputs"],
                                requested_batch_size=qwen_prefill_batch_size,
                                ablation_controller=handles,
                            )
                            for sample_idx, s in enumerate(coarse_samples):
                                gold = s["gold"]
                                wrong = s["wrong"]
                                nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                                ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                                ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                                eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                                base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                                n += 1
                            rerank_item = {"layer": layer_idx, "head": head_idx, "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1), "coarse_mean_abs_base_change": base_change_sum / max(n, 1)}
                            coarse_rerank_scores_map[(layer_idx, head_idx)] = rerank_item
                            tracker.append_record(resume_results_file_name(round_name, rerank_stage_tag), rerank_item)
                            tracker.mark_done(resume_key_for_head(round_name, rerank_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": rerank_stage_tag, "round": round_name})
                        else:
                            layer_idx, head_idx = head_group[0]
                            layer_idx = int(layer_idx)
                            head_idx = int(head_idx)
                            eff_red_sum = 0.0
                            base_change_sum = 0.0
                            n = 0
                            for s in coarse_samples:
                                gold = s["gold"]
                                wrong = s["wrong"]
                                nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                                ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                                nc_ab_scores = score_two_options_from_cache(
                                    model,
                                    processor,
                                    nc_cache_ab,
                                    gold,
                                    wrong,
                                    gold_token_ids=s.get("gold_token_ids"),
                                    wrong_token_ids=s.get("wrong_token_ids"),
                                )
                                ctx_ab_scores = score_two_options_from_cache(
                                    model,
                                    processor,
                                    ctx_cache_ab,
                                    gold,
                                    wrong,
                                    gold_token_ids=s.get("gold_token_ids"),
                                    wrong_token_ids=s.get("wrong_token_ids"),
                                )
                                ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                                eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                                base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                                n += 1
                            rerank_item = {"layer": layer_idx, "head": head_idx, "coarse_mean_abs_effect_reduction": eff_red_sum / max(n, 1), "coarse_mean_abs_base_change": base_change_sum / max(n, 1)}
                            coarse_rerank_scores_map[(layer_idx, head_idx)] = rerank_item
                            tracker.append_record(resume_results_file_name(round_name, rerank_stage_tag), rerank_item)
                            tracker.mark_done(resume_key_for_head(round_name, rerank_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": rerank_stage_tag, "round": round_name})
                    finally:
                        remove_hooks(handles)
                rerank_head_set = {(int(item["layer"]), int(item["head"])) for item in rerank_pool}
                coarse_rerank_scores = sort_stage_results(rerank_stage_tag, [item for key, item in coarse_rerank_scores_map.items() if key in rerank_head_set])
                candidate_heads = [(x["layer"], x["head"]) for x in coarse_rerank_scores[:keep_n]]
            else:
                candidate_heads = [(x["layer"], x["head"]) for x in coarse_scores]

            refine_stage_tag = "full_refine"
            results_map = load_stage_resume_results(tracker, round_name, refine_stage_tag)
            records_map = load_stage_resume_records(tracker, round_name, refine_stage_tag) if args.save_records else {}
            pending_refine_heads = []
            for layer_idx, head_idx in candidate_heads:
                head_key = (int(layer_idx), int(head_idx))
                resume_key = resume_key_for_head(round_name, refine_stage_tag, head_key[0], head_key[1])
                if tracker.is_done(resume_key) and head_key in results_map:
                    continue
                pending_refine_heads.append(head_key)
            pending_refine_groups = build_head_batch_groups(pending_refine_heads, head_batch_size)
            for head_group in tqdm(pending_refine_groups, desc=f"Refine {round_name}", unit="group"):
                layer2heads = {}
                for layer_idx, head_idx in head_group:
                    layer2heads.setdefault(int(layer_idx), []).append(int(head_idx))
                handles = install_head_ablation_hooks(model, layer2heads, keep_mode=args.keep_mode, impl=args.head_ablation_impl)
                try:
                    if len(head_group) > 1:
                        expanded_samples, _ = expand_samples_for_head_group(samples, head_group)
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            expanded_samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=resolve_group_eval_batch_size(model, qwen_prefill_batch_size, len(head_group)),
                            ablation_controller=handles,
                        )
                        grouped_scores = summarize_head_group_scores(
                            head_group,
                            ablated_scores,
                            samples,
                            metrics,
                            weights,
                            args.trace_mode,
                            include_nc=True,
                        )
                        sample_count = len(samples)
                        for head_pos, (layer_idx, head_idx) in enumerate(head_group):
                            group_item = grouped_scores[(int(layer_idx), int(head_idx))]
                            item = {
                                "layer": int(layer_idx),
                                "head": int(head_idx),
                                "mean_abs_effect_reduction": group_item["mean_abs_effect_reduction"],
                                "mean_abs_base_change": group_item["mean_abs_base_change"],
                                "metric_mean_abs_effect_reduction": group_item["metric_mean_abs_effect_reduction"],
                                "metric_mean_abs_base_change": group_item["metric_mean_abs_base_change"],
                            }
                            results_map[(int(layer_idx), int(head_idx))] = item
                            tracker.append_record(resume_results_file_name(round_name, refine_stage_tag), item)
                            if args.save_records:
                                start = head_pos * sample_count
                                end = start + sample_count
                                chunk_nc = ablated_scores["nc_inputs"][start:end]
                                chunk_ctx = ablated_scores["ctx_inputs"][start:end]
                                per_head_records = []
                                for sample_idx, s in enumerate(samples):
                                    gold = s["gold"]
                                    wrong = s["wrong"]
                                    nc_ab_scores = chunk_nc[sample_idx]
                                    ctx_ab_scores = chunk_ctx[sample_idx]
                                    ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                                    effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                                    per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                                record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                                records_map[(int(layer_idx), int(head_idx))] = record_item
                                tracker.append_record(resume_records_file_name(round_name, refine_stage_tag), record_item)
                            tracker.mark_done(resume_key_for_head(round_name, refine_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": refine_stage_tag, "round": round_name})
                    elif qwen_accel_mode:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        metric_eff_red = {m: 0.0 for m in metrics}
                        metric_base_change = {m: 0.0 for m in metrics}
                        per_head_records = []
                        ablated_scores = score_samples_for_prompt_keys_qwen(
                            model,
                            processor,
                            samples,
                            ["nc_inputs", "ctx_inputs"],
                            requested_batch_size=qwen_prefill_batch_size,
                            ablation_controller=handles,
                        )
                        for sample_idx, s in enumerate(samples):
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_ab_scores = ablated_scores["nc_inputs"][sample_idx]
                            ctx_ab_scores = ablated_scores["ctx_inputs"][sample_idx]
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            for m in metrics:
                                nc_m_base = s["base_metric_values"][m]["nc"]
                                ctx_m_ab = metric_from_scores(m, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_m_ab = metric_from_scores(m, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_m_ab = ctx_m_ab - nc_m_base
                                metric_eff_red[m] += abs(s["base_metric_values"][m]["effect"]) - abs(effect_m_ab)
                                metric_base_change[m] += abs(nc_m_ab - nc_m_base)
                            if args.save_records:
                                per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                            del nc_cache_ab
                            del ctx_cache_ab
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "mean_abs_effect_reduction": eff_red_sum / max(n, 1), "mean_abs_base_change": base_change_sum / max(n, 1), "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(n, 1) for m in metrics}, "metric_mean_abs_base_change": {m: metric_base_change[m] / max(n, 1) for m in metrics}}
                        results_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, refine_stage_tag), item)
                        if args.save_records:
                            record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                            records_map[(int(layer_idx), int(head_idx))] = record_item
                            tracker.append_record(resume_records_file_name(round_name, refine_stage_tag), record_item)
                        tracker.mark_done(resume_key_for_head(round_name, refine_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": refine_stage_tag, "round": round_name})
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        layer_idx, head_idx = head_group[0]
                        eff_red_sum = 0.0
                        base_change_sum = 0.0
                        n = 0
                        metric_eff_red = {m: 0.0 for m in metrics}
                        metric_base_change = {m: 0.0 for m in metrics}
                        per_head_records = []
                        for s in samples:
                            gold = s["gold"]
                            wrong = s["wrong"]
                            nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                            ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                            nc_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                nc_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scores = score_two_options_from_cache(
                                model,
                                processor,
                                ctx_cache_ab,
                                gold,
                                wrong,
                                gold_token_ids=s.get("gold_token_ids"),
                                wrong_token_ids=s.get("wrong_token_ids"),
                            )
                            ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                            nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)
                            effect_ab = ctx_ab_scalar - s["base_nc_scalar"]
                            eff_red_sum += abs(s["base_effect_scalar"]) - abs(effect_ab)
                            base_change_sum += abs(nc_ab_scalar - s["base_nc_scalar"])
                            for m in metrics:
                                nc_m_base = s["base_metric_values"][m]["nc"]
                                ctx_m_ab = metric_from_scores(m, ctx_ab_scores, gold, wrong, args.trace_mode)
                                nc_m_ab = metric_from_scores(m, nc_ab_scores, gold, wrong, args.trace_mode)
                                effect_m_ab = ctx_m_ab - nc_m_base
                                metric_eff_red[m] += abs(s["base_metric_values"][m]["effect"]) - abs(effect_m_ab)
                                metric_base_change[m] += abs(nc_m_ab - nc_m_base)
                            if args.save_records:
                                per_head_records.append({"question": s["question"], "gold": gold, "wrong": wrong, "image_path": s["image_path"], "base_nc": s["nc_scores"], "base_ctx": s["ctx_scores"], "ablated_nc": nc_ab_scores, "ablated_ctx": ctx_ab_scores, "base_effect_scalar": s["base_effect_scalar"], "ablated_effect_scalar": effect_ab})
                            n += 1
                        item = {"layer": int(layer_idx), "head": int(head_idx), "mean_abs_effect_reduction": eff_red_sum / max(n, 1), "mean_abs_base_change": base_change_sum / max(n, 1), "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(n, 1) for m in metrics}, "metric_mean_abs_base_change": {m: metric_base_change[m] / max(n, 1) for m in metrics}}
                        results_map[(int(layer_idx), int(head_idx))] = item
                        tracker.append_record(resume_results_file_name(round_name, refine_stage_tag), item)
                        if args.save_records:
                            record_item = {"layer": int(layer_idx), "head": int(head_idx), "records": per_head_records}
                            records_map[(int(layer_idx), int(head_idx))] = record_item
                            tracker.append_record(resume_records_file_name(round_name, refine_stage_tag), record_item)
                        tracker.mark_done(resume_key_for_head(round_name, refine_stage_tag, layer_idx, head_idx), {"status": "ok", "stage": refine_stage_tag, "round": round_name})
                finally:
                    remove_hooks(handles)
            candidate_head_set = {(int(layer_idx), int(head_idx)) for layer_idx, head_idx in candidate_heads}
            results = sort_stage_results(refine_stage_tag, [item for key, item in results_map.items() if key in candidate_head_set])
            payload = {"data_csv": args.data_csv, "image_root": args.image_root, "model": args.model, "trace_mode": args.trace_mode, "position": args.position, "keep_mode": args.keep_mode, "metrics": metrics, "weights": weights, "n_samples": len(samples), "skipped": skipped, "n_layers": n_layers, "n_heads": n_heads, "round": round_name, "scan_layers": scan_layers, "coarse": {"coarse_limit": args.coarse_limit, "coarse_with_nc": bool(args.coarse_with_nc), "coarse_seed": int(args.coarse_seed), "coarse_topk": args.coarse_topk, "n_all_heads": len(all_heads), "n_coarse_samples": len(coarse_samples), "n_rerank_heads": len(coarse_rerank_scores), "n_refined_heads": len(candidate_heads), "parallel_mode": "single_process_accel", "num_shards": 1}, "coarse_top20": coarse_scores[:20], "coarse_rerank_top20": coarse_rerank_scores[:20], "top20": results[:20], "results": results}
            if args.save_records:
                payload["records"] = [records_map[k] for k in sorted(records_map) if k in candidate_head_set]
            Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return payload

        raise SystemExit(f"Unsupported parallel_stage for this wrapper path: {stage}")
    if args.plan:
        _, round_layers = load_plan_layers(args.plan, args.round or None)
        Path(args.plan_out_dir).mkdir(parents=True, exist_ok=True)

        if args.parallel_stage in {"coarse", "coarse_rerank", "refine"}:
            for rname, scan_layers in round_layers:
                out_path = str(Path(args.plan_out_dir) / f"head_scan_{rname}.json")
                run_round(scan_layers, out_path, rname)
            print(f"[OK] Completed stage {args.parallel_stage} for {len(round_layers)} round(s)")
            return

        all_rounds = []
        for rname, scan_layers in round_layers:
            out_path = str(Path(args.plan_out_dir) / f"head_scan_{rname}.json")
            run_round(scan_layers, out_path, rname)
            saved_out_path = out_path
            if args.num_shards > 1:
                saved_out_path = build_shard_file_path(saved_out_path, args.shard_idx, args.num_shards)
            all_rounds.append({"round": rname, "result_path": saved_out_path, "scan_layers": scan_layers})

        summary_path = str(Path(args.plan_out_dir) / "head_scan_summary.json")
        if args.num_shards > 1:
            summary_path = build_shard_file_path(summary_path, args.shard_idx, args.num_shards)
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
                "coarse_limit": args.coarse_limit,
                "coarse_topk": (0 if args.num_shards > 1 else args.coarse_topk),
                "coarse_with_nc": bool(args.coarse_with_nc),
                "coarse_seed": int(args.coarse_seed),
                "plan": args.plan,
                "parallel_mode": ("head_shard_exact" if args.num_shards > 1 else "single_process_accel"),
                "num_shards": int(args.num_shards),
                "shard_idx": int(args.shard_idx),
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


