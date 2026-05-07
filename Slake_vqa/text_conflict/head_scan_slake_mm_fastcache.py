import os
import json
import argparse
import importlib.util
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
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)
SYSTEM_PROMPT = "You are a helpful medical QA assistant."
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
    base_inputs = move_to_device(base_inputs, device)
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
def score_two_options_from_cache(model, processor, cache_pack, gold: str, wrong: str):
    gold_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + gold)
    wrong_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + wrong)
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

        # Uniformly wrap new-schema layers as a pseudo round.
        round_layers = [("merged_unique_layers", merged_layers)]

    if round_name is None:
        return plan, round_layers

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

    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["cc", "conflict"])
    ap.add_argument("--position", type=str, default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])

    ap.add_argument("--metrics", type=str, default="gold_wrong_margin,follow_context")
    ap.add_argument("--weights", type=str, default="")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan", type=str, default="")
    ap.add_argument("--round", default='merged_unique_layers', help="optional group name; for new scan_plan use 'merged_unique_layers'")
    ap.add_argument("--layers", type=str, default="")
    ap.add_argument("--plan_out_dir", type=str, default="result/headscan_slake_mm")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--save_records", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.plan_out_dir = prefix_model_relative_path(args.plan_out_dir, model_name=args.model_name, model=args.model)

    metrics, weights = parse_metric_weights(args.metrics, args.weights)

    df = pd.read_csv(args.data_csv)
    if "gold" not in df.columns:
        raise ValueError("CSV must contain a 'gold' column.")
    if "wrong" not in df.columns:
        raise ValueError("CSV must contain a 'wrong' column.")
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

        nc_inputs = build_inputs_mm(processor, nc_prompt, image, model.device)
        ctx_inputs = build_inputs_mm(processor, ctx_prompt, image, model.device)

        nc_cache = build_prompt_cache_from_inputs(model, nc_inputs)
        ctx_cache = build_prompt_cache_from_inputs(model, ctx_inputs)
        nc_scores = score_two_options_from_cache(model, processor, nc_cache, gold, wrong)
        ctx_scores = score_two_options_from_cache(model, processor, ctx_cache, gold, wrong)

        base_nc_scalar = weighted_metric(metrics, weights, nc_scores, gold, wrong, args.trace_mode)
        base_ctx_scalar = weighted_metric(metrics, weights, ctx_scores, gold, wrong, args.trace_mode)
        base_effect_scalar = base_ctx_scalar - base_nc_scalar

        sample = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "image_path": str(img_path),
            "nc_inputs": nc_inputs,
            "ctx_inputs": ctx_inputs,
            "nc_scores": nc_scores,
            "ctx_scores": ctx_scores,
            "base_nc_scalar": base_nc_scalar,
            "base_ctx_scalar": base_ctx_scalar,
            "base_effect_scalar": base_effect_scalar,
            "base_metric_values": {
                m: {
                    "nc": metric_from_scores(m, nc_scores, gold, wrong, args.trace_mode),
                    "ctx": metric_from_scores(m, ctx_scores, gold, wrong, args.trace_mode),
                    "effect": metric_from_scores(m, ctx_scores, gold, wrong, args.trace_mode)
                              - metric_from_scores(m, nc_scores, gold, wrong, args.trace_mode),
                }
                for m in metrics
            },
        }
        samples.append(sample)

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
        plan_out_dir=args.plan_out_dir,
        out=args.out,
        save_records=bool(args.save_records),
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
                handles = install_head_mask_hooks(model, {int(layer_idx): [int(head_idx)]}, keep_mode=args.keep_mode)
                try:
                    eff_red_sum = 0.0
                    base_change_sum = 0.0
                    n = 0
                    metric_eff_red = {m: 0.0 for m in metrics}
                    metric_base_change = {m: 0.0 for m in metrics}

                    per_head_records = []
                    for s in samples:
                        gold = s["gold"]
                        wrong = s["wrong"]

                        # Rebuild prompt cache under current ablation, then score continuation.
                        nc_cache_ab = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                        ctx_cache_ab = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
                        nc_ab_scores = score_two_options_from_cache(model, processor, nc_cache_ab, gold, wrong)
                        ctx_ab_scores = score_two_options_from_cache(model, processor, ctx_cache_ab, gold, wrong)

                        ctx_ab_scalar = weighted_metric(metrics, weights, ctx_ab_scores, gold, wrong, args.trace_mode)
                        nc_ab_scalar = weighted_metric(metrics, weights, nc_ab_scores, gold, wrong, args.trace_mode)

                        # Keep consistent with layer_trace: use original NC as fixed baseline.
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
                            per_head_records.append({
                                "question": s["question"],
                                "gold": gold,
                                "wrong": wrong,
                                "image_path": s["image_path"],
                                "base_nc": s["nc_scores"],
                                "base_ctx": s["ctx_scores"],
                                "ablated_nc": nc_ab_scores,
                                "ablated_ctx": ctx_ab_scores,
                                "base_effect_scalar": s["base_effect_scalar"],
                                "ablated_effect_scalar": effect_ab,
                            })
                        n += 1
                finally:
                    remove_hooks(handles)

                item = {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "mean_abs_effect_reduction": eff_red_sum / max(n, 1),
                    "mean_abs_base_change": base_change_sum / max(n, 1),
                    "metric_mean_abs_effect_reduction": {m: metric_eff_red[m] / max(n, 1) for m in metrics},
                    "metric_mean_abs_base_change": {m: metric_base_change[m] / max(n, 1) for m in metrics},
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
                        "records": per_head_records,
                    }
                    records.append(record_item)
                    tracker.append_record(f"{round_name}_records.jsonl", record_item)

                tracker.update(
                    current_round=round_name,
                    current_output=out_path,
                    completed_heads=len(results),
                    total_heads=sum(len(range(n_heads)) for _ in scan_layers),
                )

        results.sort(key=lambda r: (-(r["mean_abs_effect_reduction"]), r["mean_abs_base_change"], r["layer"], r["head"]))
        out = {
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "model": args.model,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "keep_mode": args.keep_mode,
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
        _, round_layers = load_plan_layers(args.plan, args.round or None)
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

