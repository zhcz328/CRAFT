#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import copy
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq

SYSTEM_PROMPT = "You are a helpful medical QA assistant."
BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)
EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)


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


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    # transformers BatchEncoding/BatchFeature expose .to(device)
    if hasattr(obj, "to") and callable(getattr(obj, "to")):
        try:
            return obj.to(device)
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    return obj


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if len(xs) > 0 else 0.0


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

def _make_head_mask_pre_hook(layer_idx: int, heads_to_mask: List[int], num_heads: int, keep_mode: str = "self"):
    heads_to_mask = sorted(set(int(h) for h in heads_to_mask))

    def _hook(module, args, kwargs):
        kwargs = {} if kwargs is None else kwargs

        attn_mask = None
        from_kwargs = False
        if "attention_mask" in kwargs:
            attn_mask = kwargs["attention_mask"]
            from_kwargs = True
        elif len(args) >= 2:
            attn_mask = args[1]

        if attn_mask is None:
            return

        # 期望支持 [bs, 1, q, k] 或 [bs, num_heads, q, k]
        if attn_mask.dim() != 4:
            return

        bsz = attn_mask.shape[0]
        q_len = attn_mask.shape[-2]
        k_len = attn_mask.shape[-1]
        device = attn_mask.device
        dtype = attn_mask.dtype

        neg = torch.finfo(dtype).min

        # 扩成 [bs, num_heads, q, k]
        if attn_mask.shape[1] == 1:
            expanded = attn_mask.expand(bsz, num_heads, q_len, k_len).clone()
        elif attn_mask.shape[1] == num_heads:
            expanded = attn_mask.clone()
        else:
            return

        for h in heads_to_mask:
            if h < 0 or h >= num_heads:
                continue
            expanded[:, h, :, :] = neg

            if keep_mode == "self":
                diag_len = min(q_len, k_len)
                ar = torch.arange(diag_len, device=device)
                expanded[:, h, ar, ar] = 0.0
            elif keep_mode == "bos":
                expanded[:, h, :, 0] = 0.0
            else:
                raise ValueError(f"Unsupported keep_mode: {keep_mode}")

        if from_kwargs:
            kwargs["attention_mask"] = expanded
            return args, kwargs

        args = list(args)
        if len(args) >= 2:
            args[1] = expanded
            return tuple(args), kwargs

    return _hook


def install_head_mask_hooks(model, layer_to_heads: Dict[int, List[int]], keep_mode="self"):
    layer2attn, _ = _find_self_attn_modules(model)
    n_heads, _ = _get_num_heads_and_hidden(model)
    if n_heads is None:
        raise RuntimeError("Missing num_attention_heads/n_head in config or nested text config")

    handles = []
    for layer_idx, heads in layer_to_heads.items():
        if layer_idx not in layer2attn:
            continue
        mod = layer2attn[layer_idx]
        hook = _make_head_mask_pre_hook(layer_idx, heads, n_heads, keep_mode=keep_mode)
        h = mod.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(h)
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
    p = Path(img_rel)
    if p.is_file():
        return str(p)
    p2 = Path(image_root) / img_rel
    if p2.is_file():
        return str(p2)
    raise FileNotFoundError(f"Image not found: {img_rel} or {p2}")


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
    # Backward compatibility for old CLI value.
    if position == "after_question":
        return "before_answer"
    return position


def inject_evidence(base_prompt: str, evidence_block: str, position: str) -> str:
    if position == "prefix":
        return evidence_block + "\n" + base_prompt

    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    raise ValueError(f"Unsupported position: {position}")


def make_prompts(question: str, gold: str, wrong: str, position: str, trace_mode: str):
    question = normalize_text(question)
    pos = normalize_position(position)

    nc_prompt = BASE_RULE + f"Question: {question}\nAnswer:"

    evidence_ans = gold if trace_mode == "cc" else wrong
    evidence_block = EVIDENCE_TMPL.format(ans=evidence_ans)
    ctx_prompt = inject_evidence(nc_prompt, evidence_block, pos)
    return nc_prompt, ctx_prompt


def build_inputs_mm(processor, image: Image.Image, prompt: str, device: str):
    # Qwen3-VL needs multimodal chat template to align image placeholders/features.
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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
    base_inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
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
    fixed_inputs = to_device(base_inputs, model_device)
    out = model(**fixed_inputs, use_cache=True, return_dict=True)
    prompt_last_logits = out.logits[:, -1, :].detach()
    return {
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
        return 0.0

    total_lp = 0.0
    first_id = cont_ids[0, 0].item()

    first_logprobs = F.log_softmax(prompt_cache["prompt_last_logits"], dim=-1)
    total_lp += first_logprobs[0, first_id].item()

    if cont_ids.shape[1] == 1:
        return float(total_lp)

    past = prompt_cache["past_key_values"]
    rest_inp = cont_ids[:, :-1]  # 让模型预测第2..T个token
    out = model(
        input_ids=rest_inp,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
    )
    logits = out.logits  # [1, T-1, vocab]
    rest_target = cont_ids[:, 1:]  # [1, T-1]
    lp = F.log_softmax(logits, dim=-1)
    gather_lp = lp.gather(-1, rest_target.unsqueeze(-1)).squeeze(-1).sum().item()
    total_lp += gather_lp
    return float(total_lp)


@torch.no_grad()
def score_answer_candidates_with_cache(
    model,
    tokenizer,
    prompt_cache: Dict[str, Any],
    gold: str,
    wrong: str,
    trace_mode: str,
    device: str,
):
    gold_lp = score_continuation_with_cache(model, tokenizer, prompt_cache, gold, device)
    wrong_lp = score_continuation_with_cache(model, tokenizer, prompt_cache, wrong, device)

    if trace_mode == "cc":
        ctx_target = gold
        ctx_lp = gold_lp
        other_lp = wrong_lp
    elif trace_mode == "conflict":
        ctx_target = wrong
        ctx_lp = wrong_lp
        other_lp = gold_lp
    else:
        raise ValueError(f"Unsupported trace_mode: {trace_mode}")

    return {
        "gold_lp": float(gold_lp),
        "wrong_lp": float(wrong_lp),
        "ctx_target": ctx_target,
        "ctx_lp": float(ctx_lp),
        "other_lp": float(other_lp),
    }


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

    out = {}
    out["gold_wrong_margin"] = float(gold_lp - wrong_lp)
    out["follow_conflict"] = float(wrong_lp - gold_lp)
    out["follow_context"] = float(ctx_lp - other_lp)
    out["context_flip"] = float(other_lp - ctx_lp)
    return out


def pred_label(scores: Dict[str, float]) -> str:
    return "gold" if scores["gold_lp"] >= scores["wrong_lp"] else "wrong"


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


# =========================================================
# 样本构建
# =========================================================

def build_samples(args, df, processor, tokenizer, model, device):
    image_col = infer_col(df, ["image", "image_path", "img_path", "img", "path"])
    q_col = infer_col(df, ["question", "query"])
    gold_col = infer_col(df, ["answer", "gold", "gt_answer", "label"])
    wrong_col = infer_col(df, ["wrong", "conflict", "wrong_answer"], required=False)

    samples = []

    iterable = df.iterrows()
    if args.max_examples > 0:
        iterable = list(df.head(args.max_examples).iterrows())

    for idx, row in tqdm(iterable, total=(len(df.head(args.max_examples)) if args.max_examples > 0 else len(df)), desc="Build samples"):
        question = normalize_text(row[q_col])
        gold = normalize_text(row[gold_col]).lower()

        if wrong_col is not None and normalize_text(row[wrong_col]):
            wrong = normalize_text(row[wrong_col]).lower()
        else:
            wrong = invert_yesno(gold)

        if not question or not gold or not wrong or gold == wrong:
            continue

        try:
            img_path = resolve_image_path(args.image_root, normalize_text(row[image_col]))
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        nc_prompt, ctx_prompt = make_prompts(
            question=question,
            gold=gold,
            wrong=wrong,
            position=args.position,
            trace_mode=args.trace_mode,
        )

        try:
            nc_inputs = build_inputs_mm(processor, image, nc_prompt, device)
            ctx_inputs = build_inputs_mm(processor, image, ctx_prompt, device)
        except Exception:
            continue

        # baseline cache
        nc_cache = build_prompt_cache_from_inputs(model, nc_inputs)
        ctx_cache = build_prompt_cache_from_inputs(model, ctx_inputs)

        nc_scores = score_answer_candidates_with_cache(
            model, tokenizer, nc_cache, gold, wrong, args.trace_mode, device
        )
        ctx_scores = score_answer_candidates_with_cache(
            model, tokenizer, ctx_cache, gold, wrong, args.trace_mode, device
        )

        nc_metrics = compute_scalar_metrics(nc_scores, args.trace_mode)
        ctx_metrics = compute_scalar_metrics(ctx_scores, args.trace_mode)

        samples.append({
            "row_idx": int(idx),
            "img_path": img_path,
            "question": question,
            "context": "",
            "gold": gold,
            "wrong": wrong,
            "nc_prompt": nc_prompt,
            "ctx_prompt": ctx_prompt,
            "nc_inputs": nc_inputs,
            "ctx_inputs": ctx_inputs,
            "base_nc_scores": nc_scores,
            "base_ctx_scores": ctx_scores,
            "base_nc_metrics": nc_metrics,
            "base_ctx_metrics": ctx_metrics,
            "base_nc_pred": pred_label(nc_scores),
            "base_ctx_pred": pred_label(ctx_scores),
        })

    return samples


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
    ap.add_argument("--selected_heads", type=str, required=True)

    ap.add_argument("--out_json", type=str, required=True)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["cc", "conflict"])
    ap.add_argument(
        "--position",
        type=str,
        default="before_question",
        choices=["prefix", "before_question", "before_answer", "after_question"],
    )

    ap.add_argument("--metrics", type=str, default="gold_wrong_margin,follow_context")
    ap.add_argument("--mask_scope", type=str, default="all", choices=["all", "ctx_only"])
    ap.add_argument("--keep_mode", type=str, default="self", choices=["self", "bos"])

    ap.add_argument("--random_ablate", action="store_true")
    ap.add_argument("--random_seed", type=int, default=0)

    args = ap.parse_args()
    args.position = normalize_position(args.position)

    set_seed(args.seed)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    dtype = str2dtype(args.dtype)

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    for m in metrics:
        if m not in ALLOWED_METRICS:
            raise ValueError(f"Unsupported metric: {m}")

    df = pd.read_csv(args.data_csv)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # 尝试两种加载方式
    model = None
    load_errs = []
    for cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
        try:
            model = cls.from_pretrained(
                args.model,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            break
        except Exception as e:
            load_errs.append(f"{cls.__name__}: {repr(e)}")
    if model is None:
        raise RuntimeError("Failed to load model. " + " | ".join(load_errs))

    model = model.to(device)
    model.eval()

    samples = build_samples(args, df, processor, tokenizer, model, device)
    if len(samples) == 0:
        raise RuntimeError("No valid samples were built.")

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

    records = []
    try:
        for s in tqdm(samples, desc="Ablate selected heads"):
            # mask_scope: all => nc/ctx 都重算
            # ctx_only => 只重算 ctx，nc 用 baseline
            if args.mask_scope == "all":
                ab_nc_cache = build_prompt_cache_from_inputs(model, s["nc_inputs"])
                ab_nc_scores = score_answer_candidates_with_cache(
                    model, tokenizer, ab_nc_cache, s["gold"], s["wrong"], args.trace_mode, device
                )
                ab_nc_metrics = compute_scalar_metrics(ab_nc_scores, args.trace_mode)
                ab_nc_pred = pred_label(ab_nc_scores)
            else:
                ab_nc_scores = copy.deepcopy(s["base_nc_scores"])
                ab_nc_metrics = copy.deepcopy(s["base_nc_metrics"])
                ab_nc_pred = s["base_nc_pred"]

            ab_ctx_cache = build_prompt_cache_from_inputs(model, s["ctx_inputs"])
            ab_ctx_scores = score_answer_candidates_with_cache(
                model, tokenizer, ab_ctx_cache, s["gold"], s["wrong"], args.trace_mode, device
            )
            ab_ctx_metrics = compute_scalar_metrics(ab_ctx_scores, args.trace_mode)
            ab_ctx_pred = pred_label(ab_ctx_scores)

            metric_effect_reduction_fixed_nc = {}
            metric_effect_reduction_mask_scope = {}
            metric_abs_base_change = {}
            metric_abs_ctx_change = {}

            for m in metrics:
                base_effect = s["base_ctx_metrics"][m] - s["base_nc_metrics"][m]
                effect_fixed_nc = ab_ctx_metrics[m] - s["base_nc_metrics"][m]
                effect_mask_scope = ab_ctx_metrics[m] - ab_nc_metrics[m]

                metric_effect_reduction_fixed_nc[m] = float(abs(base_effect) - abs(effect_fixed_nc))
                metric_effect_reduction_mask_scope[m] = float(abs(base_effect) - abs(effect_mask_scope))
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

                "metric_effect_reduction_fixed_nc": metric_effect_reduction_fixed_nc,
                "metric_effect_reduction_mask_scope": metric_effect_reduction_mask_scope,
                "metric_abs_base_change": metric_abs_base_change,
                "metric_abs_ctx_change": metric_abs_ctx_change,
            }
            records.append(rec)

    finally:
        remove_handles(handles)

    # summary
    all_fixed = []
    all_scope = []
    all_base_ch = []
    all_ctx_ch = []

    for r in records:
        for m in metrics:
            all_fixed.append(r["metric_effect_reduction_fixed_nc"][m])
            all_scope.append(r["metric_effect_reduction_mask_scope"][m])
            all_base_ch.append(r["metric_abs_base_change"][m])
            all_ctx_ch.append(r["metric_abs_ctx_change"][m])

    base_nc_correct = 0
    base_ctx_correct = 0
    ab_nc_correct = 0
    ab_ctx_correct = 0

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

    n = len(records)
    metric_summary = summarize_metric_changes(records, metrics)

    summary = {
        "config": {
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "model": args.model,
            "selected_heads": args.selected_heads,
            "trace_mode": args.trace_mode,
            "position": args.position,
            "metrics": metrics,
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "dtype": args.dtype,
            "device": device,
            "max_examples": args.max_examples,
            "random_ablate": bool(args.random_ablate),
            "random_seed": args.random_seed,
        },
        "selected_pairs": [{"layer": l, "head": h} for l, h in selected_pairs],
        "n_selected_heads": len(selected_pairs),
        "n_samples": n,

        "mean_abs_effect_reduction_fixed_nc": safe_mean(all_fixed),
        "mean_abs_effect_reduction_mask_scope": safe_mean(all_scope),
        "mean_abs_base_change": safe_mean(all_base_ch),
        "mean_abs_ctx_change": safe_mean(all_ctx_ch),

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

        "per_metric": metric_summary,
        "records": records,
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "saved_to": args.out_json,
        "n_selected_heads": len(selected_pairs),
        "n_samples": n,
        "mean_abs_effect_reduction_fixed_nc": summary["mean_abs_effect_reduction_fixed_nc"],
        "mean_abs_effect_reduction_mask_scope": summary["mean_abs_effect_reduction_mask_scope"],
        "mean_abs_base_change": summary["mean_abs_base_change"],
        "mean_abs_ctx_change": summary["mean_abs_ctx_change"],
        "pred_ctx_acc_delta": summary["pred_ctx"]["acc_delta"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
