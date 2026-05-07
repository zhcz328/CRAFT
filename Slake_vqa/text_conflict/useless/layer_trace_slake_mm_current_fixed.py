import os
import json
import math
import argparse
import importlib.util
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
import matplotlib.pyplot as plt

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)


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


def load_mm_model(
    model_name,
    device_map=None,
    torch_dtype=None,
    trust_remote_code=True,
    attn_implementation=None,
    quantization_config=None,
):
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
            raise ValueError("base prompt missing '\nQuestion:' marker")
        return base.replace(marker, "\n" + ev + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base:
            raise ValueError("base prompt missing '\nAnswer:' marker")
        return base.replace(marker, "\n" + ev + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


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
    """
    Robustly locate decoder layers across different HF model wrappers.
    """
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
        f"Cannot locate decoder layers from model type: {type(m)}. "
        f"Tried paths: {tried}"
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


def build_inputs(processor, prompt: str, image: Image.Image, device: str):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = build_processor_inputs_mm(processor, text, image)
    if not has_non_video_image_inputs(inputs):
        keys = sorted(list(inputs.keys()))
        raise RuntimeError(f"No image-related inputs found. keys={keys}")
    return move_to_device(inputs, device)


@torch.inference_mode()
def get_prompt_cache(model, processor, prompt: str, image: Image.Image, device: str):
    inputs = build_inputs(processor, prompt, image, device)
    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(inputs, model.device, float_dtype=target_dtype)
    out = model(**fixed_inputs, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hs = out.hidden_states
    attn_mask = fixed_inputs.get("attention_mask", None)
    return fixed_inputs, out, cache, hs, attn_mask


@torch.inference_mode()
def continuation_logprob(model, processor, base_inputs, base_cache, continuation: str, device: str) -> float:
    cont_ids = processor.tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if cont_ids.numel() == 0:
        return float("-inf")

    past = base_cache
    total_logprob = 0.0
    prev_input = cont_ids[:, :1]

    for t in range(cont_ids.shape[1]):
        if t == 0:
            outputs = model(input_ids=prev_input, past_key_values=past, use_cache=True)
        else:
            outputs = model(input_ids=cont_ids[:, t:t+1], past_key_values=past, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
        if t + 1 < cont_ids.shape[1]:
            target = cont_ids[:, t + 1]
            logp = torch.log_softmax(logits, dim=-1)[0, target.item()].item()
            total_logprob += logp

    return float(total_logprob)


@torch.inference_mode()
def score_candidates(model, processor, prompt: str, image: Image.Image, gold: str, wrong: str, device: str):
    _, _, cache, hs, _ = get_prompt_cache(model, processor, prompt, image, device)
    gold_lp = continuation_logprob(model, processor, None, cache, " " + gold, device)
    wrong_lp = continuation_logprob(model, processor, None, cache, " " + wrong, device)
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "gold_adv": gold_lp - wrong_lp,
        "wrong_adv": wrong_lp - gold_lp,
        "pred": gold if gold_lp >= wrong_lp else wrong,
        "hidden_states": hs,
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


@torch.inference_mode()
def patched_score_candidates(model, processor, prompt: str, image: Image.Image, gold: str, wrong: str,
                             device: str, layer_idx: int, source_tail: torch.Tensor, patch_k: int):
    with HiddenPatcher(model, layer_idx, source_tail, patch_k):
        return score_candidates(model, processor, prompt, image, gold, wrong, device)


def metric_from_scores(metric: str, scores: dict, gold: str, wrong: str, trace_mode: str) -> float:
    """
    Available metrics:
      - gold_wrong_margin: score(gold) - score(wrong)
      - follow_conflict:   score(wrong) - score(gold)
      - follow_context:    score(context_target) - score(other_target)
                           where context_target = gold for cc, wrong for conflict
      - context_flip:      score(other_target) - score(context_target)
    """
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
        f"Unknown metric: {metric}. "
        f"Allowed: ['gold_wrong_margin', 'follow_conflict', 'follow_context', 'context_flip']"
    )


def plot_scores(layer_scores: Dict[str, List[float]], out_prefix: str):
    os.makedirs(Path(out_prefix).parent, exist_ok=True)
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
    ap.add_argument("--patch_k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", type=str, default="gold_adv,follow_evidence")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot_prefix", type=str, default="")
    ap.add_argument("--scan_plan_out", type=str, default="")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out = prefix_model_relative_path(args.out, model_name=args.model_name, model=args.model)
    if args.plot_prefix:
        args.plot_prefix = prefix_model_relative_path(args.plot_prefix, model_name=args.model_name, model=args.model)
    if args.scan_plan_out:
        args.scan_plan_out = prefix_model_relative_path(args.scan_plan_out, model_name=args.model_name, model=args.model)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
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

    layer_deltas = {m: [0.0 for _ in range(n_layers)] for m in metrics}
    used = 0
    skipped = 0
    records = []

    rows = df.to_dict("records")
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    for row in tqdm(rows, desc="Tracing"):
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

        nc_scores = score_candidates(model, processor, nc_prompt, image, gold, wrong, model.device)
        ctx_scores = score_candidates(model, processor, ctx_prompt, image, gold, wrong, model.device)
        source_tail = nc_scores["hidden_states"][-1]

        rec = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "image_path": str(img_path),
            "trace_mode": args.trace_mode,
            "nc": {k: v for k, v in nc_scores.items() if k != "hidden_states"},
            "ctx": {k: v for k, v in ctx_scores.items() if k != "hidden_states"},
            "layers": {},
        }

        base_effect = {m: metric_from_scores(m, ctx_scores, gold, wrong , args.trace_mode) - metric_from_scores(m, nc_scores, gold, wrong , args.trace_mode) for m in metrics}

        for layer_idx in range(n_layers):
            patched = patched_score_candidates(
                model, processor, ctx_prompt, image, gold, wrong,
                model.device, layer_idx, source_tail, args.patch_k
            )
            rec["layers"][str(layer_idx)] = {k: v for k, v in patched.items() if k != "hidden_states"}
            for m in metrics:
                patched_effect = metric_from_scores(m, patched, gold, wrong , args.trace_mode) - metric_from_scores(m, nc_scores, gold, wrong ,args.trace_mode)
                delta = abs(base_effect[m]) - abs(patched_effect)
                layer_deltas[m][layer_idx] += float(delta)

        records.append(rec)
        used += 1

    if used > 0:
        for m in metrics:
            layer_deltas[m] = [x / used for x in layer_deltas[m]]

    out = {
        "config": vars(args),
        "used": used,
        "skipped": skipped,
        "layer_scores": layer_deltas,
        "records": records,
    }

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    if args.plot_prefix:
        plot_scores(layer_deltas, args.plot_prefix)

    if args.scan_plan_out:
        plan = {
            "trace_mode": args.trace_mode,
            "suggested_top_layers": {
                m: sorted(range(n_layers), key=lambda i: layer_deltas[m][i], reverse=True)[: min(20, n_layers)]
                for m in metrics
            },
            "layer_scores": layer_deltas,
        }
        os.makedirs(Path(args.scan_plan_out).parent, exist_ok=True)
        with open(args.scan_plan_out, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

    print("Saved to:", args.out)
    print(json.dumps({"used": used, "skipped": skipped, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

