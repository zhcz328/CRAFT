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

from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from typing import Any
from typing import Any as TypingAny

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


def load_mm_model(model_name, device_map=None, torch_dtype=None, trust_remote_code=True, attn_implementation=None, quantization_config=None):
    load_errs = []
    for cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
        try:
            kwargs = {"trust_remote_code": trust_remote_code}
            if device_map is not None:
                kwargs["device_map"] = device_map
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            if attn_implementation is not None:
                kwargs["attn_implementation"] = attn_implementation
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as e:
            load_errs.append(f"{cls.__name__}: {repr(e)}")
    raise RuntimeError("Failed to load model. " + " | ".join(load_errs))


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

    valid_layers = [i for i, s in enumerate(scores) if s > 0]
    if not valid_layers:
        return {
            "method": "no_positive_layers",
            "plateau_start": None,
            "round0_rise": [],
            "round1_preplateau": [],
            "round1_topk_fallback": [],
            "round2_tail": [],
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

    # 1) 灏鹃儴楂樺垎灞傦紙鍙繚鐣欐鍒嗭級
    tail_candidates = list(range(max(0, n_layers - tail_k), n_layers))
    tail_layers = [i for i in tail_candidates if scores[i] > 0]

    # 2) 涓婂崌闃舵灞傦紙鍙繚鐣欑洰鏍囧眰涓烘鍒嗭級
    if rise_max_layer is None or rise_max_layer < 0:
        rise_max = max(0, n_layers - 1)
    else:
        rise_max = min(rise_max_layer, n_layers - 1)
    rise_min = max(0, min(rise_min_layer, n_layers - 1))

    diffs = []
    for i in range(0, n_layers - 1):
        if i < rise_min or i >= rise_max:
            continue
        target_layer = i + 1
        if scores[target_layer] <= 0:
            continue
        diffs.append((abs(scores[target_layer] - scores[i]), i))

    diffs.sort(key=lambda x: x[0], reverse=True)
    rise_layers = sorted(set(i + 1 for _, i in diffs[:max(0, rise_k)]))

    # 3) 骞冲彴鍖洪棿
    plateau_start = _find_plateau_start(
        scores,
        plateau_min_score_frac=plateau_min_score_frac,
        plateau_eps=plateau_eps,
        plateau_min_len=plateau_min_len,
    )

    if plateau_start is not None:
        a = max(0, plateau_start - preplateau_width)
        b = min(n_layers - 1, plateau_start + 1)
        plateau_layers = [i for i in range(a, b + 1) if scores[i] > 0]
        method = "plateau"
        topk_layers = []
    else:
        plateau_layers = []
        positive_sorted = [i for i in sorted(range(n_layers), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
        topk_layers = sorted(positive_sorted[:topk_fallback])
        method = "topk_fallback"

    # 4) round 闂撮『搴忓幓閲?    
    used = set()

    round0_rise = [i for i in rise_layers if i not in used]
    used.update(round0_rise)

    round1_preplateau = [i for i in plateau_layers if i not in used]
    used.update(round1_preplateau)

    round1_topk_fallback = [i for i in topk_layers if i not in used]
    used.update(round1_topk_fallback)

    round2_tail = [i for i in tail_layers if i not in used]
    used.update(round2_tail)

    merged_layers = sorted(used)

    return {
        "method": method,
        "plateau_start": plateau_start,
        "round0_rise": round0_rise,
        "round1_preplateau": round1_preplateau,
        "round1_topk_fallback": round1_topk_fallback,
        "round2_tail": round2_tail,
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
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
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
def score_two_options_from_cache(model, processor, cache_pack, gold: str, wrong: str):
    gold_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + gold)
    wrong_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + wrong)
    return {
        "gold_lp": gold_lp,
        "wrong_lp": wrong_lp,
        "pred": gold if gold_lp >= wrong_lp else wrong,
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


@torch.inference_mode()
# def build_patched_prompt_cache_from_inputs(model, base_inputs, layer_idx: int, source_tail: torch.Tensor, patch_k: int):
#     with HiddenPatcher(model, layer_idx, source_tail, patch_k):
#         out = model(**base_inputs, use_cache=True, output_hidden_states=False)

#     input_ids_prompt = base_inputs["input_ids"]
#     attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
#     prompt_last_logits = out.logits[:, -1, :].detach()

#     return {
#         "attn_prompt": attn_prompt,
#         "past_key_values": out.past_key_values,
#         "prompt_last_logits": prompt_last_logits,
#     }
def build_patched_prompt_cache_from_inputs(
    model, 
    base_inputs, 
    layer_idx: int, 
    source_tail: torch.Tensor, 
    patch_k: int,
    patch_all_tokens: bool = False  # 鏂板鍙傛暟
):
    """
    鏋勫缓 patched prompt cache锛屽彲浠ラ€夋嫨 patch 灏鹃儴 k 涓?token 鎴栨暣灞?token
    """
    # 杩欎釜 hook 鍐呴儴浼氭牴鎹?patch_all_tokens 鍐冲畾鏇挎崲澶氬皯 token
    class PatchHook:
        def __init__(self):
            self.handle = None

        def hook_fn(self, module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            hs2 = hs.clone()
            if patch_all_tokens:
                # patch full sequence if available; otherwise patch overlapping tail.
                kk = min(hs2.size(1), source_tail.size(1))
                hs2[:, -kk:, :] = source_tail[:, -kk:, :].to(hs2.device, dtype=hs2.dtype)
            else:
                # patch 鏈€鍚?patch_k 涓?token
                kk = min(patch_k, hs.size(1), source_tail.size(1))
                hs2[:, -kk:, :] = source_tail[:, -kk:, :].to(hs2.device, dtype=hs2.dtype)

            if isinstance(out, (tuple, list)):
                out = list(out)
                out[0] = hs2
                return tuple(out)
            return hs2

    patch_hook = PatchHook()
    layers = _get_layers(model)
    layer = layers[layer_idx]
    handle = layer.register_forward_hook(patch_hook.hook_fn)

    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)

    # forward
    out = model(**fixed_inputs, use_cache=True, output_hidden_states=False)

    # 鍗歌浇 hook
    handle.remove()

    input_ids_prompt = fixed_inputs["input_ids"]
    attn_prompt = fixed_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()

    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
    }

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
def capture_layer_outputs_lastk(model, base_inputs, patch_k, patch_all_tokens: bool = False):
    layers = _get_layers(model)
    cache = [None] * len(layers)
    handles = []

    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                if patch_all_tokens:
                    cache[ii] = hs.detach()
                else:
                    kk = min(patch_k, hs.size(1))
                    cache[ii] = hs[:, -kk:, :].detach()
            return hook
        handles.append(layer.register_forward_hook(make_hook(i)))

    target_dtype = infer_vision_input_dtype(model)
    fixed_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)
    with torch.inference_mode():
        _ = model(**fixed_inputs, use_cache=False, output_hidden_states=False)

    for h in handles:
        h.remove()

    return cache

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trace_mode", type=str, default="conflict", choices=["cc", "conflict"])
    ap.add_argument("--position", type=str, default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--patch_k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", type=str, default="gold_wrong_margin,follow_context")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot_prefix", type=str, default="")
    ap.add_argument("--scan_plan_out", type=str, default="")
    ap.add_argument(
    "--patch_all_tokens",
    action="store_true",
    help="濡傛灉璁剧疆锛宲atch 鏁村眰鎵€鏈?token锛岃€屼笉浠呬粎鏄渶鍚?k 涓?token"
    )
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
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    allowed_metrics = {"gold_wrong_margin", "follow_conflict", "follow_context", "context_flip"}
    for m in metrics:
        if m not in allowed_metrics:
            raise SystemExit(f"Unknown metric '{m}'. Allowed: {sorted(allowed_metrics)}")

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

        nc_inputs = build_inputs_mm(processor, nc_prompt, image, model.device)
        ctx_inputs = build_inputs_mm(processor, ctx_prompt, image, model.device)

        nc_cache = build_prompt_cache_from_inputs(model, nc_inputs, need_hidden_states=True)
        ctx_cache = build_prompt_cache_from_inputs(model, ctx_inputs, need_hidden_states=False)

        nc_scores = score_two_options_from_cache(model, processor, nc_cache, gold, wrong)
        ctx_scores = score_two_options_from_cache(model, processor, ctx_cache, gold, wrong)
        base_cache = capture_layer_outputs_lastk(
            model, nc_inputs, args.patch_k, patch_all_tokens=args.patch_all_tokens
        )

        rec = {
            "question": question,
            "gold": gold,
            "wrong": wrong,
            "image_path": str(img_path),
            "trace_mode": args.trace_mode,
            "nc": dict(nc_scores),
            "ctx": dict(ctx_scores),
            "layers": {},
        }

        base_effect = {
            m: metric_from_scores(m, ctx_scores, gold, wrong, args.trace_mode)
            - metric_from_scores(m, nc_scores, gold, wrong, args.trace_mode)
            for m in metrics
        }

        for layer_idx in range(n_layers):
            patched_cache = build_patched_prompt_cache_from_inputs(
            model, ctx_inputs, layer_idx, base_cache[layer_idx], args.patch_k, patch_all_tokens=args.patch_all_tokens
            )
            patched = score_two_options_from_cache(model, processor, patched_cache, gold, wrong)
            rec["layers"][str(layer_idx)] = dict(patched)
            for m in metrics:
                patched_effect = (
                    metric_from_scores(m, patched, gold, wrong, args.trace_mode)
                    - metric_from_scores(m, nc_scores, gold, wrong, args.trace_mode)
                )
                delta = abs(base_effect[m]) - abs(patched_effect)
                layer_deltas[m][layer_idx] += float(delta)

        records.append(rec)
        used += 1

    if used > 0:
        for m in metrics:
            layer_deltas[m] = [x / used for x in layer_deltas[m]]
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
                raise SystemExit(f"--scan_plan_metric '{plan_metric}' not in computed metrics {metrics}")
            plan_scores = layer_score_mean[plan_metric]
            plan_info = {"mode": "single_metric", "metric": plan_metric, "weights": {plan_metric: 1.0}}

        elif args.scan_plan_metrics:
            plan_metrics = [p.strip() for p in args.scan_plan_metrics.split(",") if p.strip()]
            for pm in plan_metrics:
                if pm not in layer_score_mean:
                    raise SystemExit(f"--scan_plan_metrics includes '{pm}' not in computed metrics {metrics}")
            w = parse_weights(args.scan_plan_weights, len(plan_metrics))
            plan_scores = combine_scores(layer_score_mean, plan_metrics, w)
            plan_info = {
                "mode": "weighted_metrics",
                "metrics": plan_metrics,
                "weights": {m: float(wi) for m, wi in zip(plan_metrics, w)},
            }

        else:
            pm = metrics[0]
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
            "patch_k": args.patch_k,
            "plan_scores_source": plan_info,
            "scan_plan": scan_plan,
        }

        sp_path = Path(args.scan_plan_out)
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        sp_path.write_text(json.dumps(scan_plan_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved to:", args.out)
    print(json.dumps({"used": used, "skipped": skipped, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

