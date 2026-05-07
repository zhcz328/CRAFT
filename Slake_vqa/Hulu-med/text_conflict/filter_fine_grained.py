# -*- coding: utf-8 -*-
import argparse
import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Any, List
from typing import Any as TypingAny

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM


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


def ensure_video_import_compat() -> None:
    """
    Hulu-Med remote processor may import video-related symbols even for image-only usage.
    Patch missing optional deps/symbols so image tasks can run without video stack.
    """
    # Optional video deps that may be imported by remote processor files.
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # transformers<some versions may not expose these typing aliases.
    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = TypingAny
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = TypingAny


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


def norm_text(x: str) -> str:
    return str(x).strip().lower()


def make_prompt_nc(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def make_prompt_cc(question: str, gold: str) -> str:
    return BASE_RULE + EVIDENCE_TMPL.format(ans=gold) + f"\nQuestion: {question}\nAnswer:"


def build_messages_multimodal(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    # 鍏煎浣犲綋鍓?transformers锛氭瘡鏉?message 鐨?content 閮界敤 block list
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]


def has_non_video_image_inputs(batch: Dict[str, Any]) -> bool:
    """
    Detect whether processor outputs include image-related tensors/fields.
    Prevent silently falling back to text-only execution.
    """
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


def extract_first_image_from_messages(messages: List[Dict[str, Any]]) -> Image.Image:
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "image" and ("image" in blk):
                return blk["image"]
    raise RuntimeError("Cannot find image block in messages.")


def move_to_device(obj, device, float_dtype=None):
    if torch.is_tensor(obj):
        if float_dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=float_dtype)
        return obj.to(device)
    # Handle BatchFeature / dict-like objects recursively first.
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
    """
    Prefer vision encoder dtype for image tensors.
    """
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


def resolve_row_image_path(row, image_root: str) -> Path | None:
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        raw = str(row["image_path"]).strip()
        if raw:
            p = Path(raw)
            candidates.extend([p, Path(image_root) / p])

    if "img_name" in row and pd.notna(row["img_name"]):
        raw = str(row["img_name"]).strip()
        if raw:
            p = Path(raw)
            candidates.extend([p, Path(image_root) / p])

    if "img_id" in row and pd.notna(row["img_id"]):
        raw = str(row["img_id"]).strip()
        if raw:
            candidates.append(Path(image_root) / raw)

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


# ---------- KV cache scoring (correct shift) ----------
@torch.inference_mode()
def build_prompt_cache_mm(model, processor, messages):
    # Path A: tokenizer+processor chat template directly.
    base_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Path B fallback (for some VLM processors): build text template first, then pass images explicitly.
    if not has_non_video_image_inputs(base_inputs):
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = extract_first_image_from_messages(messages)
        call_attempts = [
            {"text": text, "images": image},
            {"text": text, "images": [image]},
            {"text": [text], "images": image},
            {"text": [text], "images": [image]},
        ]
        last_err = None
        for kw in call_attempts:
            try:
                base_inputs = processor(
                    **kw,
                    padding=True,
                    return_tensors="pt",
                )
                if has_non_video_image_inputs(base_inputs):
                    break
            except Exception as e:
                last_err = e
                continue
        else:
            raise RuntimeError(
                "Failed to build multimodal processor inputs with explicit image injection. "
                f"Last error: {repr(last_err)}"
            )

    if not has_non_video_image_inputs(base_inputs):
        keys = sorted(list(base_inputs.keys()))
        raise RuntimeError(
            "No image-related inputs found in processor outputs. "
            "Model may be running text-only by mistake. "
            f"Current keys: {keys}"
        )
    base_inputs.pop("token_type_ids", None)
    target_dtype = infer_vision_input_dtype(model)
    base_inputs = move_to_device(base_inputs, model.device, float_dtype=target_dtype)

    out = model(**base_inputs, use_cache=True)

    input_ids_prompt = base_inputs["input_ids"]
    attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
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

    # first token from prompt_last_logits
    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)
    total = float(log_probs0[0, int(cont_ids[0, 0])])

    if cont_len == 1:
        return total

    # remaining tokens: feed cont_ids[:-1] and score cont_ids[1:]
    inp = cont_ids[:, :-1]
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)

    out = model(
        input_ids=inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
    )
    log_probs = torch.log_softmax(out.logits, dim=-1)  # [1, cont_len-1, V]

    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total += float(log_probs[0, i - 1, tok_id])

    return total


@torch.inference_mode()
def score_two_options_cached(model, processor, messages, gold: str, wrong: str) -> Dict[str, float]:
    cache_pack = build_prompt_cache_mm(model, processor, messages)
    # For speed, keep one stable option string format (leading space).
    gold_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + gold)
    wrong_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + wrong)
    return {gold: gold_lp, wrong: wrong_lp}


def argmax2(scores: Dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def read_input_table(path: str) -> pd.DataFrame:
    in_path = Path(path)
    suffix = in_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(in_path, engine="openpyxl")
    return pd.read_csv(in_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="CSV or XLSX with a wrong column.")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--resize_max_side", type=int, default=672, help="澶у浘闀胯竟缂╁埌璇ュ€硷紱0 涓嶇缉")
    args = ap.parse_args()

    df = read_input_table(args.in_csv)

    # required columns
    required = ["img_id", "question", "wrong"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col}")

    # gold label source: prefer `gold`, fallback to `answer`
    if "gold" in df.columns:
        gold_col = "gold"
    elif "answer" in df.columns:
        gold_col = "answer"
    else:
        raise ValueError("Input CSV must contain either 'gold' or 'answer' column as the correct label.")

    # normalized labels
    df["_gold_norm"] = df[gold_col].astype(str).map(norm_text)
    df["_wrong_norm"] = df["wrong"].astype(str).map(norm_text)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    model.eval()
    ensure_video_import_compat()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    out_rows = []
    seen = used = miss = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rerun filter NC&CC using wrong column"):
        if args.limit > 0 and seen >= args.limit:
            break
        seen += 1

        img_id = str(row["img_id"])
        question = str(row["question"])
        gold = norm_text(row["_gold_norm"])
        wrong = norm_text(row["_wrong_norm"])

        # 闃叉 degenerate锛歸rong==gold 鏃舵棤娉曟瀯鎴愬鐓э紙浣犳墜宸ユ敼鍙兘浼氬彂鐢燂級
        if wrong == gold or wrong == "":
            # 杩欓噷鎴戦€夋嫨璺宠繃锛涘鏋滀綘鎯充繚鐣欎篃琛岋紙浣嗕簩閫変竴娌℃剰涔夛級
            continue

        img_path = resolve_row_image_path(row, args.image_root)
        if img_path is None:
            miss += 1
            continue

        image = Image.open(img_path).convert("RGB")
        if args.resize_max_side and args.resize_max_side > 0:
            image.thumbnail((args.resize_max_side, args.resize_max_side))

        # NC
        nc_prompt = make_prompt_nc(question)
        nc_messages = build_messages_multimodal(image, nc_prompt)
        nc_scores = score_two_options_cached(model, processor, nc_messages, gold, wrong)
        nc_pred = argmax2(nc_scores)

        # CC锛坈orrect context锛夛細evidence 鍐?gold
        cc_prompt = make_prompt_cc(question, gold)
        cc_messages = build_messages_multimodal(image, cc_prompt)
        cc_scores = score_two_options_cached(model, processor, cc_messages, gold, wrong)
        cc_pred = argmax2(cc_scores)

        used += 1
        if (nc_pred == gold) and (cc_pred == gold):
            # Keep original row fields and append predictions/scores.
            out = dict(row)
            out.pop("_gold_norm", None)
            out.pop("_wrong_norm", None)
            out.update({
                "gold_norm": gold,
                "wrong_norm": wrong,
                "nc_pred": nc_pred,
                "cc_pred": cc_pred,
                "nc_lp_gold": float(nc_scores[gold]),
                "nc_lp_wrong": float(nc_scores[wrong]),
                "cc_lp_gold": float(cc_scores[gold]),
                "cc_lp_wrong": float(cc_scores[wrong]),
                "image_path": str(img_path),
            })
            out_rows.append(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print("\nDone.")
    print(f"Seen rows: {seen}")
    print(f"Used rows (images found & wrong!=gold): {used}")
    print(f"Missing images: {miss}")
    print(f"NC&CC both-correct kept: {len(out_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
