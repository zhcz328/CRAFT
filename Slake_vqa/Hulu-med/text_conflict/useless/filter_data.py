# -*- coding: utf-8 -*-
import argparse
import importlib.util
import sys
import types
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Any
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
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = TypingAny
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = TypingAny


def has_non_video_image_inputs(batch: Dict[str, Any]) -> bool:
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


# ---------- build by-category answer pool (for wrong answer generation) ----------
def build_by_cat_sorted(df: pd.DataFrame) -> Dict[str, List[str]]:
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = "other"
    df["ans"] = df["answer"].astype(str).map(norm_text)
    df["cat"] = df["category"].astype(str).map(norm_text)

    by_cat = defaultdict(list)
    for a, c in zip(df["ans"].tolist(), df["cat"].tolist()):
        by_cat[c].append(a)

    by_cat_sorted = {}
    for c, lst in by_cat.items():
        cnt = Counter(lst)
        by_cat_sorted[c] = [x for x, _ in cnt.most_common()]
    return by_cat_sorted


def choose_wrong_answer_priority_A_strict(gold: str, cat: str, by_cat_sorted: Dict[str, List[str]]) -> str:
    """
    鐢熸垚 wrong/conflict 绛旀锛堢敤浜庝簩閫変竴璇勫垎鐨勫彟涓€涓€欓€夛級锛?      - yes/no -> flip
      - 鍚﹀垯锛氬彧鍦ㄥ悓 category 鎵句竴涓渶甯歌涓斾笉鍚岀殑锛涙壘涓嶅埌灏?unknown
    """
    g = norm_text(gold)
    c = norm_text(cat)

    if g == "yes":
        return "no"
    if g == "no":
        return "yes"

    for cand in by_cat_sorted.get(c, []):
        cand_n = norm_text(cand)
        if cand_n and cand_n != g and cand_n != "unknown":
            return cand_n

    return "unknown"


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


# ---------- prompts ----------
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


# ---------- KV cache scoring (correct shift) ----------
@torch.inference_mode()
def build_prompt_cache_mm(model, processor, messages):
    # Path A: direct tokenized multimodal template.
    base_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Path B fallback for processors that only build image tensors in __call__.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--resize_max_side", type=int, default=672, help="澶у浘闀胯竟缂╁埌璇ュ€硷紱0 涓嶇缉")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    if "category" not in df.columns:
        df["category"] = "other"

    by_cat_sorted = build_by_cat_sorted(df)

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

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filter NC & CC both-correct"):
        if args.limit > 0 and seen >= args.limit:
            break
        seen += 1

        img_id = str(row["img_id"])
        question = str(row["question"])
        gold = norm_text(row["answer"])
        cat = norm_text(row["category"])

        wrong = choose_wrong_answer_priority_A_strict(gold, cat, by_cat_sorted)

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

        # CC (correct context)
        cc_prompt = make_prompt_cc(question, gold)
        cc_messages = build_messages_multimodal(image, cc_prompt)
        cc_scores = score_two_options_cached(model, processor, cc_messages, gold, wrong)
        cc_pred = argmax2(cc_scores)

        used += 1
        if (nc_pred == gold) and (cc_pred == gold):
            out_rows.append({
                "id": int(idx),
                "img_id": img_id,
                "image_path": str(img_path),
                "img_name": str(row["img_name"]) if "img_name" in row and pd.notna(row["img_name"]) else "",
                "category": cat,
                "content_type": str(row["content_type"]) if "content_type" in row and pd.notna(row["content_type"]) else "",
                "answer_type": str(row["answer_type"]) if "answer_type" in row and pd.notna(row["answer_type"]) else "",
                "base_type": str(row["base_type"]) if "base_type" in row and pd.notna(row["base_type"]) else "",
                "question": question,
                "gold": gold,
                "wrong": wrong,
                "nc_pred": nc_pred,
                "cc_pred": cc_pred,
                "nc_lp_gold": float(nc_scores[gold]),
                "nc_lp_wrong": float(nc_scores[wrong]),
                "cc_lp_gold": float(cc_scores[gold]),
                "cc_lp_wrong": float(cc_scores[wrong]),
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print("\nDone.")
    print(f"Seen rows: {seen}")
    print(f"Used rows (images found): {used}")
    print(f"Missing images: {miss}")
    print(f"NC&CC both-correct kept: {len(out_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
