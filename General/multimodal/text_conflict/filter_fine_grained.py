# -*- coding: utf-8 -*-
import argparse
import importlib.util
import re
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

from model_utils import (
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir


SYSTEM_PROMPT = "You are a helpful visual question answering assistant."

BASE_RULE = (
    "Answer the question using the image and your general world knowledge.\n"
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
    return shared_load_mm_model(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )


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


def is_qwen35_model(model) -> bool:
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        cfg = getattr(cand, "config", None)
        if getattr(cfg, "model_type", "") == "qwen3_5":
            return True
    return False


def get_qwen35_core_model(model):
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        inner = getattr(cand, "model", None)
        if getattr(getattr(cand, "config", None), "model_type", "") == "qwen3_5":
            return inner if inner is not None else cand
    return None


def set_qwen35_rope_deltas(model, rope_deltas) -> None:
    if rope_deltas is None:
        return
    core = get_qwen35_core_model(model)
    if core is not None and hasattr(core, "rope_deltas"):
        core.rope_deltas = rope_deltas


def get_qwen35_generation_model(model):
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        if getattr(getattr(cand, "config", None), "model_type", "") == "qwen3_5":
            if hasattr(cand, "prepare_inputs_for_generation"):
                return cand
    return None


def build_qwen35_continuation_kwargs(model, rope_deltas, past_key_values, attention_mask, continuation_ids):
    set_qwen35_rope_deltas(model, rope_deltas)

    generation_model = get_qwen35_generation_model(model)
    if generation_model is None:
        raise RuntimeError("Cannot find a Qwen3.5 generation model with prepare_inputs_for_generation().")

    prepared = generation_model.prepare_inputs_for_generation(
        continuation_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )
    prepared = dict(prepared)
    prepared.pop("token_type_ids", None)
    prepared["use_cache"] = True
    prepared["past_key_values"] = past_key_values
    prepared["attention_mask"] = attention_mask
    prepared.setdefault("pixel_values", None)
    prepared.setdefault("pixel_values_videos", None)
    return prepared


def build_qwen35_prompt_inputs(processor, messages):
    last_err = None
    official_attempts = [
        {
            "messages": messages,
            "kwargs": {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
        },
        {
            "messages": messages,
            "kwargs": {
                "tokenize": True,
                "add_generation_prompt": False,
                "return_dict": True,
                "return_tensors": "pt",
            },
        },
    ]
    for attempt in official_attempts:
        try:
            out = processor.apply_chat_template(
                attempt["messages"],
                **attempt["kwargs"],
            )
            if has_non_video_image_inputs(out):
                return out
        except Exception as exc:
            last_err = exc
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = extract_first_image_from_messages(messages)
    call_attempts = [
        {"text": text, "images": image},
        {"text": [text], "images": image},
        {"text": text, "images": [image]},
        {"text": [text], "images": [image]},
    ]
    for kw in call_attempts:
        try:
            out = processor(
                **kw,
                padding=True,
                return_tensors="pt",
            )
            if has_non_video_image_inputs(out):
                return out
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        "Failed to build Qwen3.5 multimodal inputs. "
        f"Last error: {repr(last_err)}"
    )


def build_qwen35_full_inputs(processor, messages, continuation: str):
    last_err = None
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + continuation
    image = extract_first_image_from_messages(messages)
    call_attempts = [
        {"text": full_text, "images": image},
        {"text": [full_text], "images": image},
        {"text": full_text, "images": [image]},
        {"text": [full_text], "images": [image]},
    ]
    for kw in call_attempts:
        try:
            out = processor(
                **kw,
                padding=True,
                return_tensors="pt",
            )
            if has_non_video_image_inputs(out):
                return out
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        "Failed to build Qwen3.5 multimodal full inputs. "
        f"Last error: {repr(last_err)}"
    )


def build_path_candidates(raw: str, image_root: str) -> List[Path]:
    raw = str(raw).strip()
    if not raw:
        return []

    root = Path(image_root)
    p = Path(raw)
    candidates: List[Path] = [p, root / p]

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


def resolve_row_image_path(row, image_root: str) -> Path | None:
    candidates = []

    if "image_path" in row and pd.notna(row["image_path"]):
        candidates.extend(build_path_candidates(row["image_path"], image_root))

    if "img_name" in row and pd.notna(row["img_name"]):
        candidates.extend(build_path_candidates(row["img_name"], image_root))

    if "img_id" in row and pd.notna(row["img_id"]):
        candidates.extend(build_path_candidates(row["img_id"], image_root))

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


def resume_key_for_row(row, idx: int) -> str:
    img_id = str(row.get("img_id", ""))
    question = str(row.get("question", ""))
    gold = str(row.get("_gold_norm", row.get("gold", row.get("answer", ""))))
    wrong = str(row.get("_wrong_norm", row.get("wrong", "")))
    return f"{idx}\t{img_id}\t{question}\t{gold}\t{wrong}"


# ---------- KV cache scoring (correct shift) ----------
@torch.inference_mode()
def build_prompt_cache_mm(model, processor, messages):
    qwen35_mode = is_qwen35_model(model)
    if qwen35_mode:
        base_inputs = build_qwen35_prompt_inputs(processor, messages)
    else:
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
        "prompt_len": int(input_ids_prompt.shape[1]),
        "qwen35_mode": qwen35_mode,
        "rope_deltas": getattr(out, "rope_deltas", None),
        "image_grid_thw": base_inputs.get("image_grid_thw"),
        "video_grid_thw": base_inputs.get("video_grid_thw"),
        "second_per_grid_ts": base_inputs.get("second_per_grid_ts"),
        "mm_token_type_ids": base_inputs.get("mm_token_type_ids"),
    }


@torch.inference_mode()
def logprob_qwen35_full_forward(model, processor, messages, continuation: str) -> float:
    prompt_inputs = build_qwen35_prompt_inputs(processor, messages)
    full_inputs = build_qwen35_full_inputs(processor, messages, continuation)

    prompt_inputs.pop("token_type_ids", None)
    full_inputs.pop("token_type_ids", None)

    target_dtype = infer_vision_input_dtype(model)
    full_inputs = move_to_device(full_inputs, model.device, float_dtype=target_dtype)

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_ids = full_inputs["input_ids"]
    full_len = int(full_ids.shape[1])
    cont_len = full_len - prompt_len
    if cont_len <= 0:
        return float("-inf")

    out = model(**full_inputs, use_cache=False)
    log_probs = torch.log_softmax(out.logits, dim=-1)

    total = 0.0
    for i in range(cont_len):
        tok_pos = prompt_len + i
        logit_pos = tok_pos - 1
        tok_id = int(full_ids[0, tok_pos])
        total += float(log_probs[0, logit_pos, tok_id])
    return total


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

    if cache_pack.get("qwen35_mode"):
        running_past = past_key_values
        for i in range(1, cont_len):
            prev_tok = cont_ids[:, i - 1 : i]
            step_attn = torch.cat(
                [attn_prompt, torch.ones((1, i), dtype=attn_prompt.dtype, device=attn_prompt.device)],
                dim=1,
            )
            model_kwargs = build_qwen35_continuation_kwargs(
                model,
                cache_pack.get("rope_deltas"),
                running_past,
                step_attn,
                prev_tok,
            )
            out = model(**model_kwargs)
            running_past = out.past_key_values
            step_log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)
            tok_id = int(cont_ids[0, i])
            total += float(step_log_probs[0, tok_id])
        return total

    # remaining tokens: feed cont_ids[:-1] and score cont_ids[1:]
    inp = cont_ids[:, :-1]
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)
    model_kwargs = {
        "input_ids": inp,
        "attention_mask": attn_full,
        "past_key_values": past_key_values,
        "use_cache": False,
    }
    out = model(**model_kwargs)
    log_probs = torch.log_softmax(out.logits, dim=-1)  # [1, cont_len-1, V]

    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total += float(log_probs[0, i - 1, tok_id])

    return total


@torch.inference_mode()
def score_two_options_cached(model, processor, messages, gold: str, wrong: str) -> Dict[str, float]:
    if is_qwen35_model(model):
        gold_lp = logprob_qwen35_full_forward(model, processor, messages, " " + gold)
        wrong_lp = logprob_qwen35_full_forward(model, processor, messages, " " + wrong)
        return {gold: gold_lp, wrong: wrong_lp}

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
    ap.add_argument("--model_name", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--resize_max_side", type=int, default=672, help="澶у浘闀胯竟缂╁埌璇ュ€硷紱0 涓嶇缉")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)

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
    processor = load_processor_with_compat(args.model)

    project_root = Path(__file__).resolve().parent
    resume_dir = build_resume_dir(
        project_root=project_root,
        task_name="re_filter",
        model_name=args.model_name,
        model=args.model,
    )
    tracker = ResumeTracker(resume_dir=resume_dir, enabled=args.resume)
    tracker.start(
        task="re_filter",
        model=args.model,
        model_name=args.model_name,
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        total_rows=int(len(df)),
    )

    out_rows = tracker.read_records("kept_rows.jsonl")
    seen = int(tracker.state.get("seen", 0))
    used = int(tracker.state.get("used", 0))
    miss = int(tracker.state.get("miss", 0))

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rerun filter NC&CC using wrong column"):
        if args.limit > 0 and seen >= args.limit:
            break
        row_key = resume_key_for_row(row, idx)
        if tracker.is_done(row_key):
            continue
        seen += 1

        img_id = str(row["img_id"])
        question = str(row["question"])
        gold = norm_text(row["_gold_norm"])
        wrong = norm_text(row["_wrong_norm"])

        # 闃叉 degenerate锛歸rong==gold 鏃舵棤娉曟瀯鎴愬鐓э紙浣犳墜宸ユ敼鍙兘浼氬彂鐢燂級
        if wrong == gold or wrong == "":
            # 杩欓噷鎴戦€夋嫨璺宠繃锛涘鏋滀綘鎯充繚鐣欎篃琛岋紙浣嗕簩閫変竴娌℃剰涔夛級
            tracker.mark_done(row_key, {"status": "skip_degenerate"})
            tracker.update(seen=seen, used=used, miss=miss, kept=len(out_rows))
            continue

        img_path = resolve_row_image_path(row, args.image_root)
        if img_path is None:
            miss += 1
            tracker.mark_done(row_key, {"status": "missing_image"})
            tracker.update(seen=seen, used=used, miss=miss, kept=len(out_rows))
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
            tracker.append_record("kept_rows.jsonl", out)
            tracker.mark_done(row_key, {"status": "kept"})
        else:
            tracker.mark_done(row_key, {"status": "filtered_out"})
        tracker.update(seen=seen, used=used, miss=miss, kept=len(out_rows))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    tracker.finish(seen=seen, used=used, miss=miss, kept=len(out_rows), out_csv=str(out_path))

    print("\nDone.")
    print(f"Seen rows: {seen}")
    print(f"Used rows (images found & wrong!=gold): {used}")
    print(f"Missing images: {miss}")
    print(f"NC&CC both-correct kept: {len(out_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
