# -*- coding: utf-8 -*-
import argparse
import importlib.util
import json
import re
import sys
import types
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from mask_utils import UNKNOWN_ANSWER, normalize_answer
from model_utils import (
    build_chat_template_inputs,
    build_processor_inputs_mm,
    extract_first_image_from_messages,
    has_non_video_image_inputs,
    infer_vision_input_dtype,
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    move_to_device,
    prefix_model_relative_path,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir


SYSTEM_PROMPT = "You are a helpful visual question answering assistant."
BASE_RULE = (
    "Answer the question using the image and your general world knowledge.\n"
    "If the image does not contain enough information to answer the question, output ONLY unknown.\n"
    "Output ONLY the final answer.\n\n"
)

DEFAULT_WRONG_CSV_CANDIDATES = (
    "data/slake_closed_all_wrong_backup.csv",
    "../text_conflict/data/slake_closed_all.csv",
)

NUMERIC_WORD_TO_VALUE = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

MAX_OPEN_CANDIDATES = 24


def ensure_video_import_compat() -> None:
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def load_mm_model(model_name, device_map=None, torch_dtype=None, trust_remote_code=True, attn_implementation=None, quantization_config=None):
    return shared_load_mm_model(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
    )


def norm_text(x: Any) -> str:
    return normalize_answer(str(x or ""))


def is_blank_text(value: Any) -> bool:
    if value is None:
        return True
    if pd.isna(value):
        return True
    return not str(value).strip()


def make_prompt(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def build_messages_multimodal(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]


def get_primary_device(model) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def resolve_row_image_path(row, image_root: str, key: str = "image_path") -> Path | None:
    root = Path(image_root)
    raw = str(row.get(key, "")).strip()
    if not raw:
        return None

    path = Path(raw)
    candidates = [path, root / path]
    parts = list(path.parts)
    for idx in range(1, len(parts)):
        candidates.append(root / Path(*parts[idx:]))
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

    seen = set()
    for candidate in candidates:
        key_name = str(candidate)
        if key_name in seen:
            continue
        seen.add(key_name)
        if candidate.exists():
            return candidate
    return None


def resize_image_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


def read_input_table(path: str) -> pd.DataFrame:
    in_path = Path(path)
    if in_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(in_path, engine="openpyxl")
    return pd.read_csv(in_path)


def parse_json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return []


def resume_key_for_row(row, idx: int) -> str:
    return (
        f"{idx}\t{row.get('id', '')}\t{row.get('qid', '')}\t{row.get('img_id', '')}\t"
        f"{row.get('question', '')}\t{row.get('gold', row.get('answer', ''))}"
    )


def discover_wrong_csv(explicit_path: str, in_csv: str, dataset_name: str = "gqa") -> Path | None:
    candidates: list[Path] = []
    if str(explicit_path or "").strip():
        explicit = Path(explicit_path).expanduser().resolve(strict=False)
        if not explicit.exists():
            raise FileNotFoundError(f"--wrong_csv does not exist: {explicit}")
        candidates.append(explicit)

    in_path = Path(in_csv).expanduser()
    if str(in_csv or "").strip():
        candidates.append(in_path.parent / "slake_closed_all_wrong_backup.csv")

    project_root = Path(__file__).resolve().parent
    for candidate in DEFAULT_WRONG_CSV_CANDIDATES:
        candidates.append(project_root / candidate)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def build_wrong_lookup(ref_df: pd.DataFrame, key_columns: Sequence[str]) -> dict[tuple[str, ...], str]:
    lookup: dict[tuple[str, ...], str] = {}
    for _, row in ref_df.iterrows():
        wrong = str(row.get("wrong", "")).strip()
        if not wrong:
            continue
        key = tuple(norm_text(row.get(col, "")) for col in key_columns)
        if any(not part for part in key):
            continue
        if key in lookup:
            continue
        lookup[key] = wrong
    return lookup


def attach_wrong_answers(df: pd.DataFrame, wrong_csv: str, in_csv: str, dataset_name: str = "gqa") -> tuple[pd.DataFrame, str, int]:
    out_df = df.copy()
    if "wrong" not in out_df.columns:
        out_df["wrong"] = ""

    present_mask = out_df["wrong"].map(lambda value: not is_blank_text(value))
    if bool(present_mask.all()):
        return out_df, "", 0

    wrong_path = discover_wrong_csv(wrong_csv, in_csv, dataset_name=dataset_name)
    if wrong_path is None:
        raise FileNotFoundError(
            "Could not locate a wrong-answer backup CSV. "
            "Provide --wrong_csv or place a <dataset>_closed_with_wrong.csv next to --in_csv."
        )

    ref_df = read_input_table(str(wrong_path))
    if "wrong" not in ref_df.columns:
        raise ValueError(f"Wrong-answer CSV missing required column 'wrong': {wrong_path}")

    lookup_specs: list[tuple[str, ...]] = [
        ("id",),
        ("split", "qid"),
        ("qid",),
        ("img_id", "question"),
    ]
    lookups: list[tuple[tuple[str, ...], dict[tuple[str, ...], str]]] = []
    for key_columns in lookup_specs:
        if all(col in out_df.columns for col in key_columns) and all(col in ref_df.columns for col in key_columns):
            lookup = build_wrong_lookup(ref_df, key_columns)
            if lookup:
                lookups.append((key_columns, lookup))

    if not lookups:
        raise ValueError(
            f"Could not build any lookup key for wrong-answer CSV: {wrong_path}. "
            "Need one of: id, (split,qid), qid, or (img_id,question)."
        )

    filled_count = 0
    new_values: list[str] = []
    for _, row in out_df.iterrows():
        current = str(row.get("wrong", "")).strip()
        if current:
            new_values.append(current)
            continue

        found = ""
        for key_columns, lookup in lookups:
            key = tuple(norm_text(row.get(col, "")) for col in key_columns)
            if any(not part for part in key):
                continue
            found = lookup.get(key, "")
            if found:
                break
        if found:
            filled_count += 1
        new_values.append(found)

    out_df["wrong"] = new_values
    return out_df, str(wrong_path), filled_count


def dedupe_candidate_options(options: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for option in options:
        norm = norm_text(option)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def is_open_answer_type(row: Mapping[str, Any]) -> bool:
    return str(row.get("answer_type", "")).strip().upper() == "OPEN"


def normalize_numeric_answer_text(value: Any) -> str:
    text = norm_text(value)
    if not text:
        return ""

    digit_match = re.search(r"(?<!\d)(\d+)(?:\.0+)?(?!\d)", text)
    if digit_match:
        return str(int(digit_match.group(1)))

    tokens = re.findall(r"[a-z]+", text)
    for token in tokens:
        if token in NUMERIC_WORD_TO_VALUE:
            return NUMERIC_WORD_TO_VALUE[token]
    return ""


def is_numeric_answer(value: Any) -> bool:
    return bool(normalize_numeric_answer_text(value))


def build_open_answer_pools(df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    by_question_type: dict[str, Counter[str]] = {}
    by_content_type: dict[str, Counter[str]] = {}

    for _, row in df.iterrows():
        if not is_open_answer_type(row):
            continue
        gold = norm_text(row.get("gold", row.get("answer", "")))
        if not gold or is_numeric_answer(gold):
            continue

        question_type = norm_text(row.get("ic_question_type", ""))
        content_type = norm_text(row.get("content_type", ""))
        if question_type:
            by_question_type.setdefault(question_type, Counter())[gold] += 1
        if content_type:
            by_content_type.setdefault(content_type, Counter())[gold] += 1

    question_type_pools = {
        key: [option for option, _count in counter.most_common()]
        for key, counter in by_question_type.items()
    }
    content_type_pools = {
        key: [option for option, _count in counter.most_common()]
        for key, counter in by_content_type.items()
    }
    return question_type_pools, content_type_pools


def build_open_candidates_for_row(
    row: Mapping[str, Any],
    question_type_pools: Mapping[str, Sequence[str]],
    content_type_pools: Mapping[str, Sequence[str]],
) -> List[str]:
    gold = norm_text(row.get("gold", row.get("answer", "")))
    question_type = norm_text(row.get("ic_question_type", ""))
    content_type = norm_text(row.get("content_type", ""))

    options: list[str] = []
    if question_type:
        options.extend(question_type_pools.get(question_type, []))
    if content_type:
        options.extend(content_type_pools.get(content_type, []))
    options.extend([gold, UNKNOWN_ANSWER])
    deduped = dedupe_candidate_options(options)
    required = dedupe_candidate_options([gold, UNKNOWN_ANSWER])
    required_set = set(required)
    prioritized = required + [option for option in deduped if option not in required_set]
    return prioritized[:MAX_OPEN_CANDIDATES]


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



def is_internvl_model(model) -> bool:
    candidates = [model]
    if hasattr(model, "module"):
        candidates.append(model.module)
    if hasattr(model, "base_model"):
        candidates.append(model.base_model)
    for cand in candidates:
        cfg = getattr(cand, "config", None)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        if "internvl" in model_type:
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
        attention_mask=attention_mask
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
            out = processor.apply_chat_template(attempt["messages"], **attempt["kwargs"])
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
            out = processor(**kw, padding=True, return_tensors="pt")
            if has_non_video_image_inputs(out):
                return out
        except Exception as exc:
            last_err = exc
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
            out = processor(**kw, padding=True, return_tensors="pt")
            if has_non_video_image_inputs(out):
                return out
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        "Failed to build Qwen3.5 multimodal full inputs. "
        f"Last error: {repr(last_err)}"
    )


@torch.inference_mode()
def build_prompt_cache_mm(model, processor, messages):
    if is_qwen35_model(model):
        base_inputs = build_qwen35_prompt_inputs(processor, messages)
    else:
        base_inputs = build_chat_template_inputs(processor, messages)

    base_inputs.pop("token_type_ids", None)
    base_inputs.pop("use_cache", None)
    device = get_primary_device(model)
    target_dtype = infer_vision_input_dtype(model)
    base_inputs = move_to_device(base_inputs, device, float_dtype=target_dtype)

    out = model(**base_inputs, use_cache=True)
    input_ids_prompt = base_inputs["input_ids"]
    attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()

    return {
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
        "rope_deltas": getattr(out, "rope_deltas", None),
        "qwen35_mode": is_qwen35_model(model),
    }


@torch.inference_mode()
def logprob_qwen35_full_forward(model, processor, messages, continuation: str) -> float:
    prompt_inputs = build_qwen35_prompt_inputs(processor, messages)
    full_inputs = build_qwen35_full_inputs(processor, messages, continuation)

    prompt_inputs.pop("token_type_ids", None)
    full_inputs.pop("token_type_ids", None)

    device = get_primary_device(model)
    target_dtype = infer_vision_input_dtype(model)
    full_inputs = move_to_device(full_inputs, device, float_dtype=target_dtype)

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
    device = get_primary_device(model)
    tok = processor.tokenizer
    cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
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
def score_candidate_options_cached(
    model,
    processor,
    messages,
    options: Sequence[str],
    internvl_full_forward: bool = False,
) -> Dict[str, float]:
    normalized = dedupe_candidate_options(options)
    if not normalized:
        return {}

    if internvl_full_forward and is_internvl_model(model):
        scores: Dict[str, float] = {}
        prompt_inputs = build_chat_template_inputs(processor, messages)
        prompt_inputs.pop("token_type_ids", None)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = extract_first_image_from_messages(messages)

        for option in normalized:
            continuation = " " + option
            full_inputs = build_processor_inputs_mm(processor, prompt_text + continuation, image)
            full_inputs.pop("token_type_ids", None)
            full_inputs.pop("use_cache", None)

            device = get_primary_device(model)
            target_dtype = infer_vision_input_dtype(model)
            full_inputs = move_to_device(full_inputs, device, float_dtype=target_dtype)

            full_ids = full_inputs["input_ids"]
            cont_len = int(full_ids.shape[1]) - prompt_len
            if cont_len <= 0:
                scores[option] = float("-inf")
                continue

            out = model(**full_inputs, use_cache=False)
            log_probs = torch.log_softmax(out.logits, dim=-1)

            total = 0.0
            for i in range(cont_len):
                tok_pos = prompt_len + i
                logit_pos = tok_pos - 1
                tok_id = int(full_ids[0, tok_pos])
                total += float(log_probs[0, logit_pos, tok_id])
            scores[option] = total
        return scores

    if is_qwen35_model(model):
        return {option: logprob_qwen35_full_forward(model, processor, messages, " " + option) for option in normalized}

    cache_pack = build_prompt_cache_mm(model, processor, messages)
    return {option: logprob_from_cache_mm(model, processor, cache_pack, " " + option) for option in normalized}


def argmax_scores(scores: Dict[str, float]) -> str:
    return max(scores, key=lambda key: scores[key])


@torch.inference_mode()
def generate_answer_greedy(model, processor, messages, max_new_tokens: int) -> str:
    if is_qwen35_model(model):
        base_inputs = build_qwen35_prompt_inputs(processor, messages)
    else:
        base_inputs = build_chat_template_inputs(processor, messages)

    base_inputs.pop("token_type_ids", None)
    base_inputs.pop("use_cache", None)
    if is_internvl_model(model):
        # InternVL may route generate() into language_model.generate(),
        # which rejects vision-only helper flags.
        base_inputs.pop("image_flags", None)
        base_inputs.pop("pixel_values_videos", None)
    device = get_primary_device(model)
    target_dtype = infer_vision_input_dtype(model)
    base_inputs = move_to_device(base_inputs, device, float_dtype=target_dtype)

    prompt_len = int(base_inputs["input_ids"].shape[1])
    output_ids = model.generate(
        **base_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )
    new_tokens = output_ids[:, prompt_len:]
    tokenizer = processor.tokenizer
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()


def maybe_attach_wrong_answers(df: pd.DataFrame, wrong_csv: str, in_csv: str, dataset_name: str = "gqa") -> tuple[pd.DataFrame, str, int, int]:
    out_df, used_wrong_csv, filled_count = attach_wrong_answers(df, wrong_csv, in_csv, dataset_name=dataset_name)
    missing_wrong = int(sum(is_blank_text(value) for value in out_df["wrong"]))
    return out_df, used_wrong_csv, filled_count, missing_wrong


def main():
    ap = argparse.ArgumentParser(description="Keep only NC-correct rows that can be used for image-conflict masking.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--wrong_csv", default="")
    ap.add_argument("--dataset_name", default="gqa")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--resize_max_side", type=int, default=672)
    ap.add_argument("--max_new_tokens", type=int, default=12, help="Reserved for backward compatibility.")
    ap.add_argument(
        "--internvl_closed_full_forward",
        action="store_true",
        help="Use full-forward candidate scoring (no KV-cache shortcut) for InternVL closed candidates.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_csv = prefix_model_relative_path(args.out_csv, model_name=args.model_name, model=args.model)

    df = read_input_table(args.in_csv)
    if "question" not in df.columns:
        raise ValueError("Input CSV missing required column: question")
    if "gold" not in df.columns and "answer" not in df.columns:
        raise ValueError("Input CSV must contain either 'gold' or 'answer'.")

    df, used_wrong_csv, filled_wrong_count, missing_wrong = maybe_attach_wrong_answers(
        df,
        args.wrong_csv,
        args.in_csv,
        dataset_name=args.dataset_name,
    )
    if "wrong" not in df.columns:
        raise ValueError("Failed to attach wrong-answer column.")
    open_question_type_pools, open_content_type_pools = build_open_answer_pools(df)

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
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(project_root=project_root, task_name="re_filter", model_name=args.model_name, model=args.model),
        enabled=args.resume,
    )
    tracker.start(
        task="image_conflict_re_filter",
        in_csv=args.in_csv,
        wrong_csv=used_wrong_csv,
        out_csv=args.out_csv,
        model=args.model,
        model_name=args.model_name,
    )

    out_rows = tracker.read_records("kept_rows.jsonl")
    seen = int(tracker.state.get("seen", 0))
    used = int(tracker.state.get("used", 0))
    missing = int(tracker.state.get("missing", 0))
    unresolved = int(tracker.state.get("unresolved", 0))
    mode_counts = dict(tracker.state.get("mode_counts", {}))

    prompt_cache: Dict[str, str] = {}
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filter NC-correct rows"):
        if args.limit > 0 and seen >= args.limit:
            break
        row_key = resume_key_for_row(row, idx)
        if tracker.is_done(row_key):
            continue
        seen += 1

        gold = norm_text(row.get("gold", row.get("answer", "")))
        wrong = norm_text(row.get("wrong", ""))
        target_labels = parse_json_list(row.get("ic_target_labels"))
        if not target_labels:
            unresolved += 1
            tracker.mark_done(row_key, {"status": "skip_unresolved_target"})
            tracker.update(seen=seen, used=used, missing=missing, unresolved=unresolved, kept=len(out_rows))
            continue

        img_path = resolve_row_image_path(row, args.image_root, key="image_path")
        det_path = resolve_row_image_path(row, args.image_root, key="detection_path")
        mask_path = resolve_row_image_path(row, args.image_root, key="mask_path")
        if img_path is None or det_path is None:
            missing += 1
            tracker.mark_done(row_key, {"status": "missing_assets"})
            tracker.update(seen=seen, used=used, missing=missing, unresolved=unresolved, kept=len(out_rows))
            continue

        question = str(row.get("question", "")).strip()
        prompt = prompt_cache.setdefault(question, make_prompt(question))
        image = resize_image_if_needed(Image.open(img_path).convert("RGB"), args.resize_max_side)
        messages = build_messages_multimodal(image, prompt)
        nc_eval_mode = "closed_candidates"
        nc_pred = ""
        nc_pred_raw = ""
        nc_scores: Dict[str, float] = {}
        nc_candidates: List[str] = []
        nc_lp_gold: float | None = None
        nc_lp_wrong: float | None = None
        nc_lp_unknown: float | None = None

        if is_open_answer_type(row):
            if is_numeric_answer(gold):
                nc_eval_mode = "open_numeric_generate"
                nc_pred_raw = generate_answer_greedy(
                    model=model,
                    processor=processor,
                    messages=messages,
                    max_new_tokens=args.max_new_tokens,
                )
                nc_pred = normalize_numeric_answer_text(nc_pred_raw)
                nc_correct = bool(nc_pred and nc_pred == normalize_numeric_answer_text(gold))
            else:
                nc_eval_mode = "open_candidate_scores"
                nc_candidates = build_open_candidates_for_row(
                    row=row,
                    question_type_pools=open_question_type_pools,
                    content_type_pools=open_content_type_pools,
                )
                if len(nc_candidates) < 2:
                    tracker.mark_done(row_key, {"status": "skip_invalid_candidates", "mode": nc_eval_mode})
                    tracker.update(
                        seen=seen,
                        used=used,
                        missing=missing,
                        unresolved=unresolved,
                        kept=len(out_rows),
                        mode_counts=mode_counts,
                    )
                    continue
                nc_scores = score_candidate_options_cached(
                    model=model,
                    processor=processor,
                    messages=messages,
                    options=nc_candidates,
                )
                nc_pred = argmax_scores(nc_scores)
                nc_lp_gold = float(nc_scores.get(gold, float("nan")))
                nc_lp_unknown = float(nc_scores.get(UNKNOWN_ANSWER, float("nan")))
                nc_correct = nc_pred == gold
        else:
            if is_blank_text(wrong):
                tracker.mark_done(row_key, {"status": "skip_missing_wrong", "mode": nc_eval_mode})
                tracker.update(
                    seen=seen,
                    used=used,
                    missing=missing,
                    unresolved=unresolved,
                    kept=len(out_rows),
                    mode_counts=mode_counts,
                )
                continue
            nc_candidates = dedupe_candidate_options([gold, wrong, UNKNOWN_ANSWER])
            if len(nc_candidates) < 2:
                tracker.mark_done(row_key, {"status": "skip_invalid_candidates", "mode": nc_eval_mode})
                tracker.update(
                    seen=seen,
                    used=used,
                    missing=missing,
                    unresolved=unresolved,
                    kept=len(out_rows),
                    mode_counts=mode_counts,
                )
                continue
            nc_scores = score_candidate_options_cached(
                model=model,
                processor=processor,
                messages=messages,
                options=nc_candidates,
                internvl_full_forward=args.internvl_closed_full_forward,
            )
            nc_pred = argmax_scores(nc_scores)
            nc_lp_gold = float(nc_scores.get(gold, float("nan")))
            nc_lp_wrong = float(nc_scores[wrong]) if wrong in nc_scores else None
            nc_lp_unknown = float(nc_scores.get(UNKNOWN_ANSWER, float("nan")))
            nc_correct = nc_pred == gold

        used += 1
        mode_counts[nc_eval_mode] = int(mode_counts.get(nc_eval_mode, 0)) + 1

        if nc_correct:
            out = dict(row)
            out.update(
                {
                    "gold_norm": gold,
                    "wrong_norm": wrong,
                    "unknown_target": UNKNOWN_ANSWER,
                    "nc_eval_mode": nc_eval_mode,
                    "nc_pred_raw": nc_pred_raw or nc_pred,
                    "nc_pred": nc_pred,
                    "nc_correct": True,
                    "nc_candidates": json.dumps(nc_candidates, ensure_ascii=False) if nc_candidates else "",
                    "nc_scores_json": json.dumps(nc_scores, ensure_ascii=False) if nc_scores else "",
                    "nc_lp_gold": nc_lp_gold,
                    "nc_lp_wrong": nc_lp_wrong,
                    "nc_lp_unknown": nc_lp_unknown,
                    "image_path": str(img_path),
                    "detection_path": str(det_path),
                    "mask_path": str(mask_path) if mask_path is not None else "",
                    "ic_target_labels": json.dumps(target_labels, ensure_ascii=False),
                    "ic_target_label_count": len(target_labels),
                }
            )
            out_rows.append(out)
            tracker.append_record("kept_rows.jsonl", out)
            tracker.mark_done(
                row_key,
                {"status": "kept", "mode": nc_eval_mode, "nc_pred": nc_pred, "nc_pred_raw": nc_pred_raw or nc_pred},
            )
        else:
            tracker.mark_done(
                row_key,
                {
                    "status": "filtered_out",
                    "mode": nc_eval_mode,
                    "nc_pred": nc_pred,
                    "nc_pred_raw": nc_pred_raw or nc_pred,
                    "nc_scores": nc_scores if nc_scores else None,
                },
            )
        tracker.update(
            seen=seen,
            used=used,
            missing=missing,
            unresolved=unresolved,
            kept=len(out_rows),
            mode_counts=mode_counts,
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    tracker.finish(
        seen=seen,
        used=used,
        missing=missing,
        unresolved=unresolved,
        kept=len(out_rows),
        mode_counts=mode_counts,
        wrong_csv=used_wrong_csv,
        wrong_filled=filled_wrong_count,
        wrong_missing=missing_wrong,
        out_csv=str(out_path),
    )

    print("Done.")
    print(f"Seen rows: {seen}")
    print(f"Used rows: {used}")
    print(f"Missing assets: {missing}")
    print(f"Unresolved IC targets: {unresolved}")
    print(f"Rows filled from wrong CSV: {filled_wrong_count}")
    print(f"Rows still missing wrong answer: {missing_wrong}")
    if used_wrong_csv:
        print(f"Wrong CSV used: {used_wrong_csv}")
    if mode_counts:
        print(f"NC eval modes: {json.dumps(mode_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"NC-correct kept: {len(out_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()






