import csv
import importlib.util
import json
import sys
import types
from collections import Counter
from collections.abc import Mapping
from difflib import SequenceMatcher
from pathlib import Path


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

LABEL_MAP = {
    "resist": 0,
    "hijack": 1,
}

EVIDENCE_BEGIN_SENTINEL = "<<CHDP_EVIDENCE_BEGIN_9f3a1c>>"
EVIDENCE_END_SENTINEL = "<<CHDP_EVIDENCE_END_9f3a1c>>"
QUESTION_BEGIN_SENTINEL = "<<CHDP_QUESTION_BEGIN_9f3a1c>>"
QUESTION_END_SENTINEL = "<<CHDP_QUESTION_END_9f3a1c>>"
IMAGE_BEGIN_SENTINEL = "<<CHDP_IMAGE_BEGIN_9f3a1c>>"
IMAGE_END_SENTINEL = "<<CHDP_IMAGE_END_9f3a1c>>"


def norm_text(text):
    return str(text or "").strip().lower()


def sample_key(img_id, question):
    return f"{str(img_id).strip()}||{norm_text(question)}"


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_base_prompt(question):
    return BASE_RULE + f"Question: {question}\nAnswer:"


def inject_evidence(base_prompt, evidence_block, position):
    if position == "prefix":
        return evidence_block + "\n" + base_prompt
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)
    raise ValueError(f"Unsupported position: {position}")


def make_prompt_no_evidence(question):
    return make_base_prompt(question)


def make_prompt_with_evidence(question, evidence_answer, position):
    evidence_block = EVIDENCE_TMPL.format(ans=str(evidence_answer).strip())
    return inject_evidence(make_base_prompt(question), evidence_block, position)


def make_prompt_with_marked_evidence(question, evidence_answer, position):
    evidence_block = (
        f"{EVIDENCE_BEGIN_SENTINEL}"
        f"{EVIDENCE_TMPL.format(ans=str(evidence_answer).strip())}"
        f"{EVIDENCE_END_SENTINEL}"
    )
    return inject_evidence(make_base_prompt(question), evidence_block, position)


def make_prompt_with_marked_question(question, evidence_answer, position):
    prompt = (
        make_prompt_with_evidence(question, evidence_answer, position)
        if evidence_answer is not None
        else make_prompt_no_evidence(question)
    )
    target = f"Question: {question}"
    marked = f"Question: {QUESTION_BEGIN_SENTINEL}{question}{QUESTION_END_SENTINEL}"
    if target not in prompt:
        raise ValueError("Question anchor not found while building marked prompt.")
    return prompt.replace(target, marked, 1)


def build_messages_multimodal(image, prompt_text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def default_preds_path(project_root, position):
    candidates = {
        "prefix": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_root_autodl-tmp_hulumed-4B_Hulu-Med-4B"
        / "prefix"
        / "preds.jsonl",
        "before_question": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_archive_zengjiaqi_Medical_LLM_Hulu-Med-4B"
        / "before_question"
        / "preds.jsonl",
        "before_answer": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_root_autodl-tmp_hulumed-4B_Hulu-Med-4B"
        / "before_answer"
        / "preds.jsonl",
    }
    return candidates[position]


def load_outcome_map(preds_jsonl_path):
    rows = read_jsonl(preds_jsonl_path)
    outcome_map = {}
    for row in rows:
        key = sample_key(row.get("img_id", ""), row.get("question", ""))
        outcome_map[key] = {
            "img_id": row.get("img_id"),
            "question": row.get("question"),
            "gold": norm_text(row.get("gold")),
            "conflict": norm_text(row.get("conflict")),
            "nc_pred": norm_text(row.get("nc_pred")),
            "support_pred": norm_text(row.get("support_pred")),
            "conflict_pred": norm_text(row.get("conflict_pred")),
            "resist": bool(row.get("resist", False)),
            "follow_conflict": bool(row.get("follow_conflict", False)),
            "changed_vs_nc": bool(row.get("changed_vs_nc", False)),
            "position": row.get("position"),
        }
    return outcome_map


def derive_binary_label(outcome_record, gold_answer, conflict_answer):
    if not outcome_record:
        return None
    gold = norm_text(gold_answer)
    conflict = norm_text(conflict_answer)
    nc_pred = norm_text(outcome_record.get("nc_pred"))
    ic_pred = norm_text(outcome_record.get("conflict_pred"))
    if nc_pred != gold:
        return None
    if ic_pred == gold:
        return "resist"
    if ic_pred == conflict:
        return "hijack"
    return None


def ensure_video_import_compat():
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def resolve_dtype(name):
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def move_to_device(obj, device, float_dtype=None):
    import torch

    if torch.is_tensor(obj):
        if float_dtype is not None and torch.is_floating_point(obj):
            return obj.to(device=device, dtype=float_dtype)
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device, float_dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device, float_dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device, float_dtype) for v in obj)
    return obj


def load_mm_model(model_name, device_map=None, torch_dtype=None, attn_implementation=None):
    from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

    load_errors = []
    for cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            kwargs = {"trust_remote_code": True}
            if device_map is not None:
                kwargs["device_map"] = device_map
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            if attn_implementation is not None:
                kwargs["attn_implementation"] = attn_implementation
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            load_errors.append(f"{cls.__name__}: {repr(exc)}")
    raise RuntimeError("Failed to load multimodal model. " + " | ".join(load_errors))


def get_output_head(model):
    head = None
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None and hasattr(model, "lm_head"):
        head = model.lm_head
    if head is None:
        raise RuntimeError("Model does not expose output embeddings.")
    return head


def _get_attr_chain(obj, chain):
    cur = obj
    for name in chain.split("."):
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def get_final_norm(model):
    for chain in (
        "model.norm",
        "language_model.model.norm",
        "model.decoder.final_layernorm",
        "transformer.ln_f",
        "decoder.final_layernorm",
    ):
        norm_mod = _get_attr_chain(model, chain)
        if norm_mod is not None:
            return norm_mod
    return None


def resolve_image_path(project_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    return project_root / path


def resize_image_if_needed(image, image_size):
    if not image_size or image_size <= 0:
        return image
    image = image.copy()
    image.thumbnail((image_size, image_size))
    return image


def pad_and_stack_tensors(tensors, key, pad_token_id):
    import torch

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


def collate_processor_outputs(outputs, pad_token_id):
    import torch

    collated = {}
    for key in outputs[0]:
        values = [output[key] for output in outputs]
        if torch.is_tensor(values[0]):
            collated[key] = pad_and_stack_tensors(values, key, pad_token_id)
        else:
            collated[key] = values
    return collated


def build_processor_inputs_mm(processor, prompt, image):
    messages = build_messages_multimodal(image, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=text,
        images=image,
        return_tensors="pt",
    )


def build_text_only_ids(processor, prompt):
    text = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids, text


def build_mm_text_with_image_markers(processor, prompt, image):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMAGE_BEGIN_SENTINEL},
                {"type": "image", "image": image},
                {"type": "text", "text": IMAGE_END_SENTINEL + prompt},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def decode_with_offsets(tokenizer, token_ids):
    pieces = []
    offsets = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def locate_substring_token_span(tokenizer, token_ids, needle):
    decoded, offsets = decode_with_offsets(tokenizer, token_ids)
    char_start = decoded.find(needle)
    if char_start < 0:
        return None, {
            "needle": needle,
            "decoded_preview": decoded[:400],
        }
    char_end = char_start + len(needle)
    start_token = None
    end_token = None
    for idx, (start, end) in enumerate(offsets):
        if start_token is None and end > char_start:
            start_token = idx
        if start < char_end:
            end_token = idx + 1
        if start >= char_end:
            break
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = len(token_ids)
    return (start_token, end_token), {
        "needle": needle,
        "char_start": char_start,
        "char_end": char_end,
        "token_start": start_token,
        "token_end": end_token,
        "decoded_preview": decoded[:400],
    }


def marker_debug_payload(marked_input_ids, begin_marker, end_marker, tokenizer):
    begin_span, begin_meta = locate_substring_token_span(tokenizer, marked_input_ids, begin_marker)
    end_span, end_meta = locate_substring_token_span(tokenizer, marked_input_ids, end_marker)
    return {
        "begin_marker": begin_marker,
        "end_marker": end_marker,
        "begin_span": begin_span,
        "end_span": end_span,
        "begin_meta": begin_meta,
        "end_meta": end_meta,
        "method": "sentinel_marked_prompt",
    }


def locate_marked_span_from_inputs(actual_input_ids, marked_input_ids, tokenizer, begin_marker, end_marker):
    payload = marker_debug_payload(marked_input_ids, begin_marker, end_marker, tokenizer)
    begin_span = payload.get("begin_span")
    end_span = payload.get("end_span")
    begin_ids = tokenizer(begin_marker, add_special_tokens=False)["input_ids"]
    end_ids = tokenizer(end_marker, add_special_tokens=False)["input_ids"]
    if begin_span is None or end_span is None:
        return (0, 0), payload

    begin_start = int(begin_span[0])
    end_start = int(end_span[0])
    marked_span = (begin_start + len(begin_ids), end_start)
    actual_span = (
        max(0, marked_span[0] - len(begin_ids)),
        max(0, marked_span[1] - len(begin_ids)),
    )
    actual_span = (
        min(actual_span[0], len(actual_input_ids)),
        min(actual_span[1], len(actual_input_ids)),
    )
    payload["marked_span"] = [int(marked_span[0]), int(marked_span[1])]
    payload["actual_span"] = [int(actual_span[0]), int(actual_span[1])]
    payload["begin_marker_token_count"] = len(begin_ids)
    payload["end_marker_token_count"] = len(end_ids)
    return actual_span, payload


def collect_insert_like_spans(base_ids, target_ids):
    spans = []
    matcher = SequenceMatcher(a=base_ids, b=target_ids, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"} and j2 > j1:
            spans.append(
                {
                    "base_span": (i1, i2),
                    "target_span": (j1, j2),
                    "length": j2 - j1,
                }
            )
    return spans


def locate_inserted_target_span(base_ids, target_ids, expected_len=None):
    spans = collect_insert_like_spans(base_ids, target_ids)
    if not spans:
        return (0, 0)
    if expected_len is not None and expected_len > 0:
        best = min(
            spans,
            key=lambda item: (
                abs(item["length"] - expected_len),
                -item["length"],
                item["target_span"][0],
            ),
        )
        return best["target_span"]
    best = max(spans, key=lambda item: (item["length"], -item["target_span"][0]))
    return best["target_span"]


def get_merge_size(processor, model):
    values = []
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        for name in ("merge_size", "spatial_merge_size"):
            value = getattr(image_processor, name, None)
            if value is not None:
                values.append(int(value))
    config = getattr(model, "config", None)
    if config is not None:
        for name in ("vision_spatial_merge_size", "spatial_merge_size"):
            value = getattr(config, name, None)
            if value is not None:
                values.append(int(value))
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg is not None:
            for name in ("spatial_merge_size", "merge_size"):
                value = getattr(vision_cfg, name, None)
                if value is not None:
                    values.append(int(value))
    for value in values:
        if value and value > 0:
            return value
    return 1


def expected_image_token_count(inputs, processor, model):
    import torch

    grid = inputs.get("image_grid_thw")
    if grid is None:
        return None
    flat = grid[0].tolist() if torch.is_tensor(grid) else list(grid[0])
    if len(flat) != 3:
        return None
    t, h, w = [int(x) for x in flat]
    merge = get_merge_size(processor, model)
    if merge > 1 and h % merge == 0 and w % merge == 0:
        return int(t * (h // merge) * (w // merge))
    return int(t * h * w)


def refine_image_positions(input_ids, span, expected_count):
    positions = list(range(int(span[0]), int(span[1])))
    if not positions:
        return positions
    if expected_count is None or expected_count <= 0 or len(positions) == expected_count:
        return positions
    span_ids = [input_ids[idx] for idx in positions]
    token_id, count = Counter(span_ids).most_common(1)[0]
    token_positions = [positions[i] for i, value in enumerate(span_ids) if value == token_id]
    if count == expected_count:
        return token_positions
    if len(positions) > expected_count:
        return positions[:expected_count]
    return positions


def positions_to_span(positions):
    if not positions:
        return (0, 0)
    ordered = sorted(int(x) for x in positions)
    return (ordered[0], ordered[-1] + 1)


def collect_candidate_image_token_ids(processor, model):
    candidate_values = []
    tokenizer = getattr(processor, "tokenizer", None)
    for obj in (
        processor,
        tokenizer,
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
    ):
        if obj is None:
            continue
        for name in (
            "image_token_id",
            "vision_token_id",
            "image_pad_token_id",
            "img_token_id",
            "boi_token_id",
            "eoi_token_id",
        ):
            value = getattr(obj, name, None)
            if value is not None:
                candidate_values.append(value)
        for name in (
            "image_token",
            "image_pad_token",
            "boi_token",
            "eoi_token",
            "img_token",
        ):
            value = getattr(obj, name, None)
            if value:
                candidate_values.append(value)
    ids = set()
    if tokenizer is not None:
        for value in candidate_values:
            if isinstance(value, int):
                ids.add(int(value))
            elif isinstance(value, str):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(value)
                    if token_id is not None and int(token_id) >= 0:
                        ids.add(int(token_id))
                except Exception:
                    pass
    return sorted(ids)


def locate_image_positions_by_token_ids(input_ids, candidate_ids, expected_count):
    best = []
    best_score = None
    for token_id in candidate_ids:
        positions = [idx for idx, value in enumerate(input_ids) if int(value) == int(token_id)]
        if not positions:
            continue
        span_width = positions[-1] - positions[0] + 1
        density = len(positions) / max(span_width, 1)
        if expected_count is not None and expected_count > 0:
            score = (abs(len(positions) - expected_count), -density, -len(positions), positions[0])
        else:
            score = (-len(positions), -density, positions[0])
        if best_score is None or score < best_score:
            best = positions
            best_score = score
    return best


def locate_image_positions_dense_repeat(input_ids, expected_count):
    counts = Counter(int(x) for x in input_ids)
    best = []
    best_score = None
    for token_id, count in counts.items():
        if count < 8:
            continue
        positions = [idx for idx, value in enumerate(input_ids) if int(value) == token_id]
        span_width = positions[-1] - positions[0] + 1
        density = len(positions) / max(span_width, 1)
        if density < 0.5:
            continue
        if expected_count is not None and expected_count > 0:
            score = (abs(count - expected_count), -density, -count, positions[0])
        else:
            score = (-count, -density, positions[0])
        if best_score is None or score < best_score:
            best = positions
            best_score = score
    return best


def locate_image_positions(input_ids, text_only_ids, expected_count, processor, model):
    candidate_ids = collect_candidate_image_token_ids(processor, model)
    positions = locate_image_positions_by_token_ids(input_ids, candidate_ids, expected_count)
    if positions:
        return positions, "special_token_id", {"candidate_ids": candidate_ids}

    positions = locate_image_positions_dense_repeat(input_ids, expected_count)
    if positions:
        token_id = int(input_ids[positions[0]])
        return positions, "dense_repeat_token", {"token_id": token_id, "count": len(positions)}

    span = locate_inserted_target_span(text_only_ids, input_ids, expected_len=expected_count)
    positions = refine_image_positions(input_ids, span, expected_count)
    return positions, "sequence_alignment_fallback", {"raw_span": [int(span[0]), int(span[1])]}


def build_condition_pack(processor, prompt, image, model, question, position, evidence_answer=None):
    inputs = build_processor_inputs_mm(processor, prompt, image)
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    expected_count = expected_image_token_count(inputs, processor, model)

    image_positions = []
    image_span = (0, 0)
    image_method = "unresolved"
    image_debug = {"method": "none"}
    try:
        marked_image_text = build_mm_text_with_image_markers(processor, prompt, image)
        marked_image_inputs = processor(text=marked_image_text, images=image, return_tensors="pt")
        marked_image_ids = marked_image_inputs["input_ids"][0].detach().cpu().tolist()
        image_span, image_debug = locate_marked_span_from_inputs(
            input_ids,
            marked_image_ids,
            processor.tokenizer,
            IMAGE_BEGIN_SENTINEL,
            IMAGE_END_SENTINEL,
        )
        image_method = image_debug.get("method", "sentinel_marked_prompt")
        image_debug["marked_text_preview"] = marked_image_text[:400]
    except Exception as exc:
        image_debug = {"method": "image_marker_exception", "error": repr(exc)}

    if image_span[1] > image_span[0]:
        image_positions = list(range(int(image_span[0]), int(image_span[1])))
    else:
        text_only_ids, _ = build_text_only_ids(processor, prompt)
        image_positions, image_method, image_debug = locate_image_positions(
            input_ids,
            text_only_ids,
            expected_count,
            processor,
            model,
        )
        image_span = positions_to_span(image_positions)

    question_span = (0, 0)
    question_debug = {"method": "none"}
    try:
        marked_question_prompt = make_prompt_with_marked_question(question, evidence_answer, position)
        marked_question_inputs = build_processor_inputs_mm(processor, marked_question_prompt, image)
        marked_question_ids = marked_question_inputs["input_ids"][0].detach().cpu().tolist()
        question_span, question_debug = locate_marked_span_from_inputs(
            input_ids,
            marked_question_ids,
            processor.tokenizer,
            QUESTION_BEGIN_SENTINEL,
            QUESTION_END_SENTINEL,
        )
    except Exception as exc:
        question_debug = {"method": "question_marker_exception", "error": repr(exc)}

    evidence_span = (0, 0)
    evidence_debug = {"method": "none"}
    if evidence_answer is not None:
        try:
            marked_evidence_prompt = make_prompt_with_marked_evidence(question, evidence_answer, position)
            marked_evidence_inputs = build_processor_inputs_mm(processor, marked_evidence_prompt, image)
            marked_evidence_ids = marked_evidence_inputs["input_ids"][0].detach().cpu().tolist()
            evidence_span, evidence_debug = locate_marked_span_from_inputs(
                input_ids,
                marked_evidence_ids,
                processor.tokenizer,
                EVIDENCE_BEGIN_SENTINEL,
                EVIDENCE_END_SENTINEL,
            )
        except Exception as exc:
            evidence_debug = {"method": "evidence_marker_exception", "error": repr(exc)}

    return {
        "inputs": inputs,
        "input_ids": input_ids,
        "image_positions": image_positions,
        "image_span": image_span,
        "image_locate_method": image_method,
        "image_locate_debug": image_debug,
        "question_span": question_span,
        "question_locate_debug": question_debug,
        "evidence_span": evidence_span,
        "evidence_locate_debug": evidence_debug,
        "expected_image_token_count": expected_count,
    }


def first_answer_token_id(tokenizer, answer):
    text = str(answer).strip()
    if not text:
        return None
    for candidate in (f" {text}", text):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if ids:
            return int(ids[0])
    return None


def find_self_attn_modules(model):
    m = model.module if hasattr(model, "module") else model
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
        ("model", "h"),
        ("gpt_neox", "layers"),
    ]

    layers = None
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
            layers = cur
            break

    if layers is None:
        tried = [".".join(path) for path in candidates]
        raise RuntimeError(f"Cannot locate transformer layers. Tried paths: {tried}")

    modules = {}
    for layer_idx, layer in enumerate(layers):
        for name in ("self_attn", "attn", "attention"):
            if hasattr(layer, name):
                modules[layer_idx] = getattr(layer, name)
                break
    if not modules:
        raise RuntimeError("Cannot find self-attention modules from resolved layers.")
    return modules
