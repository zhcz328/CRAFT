import csv
import importlib.util
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from PIL import Image
from mask_utils import UNKNOWN_ANSWER, build_masked_image, extract_inline_mask_data, is_unknown_text

from model_utils import (
    build_chat_template_inputs,
    load_mm_model as shared_load_mm_model,
)
from model_registry import default_eval_results_dir

HALLUCINATION_MAP = {
    "unknown": 0,
    "hallucination": 1,
}

# Backward-compatible alias for older code paths.
FOLLOW_MAP = HALLUCINATION_MAP

SYSTEM_PROMPT = "You are a helpful medical QA assistant."
BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "If the image does not contain enough information to answer the question, output ONLY unknown.\n"
    "Output ONLY the final answer.\n\n"
)


def norm_text(text):
    return str(text or "").strip().lower()


def sample_key(img_id, question):
    return f"{str(img_id).strip()}||{norm_text(question)}"


def read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
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
            raise ValueError("base_prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    raise ValueError(f"Unknown position: {position}")


def make_prompt_no_evidence(question):
    return make_base_prompt(question)


def make_prompt_with_evidence(question, evidence_answer, position):
    return make_base_prompt(question)


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


def derive_follow_label(outcome_record, gold_answer, wrong_answer):
    if not outcome_record:
        return "unknown"
    conflict_pred = norm_text(outcome_record.get("conflict_pred"))
    wrong = norm_text(wrong_answer)
    # For image-conflict, hallucination means the masked-image answer is not unknown.
    if outcome_record.get("ic_is_unknown") or conflict_pred == wrong or is_unknown_text(conflict_pred):
        return "unknown"
    return "hallucination"


def default_preds_path(project_root, position, model_name="", model="hulumed-4b"):
    results_root = default_eval_results_dir(project_root, model_name=model_name, model=model).parent / "eval-results_slake_image_conflict"
    direct = results_root / "preds.jsonl"
    if direct.exists():
        return direct
    recursive_matches = sorted(
        results_root.glob("**/preds.jsonl"),
        key=lambda p: (len(str(p)), str(p)),
    )
    if recursive_matches:
        return recursive_matches[0]
    return direct


def load_outcome_map(preds_jsonl_path):
    if not preds_jsonl_path or not Path(preds_jsonl_path).exists():
        return {}
    rows = read_jsonl(preds_jsonl_path)
    outcome_map = {}
    for row in rows:
        key = sample_key(row.get("img_id", ""), row.get("question", ""))
        outcome_map[key] = {
            "img_id": row.get("img_id"),
            "question": row.get("question"),
            "gold": norm_text(row.get("gold")),
            "wrong": norm_text(row.get("unknown_target", UNKNOWN_ANSWER)),
            "nc_pred": norm_text(row.get("nc_pred_norm") or row.get("nc_pred_raw")),
            "support_pred": norm_text(row.get("nc_pred_norm") or row.get("nc_pred_raw")),
            "conflict_pred": norm_text(row.get("ic_pred_norm") or row.get("ic_pred_raw")),
            "follow_conflict": bool(row.get("ic_is_unknown", False)),
            "ic_is_unknown": bool(row.get("ic_is_unknown", False)),
            "resist": bool(row.get("ic_matches_gold", False)),
            "ic_matches_gold": bool(row.get("ic_matches_gold", False)),
            "changed_vs_nc": bool(row.get("changed_vs_nc", False)),
            "position": row.get("position"),
        }
    return outcome_map


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


def load_mm_model(model_name, device_map=None, torch_dtype=None):
    return shared_load_mm_model(model_name, device_map=device_map, torch_dtype=torch_dtype)


def get_output_head(model):
    head = None
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    if head is None and hasattr(model, "lm_head"):
        head = model.lm_head
    if head is None:
        raise RuntimeError("Model does not expose output embeddings.")
    return head


def resolve_image_path(project_root, image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path
    return project_root / path


def parse_json_list(raw):
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []
    if isinstance(obj, list):
        return [str(item) for item in obj]
    return []


def resize_image_if_needed(image, image_size):
    if not image_size or image_size <= 0:
        return image
    image = image.copy()
    image.thumbnail((image_size, image_size))
    return image


def load_row_image(row, project_root, image_size, mask_scale=1.0):
    image_variant = str(row.get("image_variant", "nc")).strip().lower() or "nc"
    image_path = resolve_image_path(project_root, row["image_path"])
    if image_variant != "ic":
        with Image.open(image_path) as image_file:
            return resize_image_if_needed(image_file.convert("RGB"), image_size)

    detection_path = resolve_image_path(project_root, row["detection_path"])
    mask_path = resolve_image_path(project_root, row["mask_path"])
    target_labels = parse_json_list(row.get("ic_target_labels"))
    inline_mask = extract_inline_mask_data(row)
    masked, _meta = build_masked_image(
        source_path=image_path,
        detection_path=detection_path,
        target_labels=target_labels,
        mask_path=mask_path,
        mask_rle=inline_mask["mask_rle"] if inline_mask is not None else None,
        mask_height=inline_mask["mask_height"] if inline_mask is not None else None,
        mask_width=inline_mask["mask_width"] if inline_mask is not None else None,
        mask_scale=mask_scale,
    )
    return resize_image_if_needed(masked, image_size)


def pad_and_stack_tensors(tensors, key, pad_token_id):
    torch = __import__("torch")

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
    torch = __import__("torch")

    collated = {}
    for key in outputs[0]:
        values = [output[key] for output in outputs]
        if torch.is_tensor(values[0]):
            collated[key] = pad_and_stack_tensors(values, key, pad_token_id)
        else:
            collated[key] = values
    return collated


def build_batch_inputs(processor, rows, project_root, image_size=672, mask_scale=1.0):
    outputs = []
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    for row in rows:
        image = load_row_image(row, project_root, image_size, mask_scale=mask_scale)
        messages = build_messages_multimodal(image, row["prompt_text"])
        outputs.append(build_chat_template_inputs(processor, messages))
    return collate_processor_outputs(outputs, pad_token_id)


def infer_hidden_shape(model, processor, row, project_root, device, image_size=672, mask_scale=1.0):
    import torch

    batch_inputs = build_batch_inputs(processor, [row], project_root, image_size=image_size, mask_scale=mask_scale)
    batch_inputs.pop("token_type_ids", None)
    model_dtype = next(model.parameters()).dtype
    batch_inputs = move_to_device(batch_inputs, device, model_dtype)
    with torch.no_grad():
        out = model(**batch_inputs, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states[1:]
    return len(hidden_states), hidden_states[0].shape[-1]


def single_token_id(tokenizer, answer):
    text = str(answer).strip()
    for candidate in (f" {text}", text):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def answer_token_ids(tokenizer, answer):
    text = str(answer).strip()
    if not text:
        return []

    seen = set()
    for candidate in (f" {text}", text):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        key = tuple(ids)
        if ids and key not in seen:
            seen.add(key)
            return ids
    return []
