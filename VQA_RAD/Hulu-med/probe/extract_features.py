import argparse
import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import build_messages_multimodal, read_jsonl


def ensure_video_import_compat():
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def move_to_device(obj, device, float_dtype=None):
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
    load_errors = []
    for cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            kwargs = {"trust_remote_code": True}
            if device_map is not None:
                kwargs["device_map"] = device_map
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            load_errors.append(f"{cls.__name__}: {repr(exc)}")
    raise RuntimeError("Failed to load multimodal model. " + " | ".join(load_errors))


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
    collated = {}
    for key in outputs[0]:
        values = [output[key] for output in outputs]
        if torch.is_tensor(values[0]):
            collated[key] = pad_and_stack_tensors(values, key, pad_token_id)
        else:
            collated[key] = values
    return collated


def build_batch_inputs(processor, rows, project_root, image_size):
    outputs = []
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    for row in rows:
        with Image.open(resolve_image_path(project_root, row["image_path"])) as image_file:
            image = resize_image_if_needed(image_file.convert("RGB"), image_size)
        messages = build_messages_multimodal(image, row["prompt_text"])
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs.append(
            processor(
                text=text,
                images=image,
                return_tensors="pt",
            )
        )
    return collate_processor_outputs(outputs, pad_token_id)


def main():
    ap = argparse.ArgumentParser(description="Extract Hulu-med multimodal hidden states for probe training.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    ensure_video_import_compat()
    manifest_path = Path(args.manifest)
    project_root = Path(__file__).resolve().parents[1]
    rows = read_jsonl(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    hidden_batches = []
    metadata = {
        "sample_ids": [],
        "sample_keys": [],
        "pair_ids": [],
        "img_ids": [],
        "image_paths": [],
        "prompt_types": [],
        "gold_answers": [],
        "wrong_answers": [],
        "conflict_targets": [],
        "follow_targets": [],
        "splits": [],
    }

    batches = list(chunked(rows, args.batch_size))
    progress = tqdm(
        batches,
        total=len(batches),
        desc="extract features",
        unit="batch",
    )

    for batch_rows in progress:
        batch_inputs = build_batch_inputs(processor, batch_rows, project_root, args.image_size)
        batch_inputs.pop("token_type_ids", None)
        batch_inputs = move_to_device(batch_inputs, first_device, model_dtype)
        positions = batch_inputs["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

        hidden_states = list(out.hidden_states)
        if not args.include_embeddings:
            hidden_states = hidden_states[1:]

        layer_states = []
        for hs in hidden_states:
            batch_index = torch.arange(len(batch_rows), device=hs.device)
            local_positions = positions.to(hs.device)
            last_token = hs[batch_index, local_positions]
            layer_states.append(last_token.detach().to(resolve_dtype(args.save_dtype)).cpu())
        hidden_batches.append(torch.stack(layer_states, dim=1))

        for row in batch_rows:
            metadata["sample_ids"].append(row["record_id"])
            metadata["sample_keys"].append(row["sample_key"])
            metadata["pair_ids"].append(row["sample_id"])
            metadata["img_ids"].append(row["img_id"])
            metadata["image_paths"].append(row["image_path"])
            metadata["prompt_types"].append(row["prompt_type"])
            metadata["gold_answers"].append(row["gold_answer"])
            metadata["wrong_answers"].append(row["wrong_answer"])
            metadata["conflict_targets"].append(row["conflict_target"])
            metadata["follow_targets"].append(row["follow_target"])
            metadata["splits"].append(row["split"])

        progress.set_postfix(samples=len(metadata["sample_ids"]))

    hidden_tensor = torch.cat(hidden_batches, dim=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "manifest_path": str(manifest_path),
            "hidden_states": hidden_tensor,
            "sample_ids": metadata["sample_ids"],
            "sample_keys": metadata["sample_keys"],
            "pair_ids": metadata["pair_ids"],
            "img_ids": metadata["img_ids"],
            "image_paths": metadata["image_paths"],
            "prompt_types": metadata["prompt_types"],
            "gold_answers": metadata["gold_answers"],
            "wrong_answers": metadata["wrong_answers"],
            "conflict_targets": torch.tensor(metadata["conflict_targets"], dtype=torch.long),
            "follow_targets": torch.tensor(metadata["follow_targets"], dtype=torch.long),
            "splits": metadata["splits"],
            "include_embeddings": args.include_embeddings,
            "n_layers": hidden_tensor.shape[1],
            "hidden_dim": hidden_tensor.shape[2],
        },
        out_path,
    )

    print(f"saved={out_path}")
    print(f"n_samples={hidden_tensor.shape[0]}")
    print(f"n_layers={hidden_tensor.shape[1]}")
    print(f"hidden_dim={hidden_tensor.shape[2]}")


if __name__ == "__main__":
    main()
