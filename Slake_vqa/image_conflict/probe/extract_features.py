import argparse
import importlib.util
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import build_messages_multimodal, read_jsonl
from mask_utils import build_masked_image, extract_inline_mask_data
from model_utils import (
    build_chat_template_inputs,
    load_mm_model as shared_load_mm_model,
    load_processor_with_compat,
    prefix_model_relative_path,
    resolve_model_selection,
)
from resume_utils import ResumeTracker, build_resume_dir, build_resume_scope


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
    return shared_load_mm_model(model_name, device_map=device_map, torch_dtype=torch_dtype)


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
        obj = __import__("json").loads(text)
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


def build_batch_inputs(processor, rows, project_root, image_size, mask_scale=1.0):
    outputs = []
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    for row in rows:
        image = load_row_image(row, project_root, image_size, mask_scale=mask_scale)
        messages = build_messages_multimodal(image, row["prompt_text"])
        outputs.append(build_chat_template_inputs(processor, messages))
    return collate_processor_outputs(outputs, pad_token_id)


def infer_position_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem
    for candidate in ("image_conflict", "before_question", "before_answer", "prefix"):
        if stem.endswith(candidate):
            return candidate
    return ""


def main():
    ap = argparse.ArgumentParser(description="Extract Hulu-med multimodal hidden states for probe training.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out = prefix_model_relative_path(args.out, model_name=args.model_name, model=args.model)

    ensure_video_import_compat()
    manifest_path = Path(args.manifest)
    project_root = Path(__file__).resolve().parents[1]
    position = infer_position_from_manifest(manifest_path)
    resume_scope = build_resume_scope(manifest_path)
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=project_root,
            task_name="probe_extract_features",
            model_name=args.model_name,
            model=args.model,
            position=position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="probe_extract_features",
        manifest=str(manifest_path),
        out=args.out,
        position=position,
        mask_scale=float(args.mask_scale),
    )
    rows = read_jsonl(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    processor = load_processor_with_compat(args.model)
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()
    first_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    batches = list(chunked(rows, args.batch_size))
    progress = tqdm(
        list(enumerate(batches)),
        total=len(batches),
        desc="extract features",
        unit="batch",
    )
    if args.resume:
        shard_dir = tracker.resume_dir / "feature_shards"
    else:
        out_hint = Path(args.out)
        shard_dir = out_hint.parent / f".{out_hint.stem}_feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch_rows in progress:
        shard_path = shard_dir / f"batch_{batch_idx:06d}.pt"
        batch_key = f"batch:{batch_idx}"
        if args.resume and tracker.is_done(batch_key) and shard_path.exists():
            progress.set_postfix(resumed=batch_idx)
            continue
        batch_inputs = build_batch_inputs(
            processor,
            batch_rows,
            project_root,
            args.image_size,
            mask_scale=args.mask_scale,
        )
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
        shard_payload = {
            "batch_idx": batch_idx,
            "hidden_states": torch.stack(layer_states, dim=1),
            "sample_ids": [row["record_id"] for row in batch_rows],
            "sample_keys": [row["sample_key"] for row in batch_rows],
            "pair_ids": [row["sample_id"] for row in batch_rows],
            "img_ids": [row["img_id"] for row in batch_rows],
            "image_paths": [row["image_path"] for row in batch_rows],
            "prompt_types": [row["prompt_type"] for row in batch_rows],
            "gold_answers": [row["gold_answer"] for row in batch_rows],
            "wrong_answers": [row["wrong_answer"] for row in batch_rows],
            "conflict_targets": [int(row["conflict_target"]) for row in batch_rows],
            "follow_targets": [int(row["follow_target"]) for row in batch_rows],
            "hallucination_targets": [int(row.get("hallucination_target", row["follow_target"])) for row in batch_rows],
            "splits": [row["split"] for row in batch_rows],
            "mask_scale": float(args.mask_scale),
        }
        torch.save(shard_payload, shard_path)
        tracker.mark_done(batch_key, {"batch_idx": int(batch_idx), "n_rows": len(batch_rows)})
        tracker.update(completed_batches=tracker.completed_count, total_batches=len(batches))
        progress.set_postfix(completed=tracker.completed_count)

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
        "hallucination_targets": [],
        "splits": [],
    }
    for batch_idx in range(len(batches)):
        shard_path = shard_dir / f"batch_{batch_idx:06d}.pt"
        if not shard_path.exists():
            raise RuntimeError(f"Missing feature shard: {shard_path}")
        shard_payload = torch.load(shard_path, map_location="cpu")
        hidden_batches.append(shard_payload["hidden_states"])
        metadata["sample_ids"].extend(shard_payload["sample_ids"])
        metadata["sample_keys"].extend(shard_payload["sample_keys"])
        metadata["pair_ids"].extend(shard_payload["pair_ids"])
        metadata["img_ids"].extend(shard_payload["img_ids"])
        metadata["image_paths"].extend(shard_payload["image_paths"])
        metadata["prompt_types"].extend(shard_payload["prompt_types"])
        metadata["gold_answers"].extend(shard_payload["gold_answers"])
        metadata["wrong_answers"].extend(shard_payload["wrong_answers"])
        metadata["conflict_targets"].extend(shard_payload["conflict_targets"])
        metadata["follow_targets"].extend(shard_payload["follow_targets"])
        metadata["splits"].extend(shard_payload["splits"])

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
    tracker.finish(out=str(out_path), n_samples=int(hidden_tensor.shape[0]), n_layers=int(hidden_tensor.shape[1]))


if __name__ == "__main__":
    main()
