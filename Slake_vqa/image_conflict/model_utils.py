from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from model_registry import (
    default_eval_results_dir,
    infer_model_spec,
    prefix_model_relative_path,
    resolve_model_selection,
    sanitize_path_component,
)


INTERNVL_IMAGE_PLACEHOLDER = "<image>"
INTERNVL_IMG_START_TOKEN = "<img>"
INTERNVL_IMG_END_TOKEN = "</img>"
INTERNVL_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_video_import_compat() -> None:
    for mod_name in ("decord", "ffmpeg", "imageio"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = object
    if not hasattr(image_utils, "VideoOutput"):
        image_utils.VideoOutput = object


def clear_hf_dynamic_modules_for_hulumed() -> None:
    for mod_name in list(sys.modules):
        lower_name = mod_name.lower()
        if not mod_name.startswith("transformers_modules."):
            continue
        if "hulumed" in lower_name or "hulu_med" in lower_name:
            sys.modules.pop(mod_name, None)


def _maybe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _looks_like_local_path(model_name: str) -> bool:
    raw = str(model_name or "").strip()
    if not raw:
        return False
    if raw.startswith((".", "/", "~")):
        return True
    if len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/"):
        return True
    return Path(raw).exists()


def _as_existing_path(model_name: str) -> Path | None:
    raw = str(model_name or "").strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    return expanded if expanded.exists() else None


def _is_internvl_custom_repo(model_name: str) -> bool:
    lower_name = str(model_name or "").lower()
    if "internvl" not in lower_name and not _looks_like_local_path(model_name):
        return False
    local_path = _as_existing_path(model_name)
    if local_path is None:
        return ("internvl" in lower_name) and ("-hf" not in lower_name)
    required = (
        "configuration_internvl_chat.py",
        "modeling_internvl_chat.py",
    )
    return any((local_path / name).exists() for name in required)


def _load_internvl_repo_metadata(model_name: str) -> dict[str, Any]:
    local_path = _as_existing_path(model_name)
    if local_path is None:
        return {}

    config = _maybe_read_json(local_path / "config.json")
    tokenizer_config = _maybe_read_json(local_path / "tokenizer_config.json")
    special_tokens_map = _maybe_read_json(local_path / "special_tokens_map.json")
    processor_config = _maybe_read_json(local_path / "processor_config.json")
    preprocessor_config = _maybe_read_json(local_path / "preprocessor_config.json")

    vision_config = config.get("vision_config", {})
    image_size = (
        config.get("force_image_size")
        or vision_config.get("image_size")
        or preprocessor_config.get("force_image_size")
        or preprocessor_config.get("image_size")
        or processor_config.get("force_image_size")
        or processor_config.get("image_size")
        or 448
    )
    patch_size = vision_config.get("patch_size", 14)
    downsample_ratio = config.get("downsample_ratio", 0.5)
    max_dynamic_patch = config.get("max_dynamic_patch", 12)
    num_image_token = processor_config.get("image_seq_length")
    if not num_image_token:
        num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

    token_attrs = {
        "start_image_token": tokenizer_config.get("start_image_token", INTERNVL_IMG_START_TOKEN),
        "end_image_token": tokenizer_config.get("end_image_token", INTERNVL_IMG_END_TOKEN),
        "context_image_token": tokenizer_config.get("context_image_token", INTERNVL_IMG_CONTEXT_TOKEN),
        "video_token": tokenizer_config.get("video_token", "<video>"),
    }
    for key, fallback in (
        ("bos_token", None),
        ("eos_token", None),
        ("pad_token", None),
    ):
        if token_attrs.get(key) is None:
            token_attrs[key] = special_tokens_map.get(key)

    return {
        "local_path": local_path,
        "config": config,
        "tokenizer_config": tokenizer_config,
        "special_tokens_map": special_tokens_map,
        "processor_config": processor_config,
        "preprocessor_config": preprocessor_config,
        "image_size": int(image_size),
        "patch_size": int(patch_size),
        "downsample_ratio": float(downsample_ratio),
        "max_dynamic_patch": int(max_dynamic_patch),
        "num_image_token": int(num_image_token),
        **token_attrs,
    }


def _build_internvl_transform(input_size: int):
    try:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise ImportError(
            "InternVL original-format preprocessing requires torchvision. "
            "Please install torchvision in the runtime environment."
        ) from exc

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess_internvl(
    image: Image.Image,
    image_size: int = 448,
    min_num: int = 1,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    tiles_per_row = target_width // image_size
    for idx in range(blocks):
        box = (
            (idx % tiles_per_row) * image_size,
            (idx // tiles_per_row) * image_size,
            ((idx % tiles_per_row) + 1) * image_size,
            ((idx // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _extract_images_from_messages(messages) -> list[Image.Image]:
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "image" and ("image" in blk):
                images.append(blk["image"])
    return images


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for blk in content:
        if not isinstance(blk, dict):
            continue
        blk_type = blk.get("type")
        if blk_type == "image":
            parts.append(INTERNVL_IMAGE_PLACEHOLDER)
        elif blk_type == "text":
            parts.append(str(blk.get("text", "")))
    return "".join(parts)


class InternVLCustomProcessor:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.meta = _load_internvl_repo_metadata(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        self.image_size = int(self.meta.get("image_size", 448))
        self.max_dynamic_patch = int(self.meta.get("max_dynamic_patch", 12))
        self.num_image_token = int(self.meta.get("num_image_token", 256))
        self.start_image_token = self.meta.get("start_image_token", INTERNVL_IMG_START_TOKEN)
        self.end_image_token = self.meta.get("end_image_token", INTERNVL_IMG_END_TOKEN)
        self.context_image_token = self.meta.get("context_image_token", INTERNVL_IMG_CONTEXT_TOKEN)
        self.video_token = self.meta.get("video_token", "<video>")
        self._transform = _build_internvl_transform(self.image_size)
        self._patch_tokenizer_attrs()

    def _patch_tokenizer_attrs(self) -> None:
        for attr_name, value in (
            ("start_image_token", self.start_image_token),
            ("end_image_token", self.end_image_token),
            ("context_image_token", self.context_image_token),
            ("video_token", self.video_token),
        ):
            if not hasattr(self.tokenizer, attr_name):
                setattr(self.tokenizer, attr_name, value)
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sanitize_messages_for_template(self, messages):
        sanitized = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                new_content = []
                for blk in content:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "image":
                        new_content.append({"type": "image"})
                    elif blk.get("type") == "text":
                        new_content.append({"type": "text", "text": blk.get("text", "")})
                sanitized.append({"role": msg.get("role", "user"), "content": new_content})
            else:
                sanitized.append({"role": msg.get("role", "user"), "content": str(content)})
        return sanitized

    def _fallback_render_messages(self, messages, add_generation_prompt: bool) -> str:
        rendered_parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            text = _message_content_to_text(msg.get("content", []))
            if role == "system" and text:
                rendered_parts.append(text.strip())
            elif role == "user" and text:
                rendered_parts.append(text.strip())
            elif role == "assistant" and text:
                rendered_parts.append(text.strip())
        text = "\n".join(part for part in rendered_parts if part)
        if add_generation_prompt and text and not text.endswith("\n"):
            text += "\n"
        return text

    def render_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        sanitized = self._sanitize_messages_for_template(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    sanitized,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                pass
        return self._fallback_render_messages(messages, add_generation_prompt)

    def _ensure_image_prompt(self, text: str, image_count: int) -> str:
        text = str(text)
        if image_count > 0 and INTERNVL_IMAGE_PLACEHOLDER not in text:
            return f"{INTERNVL_IMAGE_PLACEHOLDER}\n{text}"
        return text

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        tiles = _dynamic_preprocess_internvl(
            image,
            image_size=self.image_size,
            max_num=self.max_dynamic_patch,
            use_thumbnail=True,
        )
        pixel_values = [self._transform(tile) for tile in tiles]
        return torch.stack(pixel_values)

    def _prepare_single(self, text: str, image_input) -> dict[str, Any]:
        images: list[Image.Image] = []
        if image_input is None:
            images = []
        elif isinstance(image_input, (list, tuple)):
            images = [img for img in image_input if img is not None]
        else:
            images = [image_input]

        pixel_tensors = [self._prepare_image(image) for image in images]
        num_patches_list = [tensor.shape[0] for tensor in pixel_tensors]
        text = self._ensure_image_prompt(text, len(num_patches_list))
        query = text
        for num_patches in num_patches_list:
            image_tokens = (
                self.start_image_token
                + self.context_image_token * (self.num_image_token * num_patches)
                + self.end_image_token
            )
            query = query.replace(INTERNVL_IMAGE_PLACEHOLDER, image_tokens, 1)

        tokenized = self.tokenizer(query, return_tensors="pt")
        if pixel_tensors:
            tokenized["pixel_values"] = torch.cat(pixel_tensors, dim=0)
            tokenized["image_flags"] = torch.ones(
                (tokenized["pixel_values"].shape[0], 1),
                dtype=torch.long,
            )
        return tokenized

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        if return_tensors not in (None, "pt"):
            raise ValueError("InternVL custom processor only supports return_tensors='pt'.")

        if isinstance(text, tuple):
            text = list(text)
        if isinstance(images, tuple):
            images = list(images)

        if isinstance(text, list):
            text_list = [str(item) for item in text]
            if isinstance(images, list):
                image_list = images
            else:
                image_list = [images] * len(text_list)
            if len(image_list) != len(text_list):
                raise ValueError("Number of images must match number of text prompts.")
            features = [self._prepare_single(sample_text, sample_image) for sample_text, sample_image in zip(text_list, image_list)]
            return _collate_processor_features(features, getattr(self.tokenizer, "pad_token_id", 0) or 0)

        return self._prepare_single(str(text or ""), images)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        text = self.render_chat_template(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text

        images = _extract_images_from_messages(messages)
        image_input: Any
        if not images:
            image_input = None
        elif len(images) == 1:
            image_input = images[0]
        else:
            image_input = images
        features = self(text=text, images=image_input, padding=True, return_tensors=return_tensors or "pt")
        return features if return_dict else features["input_ids"]


class InternVLForwardAdapter(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    @property
    def device(self):
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def generate(self, *args, **kwargs):
        try:
            return self.base_model.generate(*args, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "multiple values for keyword argument 'use_cache'" in msg and "use_cache" in kwargs:
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("use_cache", None)
                return self.base_model.generate(*args, **retry_kwargs)
            raise

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        image_flags=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if pixel_values is None:
            return self.base_model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if image_flags is None:
            image_flags = torch.ones(
                (pixel_values.shape[0], 1),
                dtype=torch.long,
                device=pixel_values.device,
            )

        return self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_flags=image_flags,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def _configure_internvl_custom_model(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    from transformers import AutoTokenizer

    meta = _load_internvl_repo_metadata(model_name)
    context_token = meta.get("context_image_token", INTERNVL_IMG_CONTEXT_TOKEN)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        img_context_token_id = tokenizer.convert_tokens_to_ids(context_token)
    except Exception:
        img_context_token_id = None

    wrapped = InternVLForwardAdapter(model)
    wrapped.internvl_meta = meta
    if img_context_token_id is not None and img_context_token_id != getattr(tokenizer, "unk_token_id", None):
        wrapped.base_model.img_context_token_id = img_context_token_id
    return wrapped


def _pad_and_stack_tensors(tensors, key: str, pad_token_id: int):
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


def _collate_processor_features(features, pad_token_id: int):
    collated = {}
    for key in features[0]:
        values = [feature[key] for feature in features]
        if torch.is_tensor(values[0]):
            collated[key] = _pad_and_stack_tensors(values, key, pad_token_id)
        else:
            collated[key] = values
    return collated


def load_processor_with_compat(model_name: str):
    from transformers import AutoProcessor

    ensure_video_import_compat()
    if _is_internvl_custom_repo(model_name):
        return InternVLCustomProcessor(model_name)
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except AttributeError as exc:
        if _is_internvl_custom_repo(model_name) or "start_image_token" in str(exc):
            return InternVLCustomProcessor(model_name)
        raise
    except ImportError as exc:
        msg = str(exc)
        if ("VideoInput" not in msg) and ("VideoOutput" not in msg):
            raise
        ensure_video_import_compat()
        clear_hf_dynamic_modules_for_hulumed()
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def iter_model_loader_classes(model_name: str = ""):
    import transformers

    spec = infer_model_spec(model_name)
    if spec is not None and spec.family == "hulumed":
        names = [
            "AutoModelForCausalLM",
            "AutoModel",
        ]
    else:
        names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ]
    seen = set()
    classes = []
    for name in names:
        cls = getattr(transformers, name, None)
        if cls is None or cls in seen:
            continue
        seen.add(cls)
        classes.append(cls)
    return classes


def load_mm_model(
    model_name,
    device_map=None,
    torch_dtype=None,
    trust_remote_code=True,
    attn_implementation=None,
    quantization_config=None,
):
    load_errors = []
    missing_modules = set()
    for cls in iter_model_loader_classes(model_name):
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
            model = cls.from_pretrained(model_name, **kwargs)
            if _is_internvl_custom_repo(model_name):
                model = _configure_internvl_custom_model(model_name, model)
            return model
        except ModuleNotFoundError as exc:
            if exc.name:
                missing_modules.add(exc.name)
            load_errors.append(f"{cls.__name__}: {repr(exc)}")
        except Exception as exc:
            if isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc):
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                raise RuntimeError(
                    "Failed to load multimodal model because the selected loader hit CUDA OOM. "
                    f"Loader={cls.__name__}, model={model_name}. Original error: {repr(exc)}"
                ) from exc
            load_errors.append(f"{cls.__name__}: {repr(exc)}")
    if missing_modules:
        missing_list = ", ".join(sorted(missing_modules))
        load_errors.append(
            f"Missing Python dependency(s): {missing_list}. "
            "Please install them in the runtime environment before loading the model."
        )
    raise RuntimeError("Failed to load multimodal model. " + " | ".join(load_errors))


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


def extract_first_image_from_messages(messages):
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "image" and ("image" in blk):
                return blk["image"]
    raise RuntimeError("Cannot find image block in messages.")


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
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(
        "Failed to build multimodal processor inputs with explicit image injection. "
        f"Last error: {repr(last_err)}"
    )


def build_chat_template_inputs(processor, messages):
    try:
        base_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if has_non_video_image_inputs(base_inputs):
            return base_inputs
    except Exception:
        pass

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image = extract_first_image_from_messages(messages)
    return build_processor_inputs_mm(processor, text, image)
