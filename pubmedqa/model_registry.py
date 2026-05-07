from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    slug: str
    hf_repo: str
    family: str
    aliases: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        key="qwen3-4b",
        slug="qwen3_4b",
        hf_repo="Qwen/Qwen3-4B",
        family="qwen3",
        aliases=(
            "qwen3-4b",
            "qwen3-4B",
            "Qwen3-4B",
            "qwen3_4b",
            "Qwen/Qwen3-4B",
        ),
    ),
    ModelSpec(
        key="llama3.2-3b",
        slug="llama32_3b",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        family="llama3_2",
        aliases=(
            "llama3.2-3b",
            "llama3.2-3B",
            "Llama3.2-3B",
            "llama-3.2-3b",
            "llama-3.2-3B",
            "Llama-3.2-3B",
            "llama32-3b",
            "llama32_3b",
            "lama3.2-3b",
            "lama3.2-3B",
            "meta-llama/Llama-3.2-3B-Instruct",
        ),
    ),
)

_ALIAS_TO_SPEC: dict[str, ModelSpec] = {}
for _spec in MODEL_SPECS:
    _ALIAS_TO_SPEC[_spec.key.lower()] = _spec
    _ALIAS_TO_SPEC[_spec.hf_repo.lower()] = _spec
    for _alias in _spec.aliases:
        _ALIAS_TO_SPEC[_alias.lower()] = _spec


def normalize_model_token(value: str) -> str:
    return str(value or "").strip().lower()


def get_model_spec(model_name: str) -> ModelSpec:
    token = normalize_model_token(model_name)
    if token in _ALIAS_TO_SPEC:
        return _ALIAS_TO_SPEC[token]
    raise ValueError(
        f"Unsupported model '{model_name}'. Supported models: "
        f"{', '.join(spec.key for spec in MODEL_SPECS)}"
    )


def infer_model_spec(model_or_path: str) -> ModelSpec | None:
    token = normalize_model_token(model_or_path)
    if not token:
        return None
    for key, spec in _ALIAS_TO_SPEC.items():
        if key and key in token:
            return spec
    return None


def resolve_model_selection(model_name: str = "", model: str = "") -> tuple[str, ModelSpec]:
    if str(model_name or "").strip():
        spec = get_model_spec(model_name)
        resolved_model = str(model or "").strip() or spec.hf_repo
        return resolved_model, spec

    raw_model = str(model or "").strip()
    if not raw_model:
        spec = get_model_spec("qwen3-4b")
        return spec.hf_repo, spec

    spec = infer_model_spec(raw_model)
    if spec is not None:
        if normalize_model_token(raw_model) in _ALIAS_TO_SPEC:
            return spec.hf_repo, spec
        return raw_model, spec

    raise ValueError(
        "Cannot infer model family from --model. "
        "If you are passing a local path, also provide --model_name. "
        "Supported model names: " + ", ".join(spec.key for spec in MODEL_SPECS)
    )


def default_output_root(project_root: str | Path, model_name: str = "", model: str = "") -> Path:
    _, spec = resolve_model_selection(model_name=model_name, model=model)
    return Path(project_root) / spec.slug


def sanitize_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
