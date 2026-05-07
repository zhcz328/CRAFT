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
        key="hulumed-4b",
        slug="hulumed4b",
        hf_repo="ZJU-AI4H/Hulu-Med-4B",
        family="hulumed",
        aliases=("hulumed-4b", "hulumed-4B", "hulu-med-4b", "hulu-med-4B", "Hulu-Med-4B"),
    ),
    ModelSpec(
        key="internvl3_5-4b",
        slug="internvl35_4b",
        hf_repo="OpenGVLab/InternVL3_5-4B-HF",
        family="internvl",
        aliases=(
            "internvl3_5-4b",
            "internvl3_5-4B",
            "InternVL3_5-4B",
            "internvl3.5-4b",
            "internvl3.5-4B",
            "InternVL3.5-4B",
            "OpenGVLab/InternVL3_5-4B",
            "OpenGVLab/InternVL3_5-4B-HF",
        ),
    ),
    ModelSpec(
        key="qwen3.5-4b",
        slug="qwen35_4b",
        hf_repo="Qwen/Qwen3.5-4B",
        family="qwen3_5",
        aliases=("qwen3.5-4b", "qwen3.5-4B", "Qwen3.5-4B", "qwen35-4b", "qwen35-4B"),
    ),
    ModelSpec(
        key="qwen3vl-4b",
        slug="qwen3vl_4b",
        hf_repo="Qwen/Qwen3-VL-4B-Instruct",
        family="qwen3_vl",
        aliases=(
            "qwen3vl-4b",
            "qwen3vl-4B",
            "qwen3-vl-4b",
            "qwen3-vl-4B",
            "qwen3-vl-4b-instruct",
            "Qwen3-VL-4B-Instruct",
        ),
    ),
)

_ALIAS_TO_SPEC = {}
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
        spec = get_model_spec("hulumed-4b")
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


def default_eval_results_dir(project_root: str | Path, model_name: str = "", model: str = "") -> Path:
    _, spec = resolve_model_selection(model_name=model_name, model=model)
    return Path(project_root) / spec.slug / "eval-results_vqarad_all"


def sanitize_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def prefix_model_relative_path(path_value: str, model_name: str = "", model: str = "") -> str:
    if not str(path_value or "").strip():
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    _, spec = resolve_model_selection(model_name=model_name, model=model)
    if path.parts and path.parts[0] == spec.slug:
        return str(path)
    return str(Path(spec.slug) / path)

