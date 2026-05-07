from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_registry import MODEL_SPECS, resolve_model_selection, sanitize_path_component


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_resume_scope(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        path = Path(raw)
        token = path.stem or path.name or raw
        token = sanitize_path_component(token)
        if token:
            parts.append(token)
    return "__".join(parts)


def build_resume_dir(
    project_root: str | Path,
    task_name: str,
    model_name: str = "",
    model: str = "",
    position: str = "",
    scope: str = "",
) -> Path:
    _, spec = resolve_model_selection(model_name=model_name, model=model)
    resume_dir = Path(project_root) / spec.slug / "resume" / sanitize_path_component(task_name)
    if str(position or "").strip():
        resume_dir = resume_dir / sanitize_path_component(position)
    if str(scope or "").strip():
        resume_dir = resume_dir / sanitize_path_component(scope)
    return resume_dir


def build_resume_dir_from_output(
    project_root: str | Path,
    task_name: str,
    output_path: str | Path,
    position: str = "",
    scope: str = "",
) -> Path:
    rel = Path(output_path)
    parts = [part for part in rel.parts if part and part != rel.anchor]
    known_slugs = {sanitize_path_component(spec.slug) for spec in MODEL_SPECS}
    model_slug = next((sanitize_path_component(part) for part in parts if sanitize_path_component(part) in known_slugs), "")
    if not model_slug:
        model_slug = sanitize_path_component(parts[0] if parts else "custom_model")
    resume_dir = Path(project_root) / model_slug / "resume" / sanitize_path_component(task_name)
    if str(position or "").strip():
        resume_dir = resume_dir / sanitize_path_component(position)
    if str(scope or "").strip():
        resume_dir = resume_dir / sanitize_path_component(scope)
    return resume_dir


@dataclass
class ResumeTracker:
    resume_dir: Path
    enabled: bool = False

    def __post_init__(self) -> None:
        self.resume_dir = Path(self.resume_dir)
        self.state_path = self.resume_dir / "resume_state.json"
        self.done_path = self.resume_dir / "completed_keys.jsonl"
        self.events_path = self.resume_dir / "events.jsonl"
        self.completed_keys: set[str] = set()
        self.state: dict[str, Any] = {}
        if self.enabled:
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            self.state = load_json(self.state_path, default={}) or {}
            for item in read_jsonl(self.done_path):
                key = str(item.get("key", "")).strip()
                if key:
                    self.completed_keys.add(key)

    @property
    def completed_count(self) -> int:
        return len(self.completed_keys)

    def is_done(self, key: str) -> bool:
        return self.enabled and str(key) in self.completed_keys

    def start(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        base = dict(self.state)
        base.update(metadata)
        base["status"] = "running"
        base["completed_count"] = self.completed_count
        base["updated_at"] = _utc_now()
        self.state = base
        atomic_write_json(self.state_path, self.state)

    def update(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self.state.update(metadata)
        self.state["completed_count"] = self.completed_count
        self.state["updated_at"] = _utc_now()
        atomic_write_json(self.state_path, self.state)

    def mark_done(self, key: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        key = str(key)
        if not key or key in self.completed_keys:
            return
        self.completed_keys.add(key)
        append_jsonl(
            self.done_path,
            {
                "key": key,
                "timestamp": _utc_now(),
            },
        )
        if payload:
            event = dict(payload)
            event["key"] = key
            event["timestamp"] = _utc_now()
            append_jsonl(self.events_path, event)
        self.update()

    def append_record(self, file_name: str, payload: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        return append_jsonl(self.resume_dir / file_name, payload)

    def read_records(self, file_name: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return read_jsonl(self.resume_dir / file_name)

    def finish(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self.state.update(metadata)
        self.state["status"] = "completed"
        self.state["completed_count"] = self.completed_count
        self.state["updated_at"] = _utc_now()
        atomic_write_json(self.state_path, self.state)
