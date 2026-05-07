from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model_registry import MODEL_SPECS, sanitize_path_component


TASK_PRESETS: dict[str, dict[str, Any]] = {
    "head_scan": {
        "script": "scripts/head_scan.sh",
        "description": "Run standard head scan.",
        "env": {},
    },
    "head_scan_accel": {
        "script": "scripts/head_scan_accel.sh",
        "description": "Run accelerated head scan.",
        "env": {},
    },
    "layer_trace_then_head_scan_accel": {
        "scripts": [
            "scripts/layer_trace.sh",
            "scripts/head_scan_accel.sh",
        ],
        "description": "Run layer trace first, then run accelerated head scan with the same resolved environment.",
        "env": {},
    },
    "head_scan_accel_parallel": {
        "script": "scripts/head_scan_accel_parallel.sh",
        "description": "Run accelerated head scan across multiple GPUs and merge shards.",
        "env": {},
    },
    "head_scan_96g": {
        "script": "scripts/head_scan_96g.sh",
        "description": "Run the 96G accelerated head scan preset.",
        "env": {},
    },
    "layer_trace": {
        "script": "scripts/layer_trace.sh",
        "description": "Run layer trace with the repo default preset.",
        "env": {},
    },
    "layer_trace_parallel": {
        "script": "scripts/layer_trace_parallel.sh",
        "description": "Run layer trace across multiple GPUs and merge shards.",
        "env": {},
    },
    "eval": {
        "script": "scripts/eval_slake.sh",
        "description": "Run support-vs-conflict evaluation.",
        "env": {},
    },
    "split_train_val": {
        "script": "scripts/split_train_val.sh",
        "description": "Split one model-specific filtered CSV into train/val CSVs.",
        "env": {},
    },
    "re_filter": {
        "script": "scripts/re_filter.sh",
        "description": "Run the fine-grained NC/CC re-filter stage.",
        "env": {},
    },
    "select_heads": {
        "script": "scripts/select.sh",
        "description": "Select heads from the position-specific head-scan output.",
        "env": {},
    },
    "ablate_head": {
        "script": "scripts/ablate_head.sh",
        "description": "Run selected-head ablation on both train and val by default.",
        "env": {},
    },
    "ablate_head_random": {
        "script": "scripts/ablate_head_random.sh",
        "description": "Run random-head ablation on both train and val by default.",
        "env": {},
    },
    "probe_lens": {
        "script": "scripts/run_probe_tuned_lens.sh",
        "description": "Run probe and tuned-lens pipeline.",
        "env": {},
    },
    "ablation_flip_analyze": {
        "script": "scripts/analyze_ablation_flip_with_probe_and_lens.sh",
        "description": "Analyze wrong->correct ablation flips with trained probes and tuned lens.",
        "env": {},
    },
    "probe_prepare_data": {
        "script": "scripts/probe/prepare_data.sh",
        "description": "Prepare probe manifests for one model and one position.",
        "env": {},
    },
    "probe_extract_features": {
        "script": "scripts/probe/extract_features.sh",
        "description": "Extract train/val probe features for one model and one position.",
        "env": {},
    },
    "probe_train": {
        "script": "scripts/probe/train_probe.sh",
        "description": "Train both conflict and follow_conflict probes.",
        "env": {},
    },
    "probe_plot_metrics": {
        "script": "scripts/probe/plot_probe.sh",
        "description": "Plot validation curves for both probe tasks.",
        "env": {},
    },
    "probe_plot_traj": {
        "script": "scripts/probe/plot_traj.sh",
        "description": "Plot mean trajectories for both trained probes.",
        "env": {},
    },
    "probe_score_manifest": {
        "script": "scripts/probe/score_probe_on_manifest.sh",
        "description": "Score a manifest with both trained probes.",
        "env": {},
    },
    "tuned_lens_prepare_data": {
        "script": "scripts/tuned_lens/prepare_data.sh",
        "description": "Prepare tuned-lens manifests for one model and one position.",
        "env": {},
    },
    "tuned_lens_train": {
        "script": "scripts/tuned_lens/train_tuned_lens.sh",
        "description": "Train tuned-lens translators for one model and one position.",
        "env": {},
    },
    "tuned_lens_plot_alignment": {
        "script": "scripts/tuned_lens/plot_alighment.sh",
        "description": "Plot tuned-lens alignment curves.",
        "env": {},
    },
    "tuned_lens_analyse_traj": {
        "script": "scripts/tuned_lens/analyse_traj.sh",
        "description": "Analyze tuned-lens trajectories.",
        "env": {},
    },
}


def build_builtin_models() -> dict[str, dict[str, Any]]:
    presets: dict[str, dict[str, Any]] = {}
    for spec in MODEL_SPECS:
        aliases = sorted({spec.slug, spec.key, spec.hf_repo, *spec.aliases})
        presets[spec.slug] = {
            "model_path": spec.hf_repo,
            "model_name": spec.key,
            "aliases": aliases,
            "family": spec.family,
        }
    return presets


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return data


def load_presets(config_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], Path | None]:
    models = build_builtin_models()
    tasks = deepcopy(TASK_PRESETS)
    used_config = None

    if config_path is None:
        candidate = SCRIPT_DIR / "run_job.local.json"
        if candidate.exists():
            config_path = candidate

    if config_path is not None:
        cfg = load_json_config(config_path)
        used_config = config_path
        for key, value in cfg.get("models", {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"Config models.{key} must be an object.")
            models[key] = deep_merge(models.get(key, {}), value)
        for key, value in cfg.get("tasks", {}).items():
            if not isinstance(value, dict):
                raise ValueError(f"Config tasks.{key} must be an object.")
            tasks[key] = deep_merge(tasks.get(key, {}), value)

    return models, tasks, used_config


def build_model_alias_map(models: dict[str, dict[str, Any]]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for key, spec in models.items():
        alias_map[str(key).strip().lower()] = key
        for alias in spec.get("aliases", []):
            alias_map[str(alias).strip().lower()] = key
    return alias_map


def resolve_key(raw: str, alias_map: dict[str, str], label: str) -> str:
    token = str(raw or "").strip().lower()
    if not token:
        raise ValueError(f"--{label} is required.")
    if token in alias_map:
        return alias_map[token]
    raise ValueError(f"Unknown {label}: {raw}")


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env value '{item}'. Use KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --env value '{item}'. Empty key.")
        overrides[key] = value
    return overrides


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def normalize_task_scripts(task_spec: dict[str, Any]) -> list[str]:
    script_rel = str(task_spec.get("script", "")).strip()
    if script_rel:
        return [script_rel]

    script_list = task_spec.get("scripts", [])
    if not isinstance(script_list, list):
        raise ValueError("Task preset field 'scripts' must be a list.")

    resolved = [str(item).strip() for item in script_list if str(item).strip()]
    if resolved:
        return resolved

    raise ValueError("Task preset is missing script/scripts.")


def print_listing(models: dict[str, dict[str, Any]], tasks: dict[str, dict[str, Any]]) -> None:
    print("Models:")
    for key in sorted(models):
        spec = models[key]
        print(f"  {key}: model_path={spec.get('model_path', '')}")
    print("")
    print("Tasks:")
    for key in sorted(tasks):
        spec = tasks[key]
        print(f"  {key}: {spec.get('description', '')}")


def to_wsl_path(path: Path) -> str:
    raw = str(path)
    if len(raw) >= 3 and raw[1:3] == ":\\":
        drive = raw[0].lower()
        tail = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    return raw.replace("\\", "/")


def resolve_shell_prefix(allow_placeholder: bool = False) -> tuple[list[str], str]:
    bash_path = shutil.which("bash")
    if bash_path is not None:
        return [bash_path], "bash"

    wsl_path = shutil.which("wsl")
    if wsl_path is not None:
        return [wsl_path, "bash"], "wsl"

    if allow_placeholder:
        return ["bash"], "unavailable"

    raise RuntimeError("Cannot find 'bash' or 'wsl' in PATH. This launcher runs the existing .sh wrappers.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Launch a text_conflict task by task+model so you do not have to retype long parameter sets."
    )
    ap.add_argument("--task", default="", help="Task preset, e.g. head_scan, layer_trace, eval.")
    ap.add_argument("--model", default="", help="Model preset key or alias, e.g. qwen35_4b.")
    ap.add_argument("--config", default="", help="Optional JSON config path. Defaults to scripts/run_job.local.json if it exists.")
    ap.add_argument("--env", action="append", default=[], help="Extra environment overrides in KEY=VALUE form.")
    ap.add_argument("--label", default="", help="Optional short label appended to the run record filename.")
    ap.add_argument("--resume", action="store_true", help="Set RESUME=1 for wrappers that support checkpoint recovery.")
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved command and save the run record without executing.")
    ap.add_argument("--list", action="store_true", help="List known tasks and models.")
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve() if args.config else None
    models, tasks, used_config = load_presets(config_path)

    if args.list:
        print_listing(models, tasks)
        return 0

    model_alias_map = build_model_alias_map(models)
    task_alias_map = {key.lower(): key for key in tasks}

    model_key = resolve_key(args.model, model_alias_map, "model")
    task_key = resolve_key(args.task, task_alias_map, "task")

    model_spec = models[model_key]
    task_spec = tasks[task_key]

    model_path = str(model_spec.get("model_path", "")).strip()
    model_name = str(model_spec.get("model_name", "")).strip()
    script_rels = normalize_task_scripts(task_spec)
    if not model_path:
        raise ValueError(f"Model preset '{model_key}' is missing model_path.")
    script_paths = [ROOT_DIR / script_rel for script_rel in script_rels]
    missing_script_paths = [str(path) for path in script_paths if not path.exists()]
    if missing_script_paths:
        raise FileNotFoundError("Task script not found:\n" + "\n".join(missing_script_paths))

    env = os.environ.copy()
    for source in (task_spec.get("env", {}), model_spec.get("env", {}), parse_env_overrides(args.env)):
        for key, value in source.items():
            env[str(key)] = str(value)
    env["MODEL_PATH"] = model_path
    env["MODEL_NAME"] = model_name
    if args.resume:
        env["RESUME"] = "1"

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{sanitize_path_component(args.label)}" if args.label else ""
    record_dir = ROOT_DIR / "runs"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{now}_{model_key}_{task_key}{label_part}.json"

    shell_prefix, shell_kind = resolve_shell_prefix(allow_placeholder=args.dry_run)
    record = {
        "timestamp": now,
        "task": task_key,
        "model": model_key,
        "scripts": [str(path) for path in script_paths],
        "shell": shell_kind,
        "cwd": str(ROOT_DIR),
        "config": str(used_config) if used_config else "",
        "dry_run": bool(args.dry_run),
        "commands": [],
        "env_overrides": {
            "MODEL_PATH": env["MODEL_PATH"],
            "MODEL_NAME": env["MODEL_NAME"],
            **({"RESUME": "1"} if args.resume else {}),
            **{k: env[k] for k in sorted(parse_env_overrides(args.env))},
        },
        "status": "prepared",
    }
    with record_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"Task:        {task_key}")
    print(f"Model:       {model_key}")
    print(f"Model path:  {model_path}")
    print(f"Model name:  {model_name}")
    if len(script_paths) == 1:
        print(f"Script:      {script_paths[0]}")
    else:
        print("Scripts:")
        for path in script_paths:
            print(f"  {path}")
    print(f"Run record:  {record_path}")
    if args.env:
        print("Overrides:")
        for item in args.env:
            print(f"  {item}")

    commands: list[list[str]] = []
    for script_path in script_paths:
        script_target = to_wsl_path(script_path) if shell_kind == "wsl" else str(script_path)
        command = [*shell_prefix, script_target]
        commands.append(command)
        print(f"Command:     {format_command(command)}")

    record["commands"] = commands
    record["command_texts"] = [format_command(cmd) for cmd in commands]
    if commands:
        record["command"] = commands[0]
        record["command_text"] = record["command_texts"][0]

    if args.dry_run:
        record["status"] = "dry_run"
        with record_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return 0

    try:
        record["steps"] = []
        returncode = 0
        for idx, (script_path, command) in enumerate(zip(script_paths, commands), start=1):
            completed = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
            step = {
                "index": idx,
                "script": str(script_path),
                "command": command,
                "command_text": format_command(command),
                "returncode": int(completed.returncode),
            }
            record["steps"].append(step)
            returncode = int(completed.returncode)
            if completed.returncode != 0:
                break
        record["status"] = "finished" if returncode == 0 else "failed"
        record["returncode"] = returncode
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        raise
    finally:
        with record_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    return int(record.get("returncode", 0))


if __name__ == "__main__":
    raise SystemExit(main())
