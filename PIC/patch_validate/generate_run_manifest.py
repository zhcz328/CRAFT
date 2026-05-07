from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import (
    LOCAL_K_VARIANTS,
    MANIFEST_ROOT,
    PATCH_VARIANTS,
    RESULT_ROOT,
    ensure_result_dirs,
    dump_json,
    get_trace_specs,
)


def write_shell_script(path: Path, commands: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        "",
    ]
    lines.extend(commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def parse_model_keys(model_args: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not model_args:
        return None
    keys: List[str] = []
    for item in model_args:
        for part in item.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    return keys or None


def build_manifest(
    selected_model_keys: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_result_dirs()
    specs = get_trace_specs()
    selected = set(selected_model_keys or [])
    available = {spec.model_key for spec in specs}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(
            "Unknown model_key(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(available))
        )
    entries: List[Dict[str, Any]] = []
    all_commands: List[str] = []
    compare_commands: List[str] = []

    for spec in specs:
        if selected and spec.model_key not in selected:
            continue
        for variant in PATCH_VARIANTS:
            entry = spec.to_manifest_entry(variant, device_override=device)
            entries.append(entry)
            all_commands.append(entry["command"])
            if spec.compare_variant == variant or variant == "all_token":
                if spec.model_key in {"qwen3_4b", "hulumed4b"}:
                    compare_commands.append(entry["command"])

    manifest = {
        "result_root": str(RESULT_ROOT),
        "variants": PATCH_VARIANTS,
        "local_k_variants": LOCAL_K_VARIANTS,
        "experiments": entries,
    }
    dump_json(MANIFEST_ROOT / "patch_trace_manifest.json", manifest)
    write_shell_script(MANIFEST_ROOT / "run_patch_traces_all.sh", all_commands)
    write_shell_script(MANIFEST_ROOT / "run_patch_traces_compare_only.sh", compare_commands)
    write_shell_script(
        MANIFEST_ROOT / "run_patch_analysis.sh",
        [
            "$PYTHON_BIN /root/logit_lens/PIC/patch_validate/compute_patch_metrics.py",
            "$PYTHON_BIN /root/logit_lens/PIC/patch_validate/plot_patch_comparison.py",
        ],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the patch validation command manifest and convenience shell scripts."
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Restrict manifest generation to specific model_key values. "
            "Can be passed multiple times or as a comma-separated list, e.g. "
            "--model qwen3_4b --model hulumed4b or --model qwen3_4b,hulumed4b"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override passed through to the underlying trace scripts, e.g. auto, cuda:0, cuda:1, cpu.",
    )
    args = parser.parse_args()
    manifest = build_manifest(
        selected_model_keys=parse_model_keys(args.model),
        device=args.device,
    )
    print(f"Saved manifest with {len(manifest['experiments'])} trace commands to {MANIFEST_ROOT}")


if __name__ == "__main__":
    main()
