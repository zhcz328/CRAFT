#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_MODEL_NAME="qwen3-4b"
DEFAULT_MODEL_PATH="/archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B"

MODEL_NAME="${MODEL_NAME:-$DEFAULT_MODEL_NAME}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
MODEL_SLUG="${MODEL_SLUG:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

join_output_path() {
  local rel_path="$1"
  if [[ -n "$OUTPUT_ROOT" ]]; then
    printf '%s/%s' "$OUTPUT_ROOT" "$rel_path"
  else
    printf '%s' "$rel_path"
  fi
}

ensure_parent_dir() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
}
