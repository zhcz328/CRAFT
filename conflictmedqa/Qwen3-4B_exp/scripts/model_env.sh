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
ARTIFACT_MODEL_NAME="${ARTIFACT_MODEL_NAME:-$MODEL_NAME}"
PROBE_SAVE_BASE="${PROBE_SAVE_BASE:-/root/autodl-tmp/probe/conflictmedqa}"
TUNED_LENS_SAVE_BASE="${TUNED_LENS_SAVE_BASE:-/root/autodl-tmp/tuned_lens/conflictmedqa}"

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

probe_save_path() {
  local position="$1"
  printf '%s/%s/%s' "$PROBE_SAVE_BASE" "$ARTIFACT_MODEL_NAME" "$position"
}

tuned_lens_save_path() {
  local position="$1"
  printf '%s/%s/%s' "$TUNED_LENS_SAVE_BASE" "$ARTIFACT_MODEL_NAME" "$position"
}
