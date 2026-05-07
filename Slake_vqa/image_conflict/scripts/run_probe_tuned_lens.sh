#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash "$ROOT_DIR/scripts/probe/prepare_data.sh" "$@"
bash "$ROOT_DIR/scripts/probe/extract_features.sh" "$@"
bash "$ROOT_DIR/scripts/probe/train_probe.sh" "$@"
bash "$ROOT_DIR/scripts/tuned_lens/prepare_data.sh" "$@"
bash "$ROOT_DIR/scripts/tuned_lens/train_tuned_lens.sh" "$@"
