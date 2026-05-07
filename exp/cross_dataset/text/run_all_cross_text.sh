#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$ROOT_DIR/run_slake_heads_on_vqarad.sh"
bash "$ROOT_DIR/run_vqarad_heads_on_slake.sh"
