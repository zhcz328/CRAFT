#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_env.sh"

METRICS_ROOT="${METRICS_ROOT:-$ROOT_DIR}"
if [[ -n "$OUTPUT_ROOT" ]]; then
  METRICS_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi

CMD=(
  "$PYTHON_BIN" compute_standard_metrics.py
  --root "$METRICS_ROOT"
)

echo "==> ${CMD[*]}"
"${CMD[@]}"
