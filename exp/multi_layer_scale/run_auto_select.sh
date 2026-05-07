#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEADSCAN_JSON="${HEADSCAN_JSON:-/root/logit_lens/Slake_vqa/image_conflict/internvl35_4b/result_image_conflict_slake/headscan_slake_mm_accel/head_scan_core_layers.json}"
SOURCE="${SOURCE:-results}"
MODE="${MODE:-conflict_specific}"
TOPK="${TOPK:-50}"
TARGET_MIN="${TARGET_MIN:-6}"
TARGET_MAX="${TARGET_MAX:-7}"
EFF_FLOOR="${EFF_FLOOR:-0.000001}"
EFF_CAP="${EFF_CAP:-0.2}"
BASE_MAX_CAP="${BASE_MAX_CAP:-0.2}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs}"

mkdir -p "$OUT_DIR"

python3 "${ROOT_DIR}/select_heads_auto.py" \
  --input_json "$HEADSCAN_JSON" \
  --source "$SOURCE" \
  --mode "$MODE" \
  --topk "$TOPK" \
  --target_min "$TARGET_MIN" \
  --target_max "$TARGET_MAX" \
  --eff_floor "$EFF_FLOOR" \
  --eff_cap "$EFF_CAP" \
  --base_max_cap "$BASE_MAX_CAP" \
  --out_json "${OUT_DIR}/selected_heads_auto.json" \
  --out_csv "${OUT_DIR}/selected_heads_auto.csv"
