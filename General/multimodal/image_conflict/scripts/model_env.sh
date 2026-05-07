#!/usr/bin/env bash

resolve_model_slug() {
  local raw="${1:-hulumed-4b}"
  local lower="${raw,,}"
  case "$lower" in
    *internvl3_5-4b*|*internvl3.5-4b*|*internvl35_4b*|*internvl35-4b*)
      echo "internvl35_4b"
      ;;
    *qwen3.5-4b*|*qwen35-4b*)
      echo "qwen35_4b"
      ;;
    *qwen3vl-4b*|*qwen3-vl-4b*|*qwen3vl_4b*)
      echo "qwen3vl_4b"
      ;;
    *qwen3vl-2b*|*qwen3-vl-2b*|*qwen3vl_2b*)
      echo "qwen3vl_2b"
      ;;
    *hulumed-4b*|*hulu-med-4b*)
      echo "hulumed4b"
      ;;
    *)
      echo "custom_model"
      ;;
  esac
}
