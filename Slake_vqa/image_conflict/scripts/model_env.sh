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
    *qwen3vl-4b*|*qwen3-vl-4b*)
      echo "qwen3vl_4b"
      ;;
    *hulumed-4b*|*hulu-med-4b*)
      echo "hulumed4b"
      ;;
    *)
      echo "custom_model"
      ;;
  esac
}

resolve_dataset_key() {
  local raw="${1:-slake_vqa}"
  local lower="${raw,,}"
  case "$lower" in
    slake|slake_vqa|slake-vqa)
      echo "slake_vqa"
      ;;
    heal|heal_medvqa|heal-medvqa|healmedvqa)
      echo "heal-medvqa"
      ;;
    *)
      echo "slake_vqa"
      ;;
  esac
}

resolve_dataset_save_root() {
  local project_root="${1:?project_root required}"
  local dataset_key
  dataset_key="$(resolve_dataset_key "${2:-slake_vqa}")"
  case "$dataset_key" in
    slake_vqa)
      echo "$project_root"
      ;;
    heal-medvqa)
      echo "/root/logit_lens/heal-medvqa"
      ;;
    *)
      echo "$project_root"
      ;;
  esac
}

resolve_dataset_tag() {
  local dataset_key
  dataset_key="$(resolve_dataset_key "${1:-slake_vqa}")"
  case "$dataset_key" in
    slake_vqa)
      echo "slake"
      ;;
    heal-medvqa)
      echo "heal_medvqa"
      ;;
    *)
      echo "slake"
      ;;
  esac
}

resolve_dataset_stem() {
  local dataset_key
  dataset_key="$(resolve_dataset_key "${1:-slake_vqa}")"
  case "$dataset_key" in
    slake_vqa)
      echo "slake_nc_correct_ic_ready"
      ;;
    heal-medvqa)
      echo "heal_medvqa_nc_correct_ic_ready"
      ;;
    *)
      echo "slake_nc_correct_ic_ready"
      ;;
  esac
}

resolve_dataset_source_csv() {
  local project_root="${1:?project_root required}"
  local dataset_key
  dataset_key="$(resolve_dataset_key "${2:-slake_vqa}")"
  case "$dataset_key" in
    slake_vqa)
      echo "${project_root}/data/slake_full_dataset_visual_local_en_v2_expanded.csv"
      ;;
    heal-medvqa)
      echo "/root/logit_lens/heal-medvqa/data/heal_medvqa_closed_like_slake_max672.csv"
      ;;
    *)
      echo "${project_root}/data/slake_full_dataset_visual_local_en_v2_expanded.csv"
      ;;
  esac
}

resolve_dataset_image_root() {
  local dataset_key
  dataset_key="$(resolve_dataset_key "${1:-slake_vqa}")"
  case "$dataset_key" in
    slake_vqa)
      echo "/root/autodl-tmp/data/SLAKE/imgs"
      ;;
    heal-medvqa)
      echo "/root/autodl-tmp/data/heal-medvqa"
      ;;
    *)
      echo "/root/autodl-tmp/data/SLAKE/imgs"
      ;;
  esac
}
