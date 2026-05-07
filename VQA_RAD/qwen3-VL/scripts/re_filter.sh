python filter_fine_grained.py \
  --in_csv ./data/nc_cc_both_correct_tmp.xlsx \
  --image_root ./data/VQA_RAD_Image_Folder \
  --model /archive/zengjiaqi/Medical_LLM/Qwen3-VL-8B-Instruct \
  --out_csv ./data/nc_cc_both_correct_rerun.csv \
  --dtype fp16 \
  --device_map auto \
  --resize_max_side 672 \
  --limit 0