python filter_fine_grained.py \
  --in_csv ./data/nc_cc_both_correct_tmp.xlsx \
  --image_root ./data/VQA_RAD_Image_Folder \
  --model /archive/zengjiaqi/Medical_LLM/Hulu-Med-4B \
  --out_csv ./data/nc_cc_both_correct_rerun_tmp.csv \
  --dtype fp16 \
  --device_map cuda:0 \
  --resize_max_side 672 \
  --limit 0
