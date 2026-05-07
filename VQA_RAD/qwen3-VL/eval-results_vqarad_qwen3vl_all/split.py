import pandas as pd

# 输入文件
input_csv = r"/home/zengjiaqi/icl/interp/logit_lens/cross_modal/eval-results_vqarad_qwen3vl_all/_archive_zengjiaqi_Medical_LLM_Qwen3-VL-8B-Instruct/before_question/nc_cc_both_correct.csv"

# 输出文件
first_300_csv = r"/home/zengjiaqi/icl/interp/logit_lens/cross_modal/eval-results_vqarad_qwen3vl_all/_archive_zengjiaqi_Medical_LLM_Qwen3-VL-8B-Instruct/before_question/nc_cc_both_correct_first300.csv"
rest_csv = r"/home/zengjiaqi/icl/interp/logit_lens/cross_modal/eval-results_vqarad_qwen3vl_all/_archive_zengjiaqi_Medical_LLM_Qwen3-VL-8B-Instruct/before_question/nc_cc_both_correct_rest.csv"

# 读取并切分
df = pd.read_csv(input_csv)
df_first_300 = df.iloc[:300]
df_rest = df.iloc[300:]

# 保存
df_first_300.to_csv(first_300_csv, index=False)
df_rest.to_csv(rest_csv, index=False)

print(f"前300行已保存: {first_300_csv}")
print(f"其余行已保存: {rest_csv}")
