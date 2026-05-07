export CUDA_VISIBLE_DEVICES=0
python eval_models.py \
  --data /home/zengjiaqi/icl/interp/logit_lens/pubmedqa/ori_pqal.json \
  --fmt json \
  --models \
    /archive/zengjiaqi/Medical_LLM/Med-Qwen2-7B/ \
    /archive/zengjiaqi/Medical_LLM/Qwen3-8B-Hippocratesv1/ \
    /archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B \
    /archive/zengjiaqi/Medical_LLM/Qwen3-8B-Medical-4bit/ \
  --limit 1000 \
  --out_dir ./result/pubmedqa_conflict_out_logit \
  --mode logit
 