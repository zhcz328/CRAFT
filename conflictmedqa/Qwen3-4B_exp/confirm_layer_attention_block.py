from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/archive/zengjiaqi/Medical_LLM/Qwen3_model/Qwen/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
model.eval()

# 打印包含 "attn" / "attention" / "self_attn" 的模块名
for name, m in model.named_modules():
    if any(k in name.lower() for k in ["attn", "attention", "self_attn"]):
        print(name, type(m))
