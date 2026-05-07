import json
import random
import re
import argparse
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

LABELS = ["yes", "no", "maybe"]
FLIP = {"yes": "no", "no": "yes", "maybe": "maybe"}  # maybe 默认不翻

SYSTEM_BASE = (
    "You are a medical QA classifier. "
    "Your output MUST be exactly one of: yes, no, maybe. "
    "Output ONLY that single word."
)

SYSTEM_NO_THINK = SYSTEM_BASE + " Do NOT output <think> or any reasoning."

def normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    # 去掉 <think>...</think> 内容（如果模型输出了）
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL)
    # 再找标签
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", t):
            return lab
    return "unknown"


def build_context(item: Dict[str, Any]) -> str:
    ctx = item.get("CONTEXTS") or item.get("contexts") or item.get("context")
    if isinstance(ctx, list):
        return "\n".join(str(x) for x in ctx)
    return str(ctx or "")

def build_question(item: Dict[str, Any]) -> str:
    return item.get("QUESTION") or item.get("question") or ""

def build_gold(item: Dict[str, Any]) -> str:
    g = item.get("final_decision") or item.get("label") or ""
    return str(g).strip().lower()

def prompt_baseline(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

def prompt_conflict(context: str, question: str, flipped_answer: str) -> str:
    demo = (
        "Example (may contain errors):\n"
        f"Question: {question}\n"
        f"Answer: {flipped_answer}\n\n"
    )
    return demo + f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"



def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    system = SYSTEM_BASE if enable_thinking else SYSTEM_NO_THINK

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # tokenizer 不支持 enable_thinking 参数时，退化为普通调用
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return f"{system}\n\n{user_prompt}"


@torch.no_grad()
def generate_one(
    model, tokenizer, prompt: str,
    enable_thinking: bool,
    max_new_tokens: int = 3,
    temperature: float = 0.0
) -> str:
    text = format_chat(tokenizer, prompt, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt")

    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


def acc(preds: List[str], golds: List[str]) -> float:
    pairs = [(p, g) for p, g in zip(preds, golds) if p in LABELS and g in LABELS]
    return sum(p == g for p, g in pairs) / max(1, len(pairs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ori_pqal.json")
    ap.add_argument("--model", required=True, help="Local path of Qwen3-4B or HF model id")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--select_k", type=int, default=4, help="How many pred1==gold samples to pick")
    ap.add_argument("--max_scan", type=int, default=5000, help="Scan at most N samples to find select_k matches")
    ap.add_argument("--selected_out", default="selected_pred1_eq_gold.jsonl")
    ap.add_argument("--analysis_out", default="selected_conflict_analysis.jsonl")

    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--enable_thinking", action="store_true",
                help="Enable Qwen3 thinking output (default off)")
    args = ap.parse_args()

    random.seed(args.seed)

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # 关键：device_map="auto" 时不要再 model.to(...)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model.eval()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    keys = list(data.keys())
    random.shuffle(keys)

    # -----------------------
    # Phase 1: 选出 pred1==gold 的 4 条
    # -----------------------
    selected = []
    scan_keys = keys[: min(len(keys), args.max_scan)]

    pbar = tqdm(scan_keys, desc=f"Scanning for {args.select_k} correct baselines", unit="sample")
    for k in pbar:
        item = data[k]
        context = build_context(item)
        question = build_question(item)
        gold = build_gold(item)
        #import pdb;pdb.set_trace()
        if gold not in LABELS or not question:
            continue

        p1 = prompt_baseline(context, question)
        raw1 = generate_one(model, tokenizer, p1, max_new_tokens=args.max_new_tokens, temperature=args.temperature,enable_thinking=args.enable_thinking)
        pred1 = normalize_label(raw1)

        if pred1 == gold:
            selected.append({
                "key": k,
                "gold": gold,
                "pred1": pred1,
                "raw1": raw1,
                "prompt1": p1,
                "question": question,
                "context": context,
            })
            pbar.set_postfix(found=len(selected))

        if len(selected) >= args.select_k:
            break

    if len(selected) < args.select_k:
        raise RuntimeError(
            f"Only found {len(selected)} samples with pred1==gold within max_scan={args.max_scan}. "
            f"Increase --max_scan."
        )

    # 保存选中的 4 条
    with open(args.selected_out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -----------------------
    # Phase 2: 只用这 4 条做 conflict 分析
    # -----------------------
    analysis = []
    pbar2 = tqdm(selected, desc="Analyzing conflict on selected samples", unit="sample")
    for r in pbar2:
        context = r["context"]
        question = r["question"]
        gold = r["gold"]
        pred1 = r["pred1"]

        flipped = FLIP.get(pred1, "maybe")
        p2 = prompt_conflict(context, question, flipped)
        raw2 = generate_one(model, tokenizer, p2, max_new_tokens=args.max_new_tokens, temperature=args.temperature,enable_thinking=args.enable_thinking)
        pred2 = normalize_label(raw2)

        analysis.append({
            "key": r["key"],
            "gold": gold,
            "pred1": pred1,
            "pred2": pred2,
            "flipped_demo": flipped,
            "raw1": r["raw1"],
            "raw2": raw2,
            "prompt1": r["prompt1"],
            "prompt2": p2,
        })

    # 保存分析结果
    with open(args.analysis_out, "w", encoding="utf-8") as f:
        for r in analysis:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 打印这 4 条的统计（只对 4 条）
    golds = [r["gold"] for r in analysis]
    preds1 = [r["pred1"] for r in analysis]
    preds2 = [r["pred2"] for r in analysis]
    flip_rate = sum((p1 in LABELS and p2 in LABELS and p1 != p2) for p1, p2 in zip(preds1, preds2)) / len(analysis)
    acc1 = acc(preds1, golds)
    acc2 = acc(preds2, golds)
    bad_flip = sum((g in LABELS and p1 == g and p2 in LABELS and p2 != g) for g, p1, p2 in zip(golds, preds1, preds2)) / len(analysis)

    print({
        "selected_k": len(analysis),
        "flip_rate": flip_rate,
        "acc1": acc1,
        "acc2": acc2,
        "bad_flip_rate": bad_flip,
        "selected_out": args.selected_out,
        "analysis_out": args.analysis_out,
    })

if __name__ == "__main__":
    main()
