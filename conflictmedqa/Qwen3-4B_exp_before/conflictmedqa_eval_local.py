import argparse
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

QUESTION = "Does the recommendation align with current clinical guidelines?"
SYSTEM = (
    "You are a medical QA verifier. "
    "Answer ONLY with 'Yes' or 'No'. No other words."
)

def build_prompt(text: str) -> str:
    return f"{text}\n\nQuestion: {QUESTION}\nAnswer:"

def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    # Prefer chat template if available
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        # Qwen3 supports enable_thinking; other models may ignore
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # fallback if tokenizer doesn't accept enable_thinking
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{SYSTEM}\n\n{user_prompt}"

def parse_yes_no(raw: str) -> str:
    """
    Robust parse:
    - drop <think> block if present (even if unclosed)
    - search for last yes/no in the remaining text
    """
    if raw is None:
        return "unknown"
    t = raw.strip()
    # remove think blocks (closed or unclosed)
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = t.lower()

    matches = re.findall(r"\b(yes|no)\b", t)
    if not matches:
        return "unknown"
    return matches[-1]  # take the last occurrence

@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, enable_thinking: bool, max_new_tokens: int, temperature: float, top_p: float):
    text = format_chat(tokenizer, prompt, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt")

    # device_map="auto" => put inputs on first param device
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

import torch.nn.functional as F

@torch.no_grad()
def logprob_continuation(model, tokenizer, prompt_text: str, continuation: str, enable_thinking: bool) -> float:
    """
    Compute log P(continuation | prompt_text) by teacher forcing, token-by-token.
    """
    full_prompt = format_chat(tokenizer, prompt_text, enable_thinking=enable_thinking)
    enc = tokenizer(full_prompt, return_tensors="pt")

    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if len(cont_ids) == 0:
        return float("-inf")

    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))

    total = 0.0
    cur_ids = input_ids
    cur_attn = attn

    for tid in cont_ids:
        out = model(input_ids=cur_ids, attention_mask=cur_attn)
        logits = out.logits[:, -1, :]  # next-token logits
        lp = F.log_softmax(logits, dim=-1)[0, tid].item()
        total += lp

        tid_t = torch.tensor([[tid]], device=cur_ids.device)
        cur_ids = torch.cat([cur_ids, tid_t], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(tid_t)], dim=1)

    return total

@torch.no_grad()
def pick_yes_no_by_logit(model, tokenizer, prompt_text: str, enable_thinking: bool):
    """
    Pick yes/no by comparing log-prob of continuations. Returns:
    (pred, yes_lp, no_lp, delta=yes_lp-no_lp)
    """
    # Try both with and without leading space; take the better-scoring variant per label
    yes_lp = max(
        logprob_continuation(model, tokenizer, prompt_text, " Yes", enable_thinking),
        logprob_continuation(model, tokenizer, prompt_text, "Yes", enable_thinking),
    )
    no_lp = max(
        logprob_continuation(model, tokenizer, prompt_text, " No", enable_thinking),
        logprob_continuation(model, tokenizer, prompt_text, "No", enable_thinking),
    )

    pred = "yes" if yes_lp >= no_lp else "no"
    return pred, yes_lp, no_lp, (yes_lp - no_lp)

def label_to_gold(label: str) -> str:
    # dataset label: "correct" -> Yes, "wrong" -> No
    if label == "correct":
        return "yes"
    if label == "wrong":
        return "no"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path to Qwen3-4B (or HF id)")
    ap.add_argument("--config", default="gpt4o", help="DriftMed config name, e.g., gpt4o or llama70b")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise evaluate first N samples")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking output (default off)")
    ap.add_argument("--out", default="conflictmedqa_preds.jsonl")
    ap.add_argument("--local_file", default="", help="Path to local DriftMed jsonl (ConflictMed.jsonl or ConflictMed_v2.jsonl). If set, ignore --config remote loading.")
    ap.add_argument("--mode", choices=["logit", "generate"], default="logit",
                help="logit: pick Yes/No by log-prob; generate: free-generate then parse.")

    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    model.eval()

    if args.local_file:
        ds = load_dataset("json", data_files={"train": args.local_file}, split="train")
    else:
        ds = load_dataset("RDBH/DriftMed", args.config)[args.split]

    # stratified sampling: take half correct and half wrong when --limit is set
    if args.limit and args.limit > 0:
        correct_idx = [i for i in range(len(ds)) if ds[i]["label"] == "correct"]
        wrong_idx   = [i for i in range(len(ds)) if ds[i]["label"] == "wrong"]

        half = args.limit // 2
        take_c = min(half, len(correct_idx))
        take_w = min(args.limit - take_c, len(wrong_idx))

        # deterministic
        correct_idx = correct_idx[:take_c]
        wrong_idx = wrong_idx[:take_w]

        ds = ds.select(correct_idx + wrong_idx)

    

    # Collect predictions
    rows = []
    pbar = tqdm(range(len(ds)), desc="Evaluating DriftMed", unit="sample")
    for i in pbar:
        # if i >=10:
        #     break
        ex = ds[i]
        text = ex["text"]
        gold = label_to_gold(ex["label"])

        prompt = build_prompt(text)

        if args.mode == "logit":
            pred, yes_lp, no_lp, delta = pick_yes_no_by_logit(
                model, tokenizer, prompt, enable_thinking=args.enable_thinking
            )
            raw = ""  # logit 模式不需要 raw
        else:
            raw = generate_one(
                model, tokenizer, prompt,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            pred = parse_yes_no(raw)
            yes_lp = no_lp = delta = None

        rows.append({
            "id": ex["id"],
            "label": ex["label"],
            "gold": gold,
            "pred": pred,
            "bias_type": ex.get("bias_type"),
            "change_category": ex.get("change_category"),
            "disease": ex.get("disease"),
            "raw": raw,
            "prompt": prompt,
            "yes_lp": yes_lp,
            "no_lp": no_lp,
            "delta": delta,  # yes_lp - no_lp，越大越偏向 Yes
        })

    # Save jsonl
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Metrics
    def acc(sub):
        valid = [r for r in sub if r["gold"] in ("yes","no") and r["pred"] in ("yes","no")]
        return sum(r["pred"] == r["gold"] for r in valid) / max(1, len(valid)), len(valid)

    overall_acc, overall_n = acc(rows)
    correct_rows = [r for r in rows if r["label"] == "correct"]
    wrong_rows = [r for r in rows if r["label"] == "wrong"]
    acc_correct, n_correct = acc(correct_rows)
    acc_wrong, n_wrong = acc(wrong_rows)

    # Pair metrics (requires full dataset pairing rule: wrong_id = correct_id + 2145)
    # We'll compute on whatever ids exist in this run.
    by_id = {r["id"]: r for r in rows}
    pair_counts = Counter()
    paired = 0
    for r in correct_rows:
        cid = r["id"]
        wid = cid + 2145
        if wid in by_id:
            paired += 1
            c_pred = by_id[cid]["pred"]
            w_pred = by_id[wid]["pred"]
            pair_counts[(c_pred, w_pred)] += 1

    def frac(key):
        return pair_counts[key] / max(1, paired)

    report = {
        "saved": args.out,
        "n_total": len(rows),
        "acc_overall": overall_acc,
        "acc_overall_n": overall_n,
        "acc_correct(Yes_on_correct)": acc_correct,
        "acc_correct_n": n_correct,
        "acc_wrong(No_on_wrong)": acc_wrong,
        "acc_wrong_n": n_wrong,
        "paired_count": paired,
        "pair_(yes,no)_ideal": frac(("yes","no")),
        "pair_(yes,yes)_overly_context_following": frac(("yes","yes")),
        "pair_(no,no)_overly_rejecting": frac(("no","no")),
        "pair_(no,yes)_inverted": frac(("no","yes")),
        "pair_counts_raw": {str(k): v for k, v in pair_counts.items()},
    }
    print(report)

if __name__ == "__main__":
    main()
