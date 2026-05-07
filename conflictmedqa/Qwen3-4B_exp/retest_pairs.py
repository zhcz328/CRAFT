#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Keep exactly the same verifier setup as your DriftMed script
QUESTION = "Does the recommendation align with current clinical guidelines?"
SYSTEM = (
    "You are a medical QA verifier. "
    "Answer ONLY with 'Yes' or 'No'. No other words."
)

YES, NO, UNK = "yes", "no", "unknown"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    """
    Match the method in your DriftMed script:
    - Use chat template if available
    - Provide SYSTEM as system message
    - Put user_prompt as user message
    - Try enable_thinking for Qwen3
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM},
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
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{SYSTEM}\n\n{user_prompt}"


def parse_yes_no(raw: str) -> str:
    """
    Robust parse (same idea as your DriftMed script):
    - drop <think> blocks (closed or unclosed)
    - take last yes/no
    """
    if raw is None:
        return UNK
    t = raw.strip()
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = t.lower()

    matches = re.findall(r"\b(yes|no)\b", t)
    return matches[-1] if matches else UNK


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    user_prompt: str,
    enable_thinking: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Exactly like your DriftMed version:
    - format_chat()
    - tokenize
    - move inputs to first param device
    - model.generate()
    - decode ONLY the generated tail (exclude prompt)
    """
    text = format_chat(tokenizer, user_prompt, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt")

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
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return gen.strip()


@torch.no_grad()
def logprob_continuation(
    model,
    tokenizer,
    user_prompt: str,
    continuation: str,
    enable_thinking: bool,
) -> float:
    """
    Same as your DriftMed version: token-by-token teacher forcing.
    """
    full_prompt = format_chat(tokenizer, user_prompt, enable_thinking=enable_thinking)
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
        logits = out.logits[:, -1, :]
        lp = F.log_softmax(logits, dim=-1)[0, tid].item()
        total += lp

        tid_t = torch.tensor([[tid]], device=cur_ids.device)
        cur_ids = torch.cat([cur_ids, tid_t], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(tid_t)], dim=1)

    return float(total)


@torch.no_grad()
def pick_yes_no_by_logit(model, tokenizer, user_prompt: str, enable_thinking: bool):
    """
    Same as your DriftMed version:
    Compare continuation logprob, try with/without leading space.
    Return (pred, yes_lp, no_lp, delta).
    """
    yes_lp = max(
        logprob_continuation(model, tokenizer, user_prompt, " Yes", enable_thinking),
        logprob_continuation(model, tokenizer, user_prompt, "Yes", enable_thinking),
    )
    no_lp = max(
        logprob_continuation(model, tokenizer, user_prompt, " No", enable_thinking),
        logprob_continuation(model, tokenizer, user_prompt, "No", enable_thinking),
    )
    pred = YES if yes_lp >= no_lp else NO
    return pred, yes_lp, no_lp, (yes_lp - no_lp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="kept_pairs_a12_b12.jsonl (each line is one pair)")
    ap.add_argument("--model", required=True, help="Local model path or HF id")
    ap.add_argument("--mode", choices=["logit", "generate"], default="logit",
                    help="logit: compare yes/no logprob; generate: free-generate then parse")
    ap.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking (default off)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")

    # generate-mode controls
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--out", default="retest_out.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    pairs = read_jsonl(args.pairs)

    ok_pairs = 0
    total_pairs = 0
    failed = []

    with open(args.out, "w", encoding="utf-8") as f:
        pbar = tqdm(pairs, desc=f"Retest ({args.mode})", unit="pair")
        for rec in pbar:
            pid = rec.get("pair_id", None)
            c = rec["correct"]
            w = rec["wrong"]

            # kept_pairs stores plain user prompts
            c_user_prompt = c.get("prompt", "")
            w_user_prompt = w.get("prompt", "")

            if args.mode == "logit":
                c_pred, c_yes, c_no, c_delta = pick_yes_no_by_logit(
                    model, tokenizer, c_user_prompt, enable_thinking=args.enable_thinking
                )
                w_pred, w_yes, w_no, w_delta = pick_yes_no_by_logit(
                    model, tokenizer, w_user_prompt, enable_thinking=args.enable_thinking
                )
                c_raw = w_raw = ""
            else:
                c_raw = generate_one(
                    model, tokenizer, c_user_prompt,
                    enable_thinking=args.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                w_raw = generate_one(
                    model, tokenizer, w_user_prompt,
                    enable_thinking=args.enable_thinking,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                c_pred = parse_yes_no(c_raw)
                w_pred = parse_yes_no(w_raw)
                c_yes = c_no = c_delta = None
                w_yes = w_no = w_delta = None

            # expected: correct -> yes, wrong -> no
            pair_ok = (c_pred == YES) and (w_pred == NO)

            total_pairs += 1
            ok_pairs += (1 if pair_ok else 0)
            if not pair_ok:
                failed.append((pid, c_pred, w_pred))

            pbar.set_postfix_str(f"acc={ok_pairs}/{total_pairs}={ok_pairs/total_pairs:.3f}")

            out_rec = {
                "pair_id": pid,
                "mode": args.mode,
                "thinking": args.enable_thinking,
                "pair_ok": pair_ok,
                "correct": {"pred": c_pred, "yes_lp": c_yes, "no_lp": c_no, "delta": c_delta, "raw": c_raw},
                "wrong": {"pred": w_pred, "yes_lp": w_yes, "no_lp": w_no, "delta": w_delta, "raw": w_raw},
            }
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"pairs={total_pairs}, ok_pairs={ok_pairs}, acc_pair={ok_pairs/total_pairs:.3f}, saved={args.out}")
    if failed:
        print("first 10 failed pairs (pair_id, correct_pred, wrong_pred):")
        for x in failed[:10]:
            print(" ", x)


if __name__ == "__main__":
    main()
