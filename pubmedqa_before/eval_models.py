# -*- coding: utf-8 -*-
"""
Evaluate causal LMs on PubMedQA-style datasets under three conditions:

- NC (No Context): answer using internal knowledge only.
- CC (Clean Context): answer using ONLY the provided abstract/context.
- IC (Intervened/Misleading Context): SCHEME-1 (flip the context itself).
    Instead of appending a "misleading summary", we rewrite the RESULT/CONCLUSION
    portion of the context so that it semantically supports the *opposite* label.

This version additionally:
1) Filters out samples whose gold label is "maybe" (keeps only yes/no).
2) Supports two evaluation modes:
   - generate: use model.generate() and parse yes/no/unknown from output.
   - logit: score candidate labels by log-probability and pick the best.
3) Adds an --enable_thinking flag that is passed into tokenizer.apply_chat_template
   when supported (Qwen3-style). If unsupported, it is silently ignored (fallback).

Notes:
- Even if enable_thinking=True, parse_answer() strips <think>...</think> blocks.
- In logit mode, we score candidates with leading spaces: " yes", " no", " unknown"
  against the prompt that ends with "Answer:".

"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["yes", "no", "maybe", "unknown"]

SYSTEM_PROMPT = "You are a helpful medical QA assistant."


def strip_think(text: str) -> str:
    """Remove common thinking/reasoning tags to avoid parse pollution."""
    t = text or ""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<reasoning>.*?</reasoning>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    # Also handle unclosed tags occasionally produced
    t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<reasoning>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    return t


def parse_answer(text: str) -> str:
    """Extract a single label token from model output."""
    t = strip_think((text or "").strip().lower())
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", t):
            return lab
    first = re.split(r"\s+|[?.,!?:;\n]", t)[0]
    return first if first in LABELS else "unknown"


def wrong_label(gold: str) -> str:
    """Return a contradictory label."""
    gold = (gold or "").lower().strip()
    if gold == "yes":
        return "no"
    if gold == "no":
        return "yes"
    # 'maybe' is filtered out in this script, but keep a fallback.
    if gold == "maybe":
        return "yes"
    return "no"


def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    """
    Format as chat if tokenizer supports chat templates.
    Mirrors the pattern used in conflictmedqa_eval_local.py:
      tokenizer.apply_chat_template(..., enable_thinking=enable_thinking) if supported,
      else fall back to normal apply_chat_template or raw prompt.
    """
    # Some tokenizers expose `chat_template`, some only support apply_chat_template.
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
            # Template doesn't accept enable_thinking
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    # No template support: just return the raw prompt.
    return user_prompt


def make_prompt(question: str, context: Optional[str], mode: str) -> str:
    """
    mode:
      - "NC": no context, internal knowledge only
      - "CC"/"IC": must use provided context only
    """
    if mode == "NC":
        return (
            "Answer the question using ONLY your internal knowledge.\n"
            "If you cannot answer with high confidence, answer: unknown.\n"
            "Output one token only from: yes / no / unknown.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the question using ONLY the provided context.\n"
        "Even if evidence is weak, choose the most supported answer. Output exactly one word: yes or no.\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    #"Output one token only from: yes / no / unknown.\n\n"


@torch.inference_mode()
def generate_one(model, tok, prompt: str, max_new_tokens: int, enable_thinking: bool) -> str:
    formatted = format_chat(tok, prompt, enable_thinking=enable_thinking)
    inputs = tok(formatted, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return gen.strip()


@torch.inference_mode()
def logprob_continuation(model, tok, prompt: str, continuation: str, enable_thinking: bool) -> float:
    """
    Compute log P(continuation | prompt) using the model's token-level log-probs.
    """
    formatted = format_chat(tok, prompt, enable_thinking=enable_thinking)

    enc_prompt = tok(formatted, add_special_tokens=False, return_tensors="pt")
    enc_full = tok(formatted + continuation, add_special_tokens=False, return_tensors="pt")

    input_ids_prompt = enc_prompt["input_ids"][0]
    input_ids_full = enc_full["input_ids"][0]

    # Continuation token ids are the tail of full after prompt.
    cont_ids = input_ids_full[len(input_ids_prompt) :]
    if cont_ids.numel() == 0:
        return float("-inf")

    input_ids_full = input_ids_full.unsqueeze(0).to(model.device)

    outputs = model(input_ids_full)
    logits = outputs.logits  # [1, seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # For each continuation token at position p, probability comes from log_probs at p-1.
    start = len(input_ids_prompt)
    total = 0.0
    for i, tok_id in enumerate(cont_ids):
        pos = start + i
        if pos - 1 < 0 or pos - 1 >= log_probs.shape[1]:
            continue
        total += float(log_probs[0, pos - 1, int(tok_id)])
    return total


@torch.inference_mode()
def pick_label_by_logit(model, tok, prompt: str, enable_thinking: bool) -> Tuple[str, Dict[str, float]]:
    """
    Score candidates by logprob and pick the best.
    Candidates use leading spaces to match typical tokenization after "Answer:".
    """
    candidates = {"yes": " yes", "no": " no", "unknown": " unknown"}
    scores = {lab: logprob_continuation(model, tok, prompt, cont, enable_thinking) for lab, cont in candidates.items()}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores


def load_local_dataset(path: str, fmt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load a local dataset from:
      - JSONL (list of dicts, one per line)
      - JSON:
         * list[dict]
         * dict[pmid -> dict]  (will be converted to list with injected 'id')
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if fmt == "json":
            obj = json.load(f)
        elif fmt == "jsonl":
            rows = []
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
            return rows
        else:
            # auto-detect json vs jsonl
            if first == "{":
                obj = json.load(f)
            else:
                rows = []
                for line in f:
                    s = line.strip()
                    if s:
                        rows.append(json.loads(s))
                return rows

        if isinstance(obj, dict):
            rows = []
            for k, ex in obj.items():
                if isinstance(ex, dict):
                    rows.append({"id": k, **ex})
                else:
                    rows.append({"id": k, "value": ex})
            return rows
        if isinstance(obj, list):
            return obj

        raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")


def get_field(ex: Dict[str, Any], key: str) -> Any:
    cur: Any = ex
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_context(ctx: Any) -> str:
    """Accept string | list[str] | dict{'contexts': [...]} | other dict -> JSON string."""
    if ctx is None:
        return ""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, list):
        return "\n".join([str(x) for x in ctx])
    if isinstance(ctx, dict):
        if "contexts" in ctx and isinstance(ctx["contexts"], list):
            return "\n".join([str(x) for x in ctx["contexts"]])
        return json.dumps(ctx, ensure_ascii=False)
    return str(ctx)


def load_model(model_name: str, device_map: str, dtype: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    torch_dtype = None
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


# ---------- Scheme-1: flip context itself ----------

def _split_sentences(text: str) -> List[str]:
    # Very lightweight split (keeps it deterministic)
    sents = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in sents if s.strip()]


def _join_sentences(sents: List[str]) -> str:
    return " ".join([s.strip() for s in sents if s.strip()])


def _rewrite_sentence_polarity(sentence: str, target: str) -> str:
    """
    Rewrite a sentence to support target in {yes,no} using heuristic polarity flips.
    This is not perfect English, but yields a robust semantic cue for conflict tests.
    """
    original = sentence
    out = sentence

    target = (target or "").strip().lower()
    assert target in ("yes", "no")

    # If target is "no", we want "no effect / not associated / no significant difference".
    # If target is "yes", we want "effect / associated / significant difference".
    if target == "no":
        flips = [
            (r"\bsignificant(ly)?\b", "not statistically significant"),
            (r"\b(improved|improves|improvement)\b", "did not improve"),
            (r"\b(reduced|reduces|reduction)\b", "did not reduce"),
            (r"\b(increased|increases|increase)\b", "did not increase"),
            (r"\b(effective|efficacious)\b", "not effective"),
            (r"\b(associated with)\b", "not associated with"),
            (r"\b(higher than)\b", "not higher than"),
            (r"\b(lower than)\b", "not lower than"),
        ]
        for pat, rep in flips:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)

        # Common abstract phrase
        if re.search(r"\b(no significant difference|not statistically significant)\b", out, re.IGNORECASE) is None:
            # If nothing suggests "no", lightly add a cue.
            out = out.rstrip(".") + ". No meaningful difference was observed."
    else:
        flips = [
            (r"\b(no significant difference)\b", "a significant difference"),
            (r"\b(not statistically significant)\b", "statistically significant"),
            (r"\b(did not improve)\b", "improved"),
            (r"\b(did not reduce)\b", "reduced"),
            (r"\b(did not increase)\b", "increased"),
            (r"\b(not effective)\b", "effective"),
            (r"\b(not associated with)\b", "associated with"),
            (r"\b(not higher than)\b", "higher than"),
            (r"\b(not lower than)\b", "lower than"),
        ]
        for pat, rep in flips:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)

        if re.search(r"\b(significant difference|statistically significant|associated with|effective)\b", out, re.IGNORECASE) is None:
            out = out.rstrip(".") + ". Overall, a meaningful difference was observed."

    # Cleanup obvious double-negation artifacts
    out = re.sub(r"\bnot not\b", "not", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdid not not\b", "did not", out, flags=re.IGNORECASE)
    out = re.sub(r"\bdoes not not\b", "does not", out, flags=re.IGNORECASE)

    changed = (out != original)

    # If nothing changed, append a short cue
    if not changed:
        if target == "no":
            out = out.rstrip(".") + ". Overall, no meaningful difference was observed."
        else:
            out = out.rstrip(".") + ". Overall, a meaningful difference was observed."
    return out


def flip_context_scheme1(ctx: str, target: str) -> str:
    """
    Scheme-1: flip the context itself to semantically support `target` (yes/no),
    focusing on RESULT/CONCLUSION parts.

    Heuristic:
      1) If lines contain section cues (RESULTS/CONCLUSION), rewrite those lines.
      2) Otherwise, rewrite the last 1-2 sentences (where conclusions usually are).
    """
    target = (target or "").strip().lower()
    if target not in ("yes", "no"):
        return ctx

    lines = [ln.strip() for ln in (ctx or "").splitlines()]
    if not lines:
        return ctx

    cues = re.compile(r"\b(results?|conclusions?|conclusion|findings?)\b", re.IGNORECASE)
    idxs = [i for i, ln in enumerate(lines) if cues.search(ln)]
    rewritten_any = False

    if idxs:
        for i in idxs:
            if not lines[i]:
                continue
            sents = _split_sentences(lines[i])
            if not sents:
                continue
            sents = [_rewrite_sentence_polarity(s, target) for s in sents]
            new_line = _join_sentences(sents)
            if new_line != lines[i]:
                rewritten_any = True
            lines[i] = new_line
    else:
        full = " ".join([ln for ln in lines if ln])
        sents = _split_sentences(full)
        if len(sents) >= 1:
            k = 2 if len(sents) >= 2 else 1
            for j in range(1, k + 1):
                sents[-j] = _rewrite_sentence_polarity(sents[-j], target)
            rewritten_any = True
            return _join_sentences(sents)

    if not rewritten_any:
        full = " ".join([ln for ln in lines if ln])
        sents = _split_sentences(full)
        if sents:
            k = 2 if len(sents) >= 2 else 1
            for j in range(1, k + 1):
                sents[-j] = _rewrite_sentence_polarity(sents[-j], target)
            return _join_sentences(sents)

    return "\n".join(lines).strip()


# ---------- Evaluation ----------

def eval_one_model(
    model_name: str,
    data: List[Dict[str, Any]],
    out_dir: str,
    q_field: str,
    c_field: str,
    y_field: str,
    id_field: str,
    limit: int,
    max_new_tokens: int,
    device_map: str,
    dtype: str,
    mode: str,
    enable_thinking: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sanitize(model_name)}.preds.jsonl")

    model, tok = load_model(model_name, device_map=device_map, dtype=dtype)

    n = 0
    skipped_maybe = 0
    acc_cc = 0
    acc_ic = 0
    OR = 0
    VM = 0
    fix_by_cc = 0
    misled_by_ic = 0

    with open(out_path, "w", encoding="utf-8") as wf:
        total_cap = limit if (limit and limit > 0) else len(data)
        for i, ex in enumerate(tqdm(data, desc=f"Eval {model_name}", total=total_cap)):
            if limit > 0 and n >= limit:
                break

            q = get_field(ex, q_field)
            ctx_raw = get_field(ex, c_field)
            gold = get_field(ex, y_field)
            sid = get_field(ex, id_field) if id_field else ex.get("id", i)

            if q is None or gold is None:
                continue

            gold = str(gold).strip().lower()

            # Filter: keep only yes/no
            if gold == "maybe":
                skipped_maybe += 1
                continue
            if gold not in ("yes", "no"):
                continue

            ctx_cc = normalize_context(ctx_raw)

            # IC: flip the context itself toward the opposite label
            wl = wrong_label(gold)
            ctx_ic = flip_context_scheme1(ctx_cc, wl)

            # Build prompts
            nc_prompt = make_prompt(str(q), None, "NC")
            cc_prompt = make_prompt(str(q), ctx_cc, "CC")
            ic_prompt = make_prompt(str(q), ctx_ic, "IC")

            # Run either in generate or logit mode
            if mode == "generate":
                nc_out = generate_one(model, tok, nc_prompt, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
                cc_out = generate_one(model, tok, cc_prompt, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
                ic_out = generate_one(model, tok, ic_prompt, max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)

                nc = parse_answer(nc_out)
                cc = parse_answer(cc_out)
                ic = parse_answer(ic_out)

                extra = {"nc_raw": nc_out, "cc_raw": cc_out, "ic_raw": ic_out}
            else:
                nc, nc_scores = pick_label_by_logit(model, tok, nc_prompt, enable_thinking=enable_thinking)
                cc, cc_scores = pick_label_by_logit(model, tok, cc_prompt, enable_thinking=enable_thinking)
                ic, ic_scores = pick_label_by_logit(model, tok, ic_prompt, enable_thinking=enable_thinking)

                # Keep compact scores for debugging
                extra = {
                    "nc_scores": nc_scores,
                    "cc_scores": cc_scores,
                    "ic_scores": ic_scores,
                }

            n += 1
            nc_ok = (nc == gold)
            cc_ok = (cc == gold)
            ic_ok = (ic == gold)

            acc_cc += int(cc_ok)
            acc_ic += int(ic_ok)

            if (not nc_ok) and (not cc_ok):
                OR += 1
            if nc_ok and (not ic_ok):
                VM += 1
            if (not nc_ok) and cc_ok:
                fix_by_cc += 1
            if nc_ok and (not ic_ok):
                misled_by_ic += 1

            rec = {
                "id": sid,
                "gold": gold,
                "question": q,
                "mode": mode,
                "enable_thinking": bool(enable_thinking),
                "nc": nc,
                "cc": cc,
                "ic": ic,
                "ctx_cc": ctx_cc,
                "ctx_ic": ctx_ic,
                "flags": {
                    "nc_ok": nc_ok,
                    "cc_ok": cc_ok,
                    "ic_ok": ic_ok,
                    "OR": (not nc_ok) and (not cc_ok),
                    "VM": nc_ok and (not ic_ok),
                    "fix_by_cc": (not nc_ok) and cc_ok,
                    "misled_by_ic": nc_ok and (not ic_ok),
                },
            }
            rec.update(extra)
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def pct(x: int) -> float:
        return 100.0 * x / n if n else 0.0

    summary = {
        "model": model_name,
        "N": n,
        "skipped_maybe": skipped_maybe,
        "mode": mode,
        "enable_thinking": bool(enable_thinking),
        "Acc_CC": pct(acc_cc),
        "Acc_IC": pct(acc_ic),
        "OR": pct(OR),
        "VM": pct(VM),
        "NC_to_CC_fix_rate": pct(fix_by_cc),
        "NC_to_IC_misled_rate": pct(misled_by_ic),
        "pred_file": out_path,
    }

    with open(os.path.join(out_dir, f"{sanitize(model_name)}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Local dataset file path")
    ap.add_argument("--fmt", default="jsonl", choices=["jsonl", "json", "csv"], help="csv is not implemented; keep json/jsonl.")
    ap.add_argument("--out_dir", default="results_pubmedqa_conflict")

    # field mapping
    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="", help="Optional. e.g. id or PMID")

    # models
    ap.add_argument("--models", nargs="+", required=True, help="HF model names/paths, space-separated")

    # evaluation controls
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=8, help="Only used in --mode generate")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--mode", default="generate", choices=["generate", "logit"], help="Evaluation mode")
    ap.add_argument("--enable_thinking", action="store_true",
                    help="Pass enable_thinking=True into chat template when supported (Qwen3-style).")

    args = ap.parse_args()

    data = load_local_dataset(args.data, args.fmt)

    all_summaries = []
    for m in args.models:
        summary = eval_one_model(
            model_name=m,
            data=data,
            out_dir=args.out_dir,
            q_field=args.q_field,
            c_field=args.c_field,
            y_field=args.y_field,
            id_field=args.id_field if args.id_field else "",
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            dtype=args.dtype,
            mode=args.mode,
            enable_thinking=args.enable_thinking,
        )
        all_summaries.append(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ALL_SUMMARIES.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for s in all_summaries:
        print(
            f"{s['model']}\tN={s['N']}\t(skipped_maybe={s['skipped_maybe']})"
            f"\tmode={s['mode']}\tenable_thinking={s['enable_thinking']}"
            f"\tAcc_CC={s['Acc_CC']:.2f}\tAcc_IC={s['Acc_IC']:.2f}"
            f"\tOR={s['OR']:.2f}\tVM={s['VM']:.2f}\tFix={s['NC_to_CC_fix_rate']:.2f}\tMisled={s['NC_to_IC_misled_rate']:.2f}"
        )
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
