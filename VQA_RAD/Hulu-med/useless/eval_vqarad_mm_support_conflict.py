# -*- coding: utf-8 -*-
"""
VQA-RAD multimodal SUPPORT vs CONFLICT evaluation (NO 'unknown') - FINAL (image tokens aligned)

- Model: Qwen3-VL-8B-Instruct (local)
- Data: vqa_rad_*_closed.csv (CLOSED; may include non-yes/no answers)
- For each sample:
    SUPPORT  : evidence uses GOLD answer
    CONFLICT : evidence uses CONFLICT answer (Priority-A distractor; never 'unknown')
- Prediction: score ONLY {gold, conflict} by continuation logprob, then pick argmax
- No 'unknown' anywhere.

Key point:
  Use processor.apply_chat_template(messages, tokenize=True, return_dict=True, ...)
  with multimodal messages containing {"type":"image"} so that image placeholder tokens are inserted.
  Otherwise you'll hit: tokens: 0, features: XXX.

Outputs:
  out_dir/.../preds.jsonl
  out_dir/.../summary.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM


SYSTEM_PROMPT = "You are a helpful medical QA assistant."

BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)

EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)


def load_mm_model(model_name, device_map=None, torch_dtype=None, trust_remote_code=True, attn_implementation=None, quantization_config=None):
    load_errs = []
    for cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
        try:
            kwargs = {"trust_remote_code": trust_remote_code}
            if device_map is not None:
                kwargs["device_map"] = device_map
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            if attn_implementation is not None:
                kwargs["attn_implementation"] = attn_implementation
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
            return cls.from_pretrained(model_name, **kwargs)
        except Exception as e:
            load_errs.append(f"{cls.__name__}: {repr(e)}")
    raise RuntimeError("Failed to load model. " + " | ".join(load_errs))


def sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def norm_text(x: str) -> str:
    return str(x).strip().lower()


# -------------------------
# Priority-A conflict answer selection (never 'unknown')
# -------------------------
def build_answer_pools(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = "other"

    df["ans"] = df["answer"].astype(str).map(norm_text)
    df["cat"] = df["category"].astype(str).map(norm_text)

    by_cat = defaultdict(list)
    global_pool = []
    for a, c in zip(df["ans"].tolist(), df["cat"].tolist()):
        by_cat[c].append(a)
        global_pool.append(a)

    by_cat_sorted: Dict[str, List[str]] = {}
    for c, lst in by_cat.items():
        cnt = Counter(lst)
        by_cat_sorted[c] = [x for x, _ in cnt.most_common()]

    global_sorted = [x for x, _ in Counter(global_pool).most_common()]
    return by_cat_sorted, global_sorted


def choose_conflict_answer_priority_A(
    gold: str,
    cat: str,
    by_cat_sorted: Dict[str, List[str]],
    global_sorted: List[str],
) -> str:
    g = norm_text(gold)
    c = norm_text(cat)

    # 1) yes/no 鐩存帴缈昏浆
    if g == "yes":
        return "no"
    if g == "no":
        return "yes"

    # 2) 闈?yes/no锛氬彧鍦ㄥ悓 category 閲屾壘涓€涓渶甯歌涓斾笉鍚岀殑绛旀
    for cand in by_cat_sorted.get(c, []):
        cand_n = norm_text(cand)
        if cand_n and cand_n != g and cand_n != "unknown":
            return cand_n

    # 3) 鍚?category 鎵句笉鍒帮細鐩存帴 unknown
    return "unknown"

# -------------------------
# Prompt building
# -------------------------
def make_base_prompt(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def inject_evidence(base_prompt: str, evidence_block: str, position: str) -> str:
    if position == "prefix":
        return evidence_block + "\n" + base_prompt

    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    raise ValueError(f"Unknown position: {position}")


def make_prompt_with_evidence(question: str, evidence_ans: str, position: str) -> str:
    base = make_base_prompt(question)
    evidence = EVIDENCE_TMPL.format(ans=evidence_ans)
    return inject_evidence(base, evidence, position)
def make_prompt_no_evidence(question: str) -> str:
    # 涓嶅惈 EVIDENCE block
    return BASE_RULE + f"Question: {question}\nAnswer:"

def build_messages_multimodal(image: Image.Image, user_prompt: str):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


# -------------------------
# Logprob scoring (aligned with Qwen3-VL HF docs)
# -------------------------
# @torch.inference_mode()
# def logprob_continuation_mm(model, processor, messages, continuation: str) -> float:
#     """
#     log p(continuation | messages) where messages include an image block.

#     We:
#       - processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, return_dict=True)
#         => returns input_ids + vision tensors + placeholder image tokens (CRITICAL)
#       - append continuation ids to input_ids and compute token logprobs
#     """
#     tok = processor.tokenizer

#     base_inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt",
#     )
#     # some versions include token_type_ids; model doesn't need it
#     base_inputs.pop("token_type_ids", None)
#     base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}

#     cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
#     if cont_ids.numel() == 0:
#         return float("-inf")

#     input_ids_prompt = base_inputs["input_ids"]
#     attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))

#     input_ids_full = torch.cat([input_ids_prompt, cont_ids], dim=1)
#     attn_full = torch.cat([attn_prompt, torch.ones_like(cont_ids)], dim=1)

#     full_inputs = dict(base_inputs)
#     full_inputs["input_ids"] = input_ids_full
#     full_inputs["attention_mask"] = attn_full

#     outputs = model(**full_inputs)
#     logits = outputs.logits
#     log_probs = torch.log_softmax(logits, dim=-1)

#     start = input_ids_prompt.shape[1]
#     total = 0.0
#     for i in range(cont_ids.shape[1]):
#         pos = start + i
#         tok_id = int(cont_ids[0, i])
#         total += float(log_probs[0, pos - 1, tok_id])
#     return total


# @torch.inference_mode()
# def score_two_options_mm(model, processor, messages, opt_a: str, opt_b: str) -> Dict[str, float]:
#     def score(opt: str) -> float:
#         opt = str(opt)
#         return max(
#             logprob_continuation_mm(model, processor, messages, " " + opt),
#             logprob_continuation_mm(model, processor, messages, opt),
#         )
#     return {opt_a: score(opt_a), opt_b: score(opt_b)}


def argmax_two(scores: Dict[str, float]) -> Tuple[str, float]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return items[0][0], float(items[0][1])
# -------------------------
# Logprob scoring (FAST: prompt once + KV cache)
# -------------------------
@torch.inference_mode()
def build_prompt_cache_mm(model, processor, messages):
    base_inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    base_inputs.pop("token_type_ids", None)
    base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}

    # forward prompt once
    out = model(**base_inputs, use_cache=True)

    input_ids_prompt = base_inputs["input_ids"]
    attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_len = int(input_ids_prompt.shape[1])

    # 鍏抽敭锛歱rompt 鏈€鍚庝竴涓綅缃殑 logits锛岀敤鏉ラ娴?continuation 鐨勭涓€涓?token
    # logits shape: [1, prompt_len, vocab]
    prompt_last_logits = out.logits[:, -1, :].detach()  # [1, vocab]

    return {
        "prompt_len": prompt_len,
        "attn_prompt": attn_prompt,
        "past_key_values": out.past_key_values,
        "prompt_last_logits": prompt_last_logits,
    }


@torch.inference_mode()
def logprob_from_cache_mm(model, processor, cache_pack, continuation: str) -> float:
    tok = processor.tokenizer
    cont_ids = tok(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    if cont_ids.numel() == 0:
        return float("-inf")

    attn_prompt = cache_pack["attn_prompt"]
    past_key_values = cache_pack["past_key_values"]
    prompt_last_logits = cache_pack["prompt_last_logits"]  # [1, vocab]

    cont_len = int(cont_ids.shape[1])

    # 1) 绗竴涓?token锛氱敤 prompt_last_logits 鏉ョ畻
    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)  # [1, vocab]
    first_tok = int(cont_ids[0, 0])
    total = float(log_probs0[0, first_tok])

    if cont_len == 1:
        return total

    # 2) 鍚庣画 token锛氬杺鍏?cont_ids 鐨勫墠 cont_len-1 涓?token锛岄娴嬩笅涓€涓?token
    # 娉ㄦ剰锛氳繖閲岃緭鍏ラ暱搴︽槸 cont_len-1锛岃緭鍑?logits 涔熸槸 cont_len-1
    inp = cont_ids[:, :-1]  # [1, cont_len-1]

    # attention_mask 瑕嗙洊 prompt + inp
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)

    out = model(
        input_ids=inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
    )
    # out.logits: [1, cont_len-1, vocab]
    log_probs = torch.log_softmax(out.logits, dim=-1)

    # for token i>=1, use logits position i-1
    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total += float(log_probs[0, i - 1, tok_id])

    return total


@torch.inference_mode()
def score_two_options_mm(model, processor, messages, opt_a: str, opt_b: str):
    cache_pack = build_prompt_cache_mm(model, processor, messages)

    # Keep one formatting style for option strings for stable scoring.
    def score(opt: str) -> float:
        opt = str(opt)
        return logprob_from_cache_mm(model, processor, cache_pack, " " + opt)

    return {opt_a: score(opt_a), opt_b: score(opt_b)}

def margin_two(scores: Dict[str, float]) -> float:
    vals = sorted(scores.values(), reverse=True)
    if len(vals) < 2:
        return float("inf")
    return float(vals[0] - vals[1])


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", default="results_vqarad_mm_support_conflict_no_unknown_final")
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    if "category" not in df.columns:
        df["category"] = "other"
    if "answer_type" in df.columns:
        df["answer_type_norm"] = df["answer_type"].astype(str).str.strip().str.upper()
        df = df[df["answer_type_norm"] == "CLOSED"].copy()

    df["gold"] = df["answer"].astype(str).map(norm_text)
    df["cat"] = df["category"].astype(str).map(norm_text)

    by_cat_sorted, global_sorted = build_answer_pools(df)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = None if args.dtype == "auto" else dtype_map[args.dtype]

    from transformers import BitsAndBytesConfig

    # ... 浣犵殑 dtype_map / torch_dtype 閫昏緫鍙互淇濈暀锛屼絾寤鸿 3090 鐢?fp16
    bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    model = load_mm_model(
        args.model,
        device_map=args.device_map,          # 涓€鑸繕鏄?"auto"
        quantization_config=bnb_cfg,         # <--- 鏂板
        torch_dtype=torch.float16,           # <--- 寤鸿鍥哄畾 fp16锛?090 鏇村悎閫傦級
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    out_root = Path(args.out_dir) / sanitize(str(args.model)) / args.position
    out_root.mkdir(parents=True, exist_ok=True)
    preds_path = out_root / "preds.jsonl"
    summary_path = out_root / "summary.json"

    N = 0
    skipped_missing_img = 0

    support_correct = 0
    conflict_still_gold = 0
    conflict_follow_evidence = 0
    resist_cnt = 0
    changed_cnt = 0

    resist_given_sup_correct = 0
    changed_given_sup_correct = 0
    follow_conflict_given_sup_correct = 0

    sup_margin_sum = 0.0
    con_margin_sum = 0.0
    nc_correct = 0
    nc_margin_sum = 0.0
    
    # conditional on NC being correct
    resist_given_nc_correct = 0
    changed_given_nc_correct = 0
    follow_conflict_given_nc_correct = 0
    nc_correct_cnt = 0
    both_correct_rows = []  # save samples where NC and support are both correct
    unknown_conflict_cnt = 0
    # --- NC baseline stats (DO NOT use support as reference) ---
    resist_vs_nc_cnt = 0
    changed_vs_nc_cnt = 0

    conflict_pred_unknown_cnt = 0          # con_pred == unknown
    changed_to_unknown_vs_nc_cnt = 0       # nc_pred != unknown but con_pred == unknown
    with open(preds_path, "w", encoding="utf-8") as wf:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="MM SUPPORT/CONFLICT eval (final)"):
            if args.limit > 0 and N >= args.limit:
                break
          
            img_id = str(row["img_id"])
            question = str(row["question"])
            gold = norm_text(row["gold"])
            cat = norm_text(row["cat"])

            conflict_ans = choose_conflict_answer_priority_A(gold, cat, by_cat_sorted, global_sorted)
            if conflict_ans == gold:
                # ultra-safe fallback
                conflict_ans = "unknown"
            if conflict_ans == "unknown":
                unknown_conflict_cnt += 1
           

            img_path = Path(args.image_root) / img_id
            if not img_path.exists():
                skipped_missing_img += 1
                continue

            image = Image.open(img_path).convert("RGB")
            image.thumbnail((672, 672))
            # NC (no evidence)
            nc_prompt = make_prompt_no_evidence(question)
            nc_messages = build_messages_multimodal(image, nc_prompt)  # 娉ㄦ剰锛氫綘涔嬪墠淇杩?system 涔熻 list block
            nc_scores = score_two_options_mm(model, processor, nc_messages, gold, conflict_ans)
            nc_pred, _ = argmax_two(nc_scores)
            nc_m = margin_two(nc_scores)

            # SUPPORT
            sup_prompt = make_prompt_with_evidence(question, gold, args.position)
            sup_messages = build_messages_multimodal(image, sup_prompt)
            sup_scores = score_two_options_mm(model, processor, sup_messages, gold, conflict_ans)
            sup_pred, _ = argmax_two(sup_scores)
            sup_m = margin_two(sup_scores)

            # CONFLICT
            con_prompt = make_prompt_with_evidence(question, conflict_ans, args.position)
            con_messages = build_messages_multimodal(image, con_prompt)
            con_scores = score_two_options_mm(model, processor, con_messages, gold, conflict_ans)
            con_pred, _ = argmax_two(con_scores)
            con_m = margin_two(con_scores)

            N += 1
            sup_margin_sum += sup_m
            con_margin_sum += con_m
            nc_margin_sum += nc_m
            # Keep rows where NC and support are both correct.
            if (nc_pred == gold) and (sup_pred == gold):
                both_correct_rows.append({
                    "id": int(idx),
                    "img_id": img_id,
                    "category": cat,
                    "question": question,
                    "gold": gold,
                    "nc_pred": nc_pred,
                    "ic_pred": sup_pred,
                    "conflict": conflict_ans,
                    "image_path": str(img_path),
                    "nc_margin": float(nc_m),
                    "ic_margin": float(sup_m),
                })
            if nc_pred == gold:
                nc_correct += 1
                nc_correct_cnt += 1

                # Compare conflict prediction against NC baseline.
                if con_pred == nc_pred:
                    resist_given_nc_correct += 1
                else:
                    changed_given_nc_correct += 1

                if con_pred == conflict_ans:
                    follow_conflict_given_nc_correct += 1
            if sup_pred == gold:
                support_correct += 1
            if con_pred == gold:
                conflict_still_gold += 1
            if con_pred == conflict_ans:
                conflict_follow_evidence += 1

            if con_pred == sup_pred:
                resist_cnt += 1
            else:
                changed_cnt += 1
            # --- MAIN: compare CONFLICT vs NC (baseline) ---
            if con_pred == nc_pred:
                resist_vs_nc_cnt += 1
            else:
                changed_vs_nc_cnt += 1

            if con_pred == "unknown":
                conflict_pred_unknown_cnt += 1

            if (nc_pred != "unknown") and (con_pred == "unknown"):
                changed_to_unknown_vs_nc_cnt += 1
            if sup_pred == gold:
                if con_pred == sup_pred:
                    resist_given_sup_correct += 1
                else:
                    changed_given_sup_correct += 1
                if con_pred == conflict_ans:
                    follow_conflict_given_sup_correct += 1
           
            wf.write(json.dumps({
                "id": int(idx),
                "img_id": img_id,
                "image_path": str(img_path),
                "category": cat,
                "question": question,
                "gold": gold,
                "conflict": conflict_ans,
                "position": args.position,

                "support_pred": sup_pred,
                "conflict_pred": con_pred,

                "support_scores": sup_scores,
                "conflict_scores": con_scores,
                "support_margin": sup_m,
                "conflict_margin": con_m,

                "changed": (con_pred != sup_pred),
                "resist": (con_pred == sup_pred),
                "follow_conflict": (con_pred == conflict_ans),
                "resist_to_gold": (con_pred == gold),
                "nc_pred": nc_pred,

                "changed_vs_nc": (con_pred != nc_pred),
                "resist_vs_nc": (con_pred == nc_pred),

                "conflict_pred_is_unknown": (con_pred == "unknown"),
                "changed_to_unknown_vs_nc": ((nc_pred != "unknown") and (con_pred == "unknown")),
                "conflict_is_unknown": (conflict_ans == "unknown"),
            }, ensure_ascii=False) + "\n")
    both_correct_path = out_root / "nc_cc_both_correct.csv"
    pd.DataFrame(both_correct_rows).to_csv(both_correct_path, index=False)
    summary = {
        "model": str(args.model),
        "data_csv": str(args.data_csv),
        "image_root": str(args.image_root),
        "position": args.position,
        "N": N,
        "skipped_missing_img": skipped_missing_img,

        "support_correct_rate": pct(support_correct, N),
        "conflict_still_gold_rate": pct(conflict_still_gold, N),
        "conflict_follow_evidence_rate": pct(conflict_follow_evidence, N),

        "resist_rate": pct(resist_cnt, N),
        "changed_rate": pct(changed_cnt, N),

        "resist_rate_given_sup_correct": pct(resist_given_sup_correct, support_correct),
        "changed_rate_given_sup_correct": pct(changed_given_sup_correct, support_correct),
        "follow_conflict_rate_given_sup_correct": pct(follow_conflict_given_sup_correct, support_correct),

        "avg_support_margin": (sup_margin_sum / N) if N else 0.0,
        "avg_conflict_margin": (con_margin_sum / N) if N else 0.0,

        "files": {"preds_jsonl": str(preds_path), "summary_json": str(summary_path)},
        "nc_correct_rate": pct(nc_correct, N),
        "avg_nc_margin": (nc_margin_sum / N) if N else 0.0,

        "resist_rate_given_nc_correct": pct(resist_given_nc_correct, nc_correct_cnt),
        "changed_rate_given_nc_correct": pct(changed_given_nc_correct, nc_correct_cnt),
        "follow_conflict_rate_given_nc_correct": pct(follow_conflict_given_nc_correct, nc_correct_cnt),
        "nc_correct_cnt": nc_correct_cnt,
        # --- NC baseline (preferred) ---
        "resist_vs_nc_rate": pct(resist_vs_nc_cnt, N),
        "changed_vs_nc_rate": pct(changed_vs_nc_cnt, N),

        "conflict_pred_unknown_rate": pct(conflict_pred_unknown_cnt, N),
        "changed_to_unknown_vs_nc_rate": pct(changed_to_unknown_vs_nc_cnt, N),

        "unknown_conflict_cnt": unknown_conflict_cnt,
        "unknown_conflict_rate": pct(unknown_conflict_cnt, N),
    }
    summary["files"]["nc_cc_both_correct_csv"] = str(both_correct_path)
    summary["nc_cc_both_correct_cnt"] = len(both_correct_rows)
    summary["nc_cc_both_correct_rate_overall"] = pct(len(both_correct_rows), N)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {out_root}")


if __name__ == "__main__":
    main()

