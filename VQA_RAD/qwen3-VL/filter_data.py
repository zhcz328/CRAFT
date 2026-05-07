# -*- coding: utf-8 -*-
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = "You are a helpful medical QA assistant."

BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "Output ONLY the final answer.\n\n"
)

EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)


def norm_text(x: str) -> str:
    return str(x).strip().lower()


# ---------- build by-category answer pool (for wrong answer generation) ----------
def build_by_cat_sorted(df: pd.DataFrame) -> Dict[str, List[str]]:
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = "other"
    df["ans"] = df["answer"].astype(str).map(norm_text)
    df["cat"] = df["category"].astype(str).map(norm_text)

    by_cat = defaultdict(list)
    for a, c in zip(df["ans"].tolist(), df["cat"].tolist()):
        by_cat[c].append(a)

    by_cat_sorted = {}
    for c, lst in by_cat.items():
        cnt = Counter(lst)
        by_cat_sorted[c] = [x for x, _ in cnt.most_common()]
    return by_cat_sorted


def choose_wrong_answer_priority_A_strict(gold: str, cat: str, by_cat_sorted: Dict[str, List[str]]) -> str:
    """
    生成 wrong/conflict 答案（用于二选一评分的另一个候选）：
      - yes/no -> flip
      - 否则：只在同 category 找一个最常见且不同的；找不到就 unknown
    """
    g = norm_text(gold)
    c = norm_text(cat)

    if g == "yes":
        return "no"
    if g == "no":
        return "yes"

    for cand in by_cat_sorted.get(c, []):
        cand_n = norm_text(cand)
        if cand_n and cand_n != g and cand_n != "unknown":
            return cand_n

    return "unknown"


# ---------- prompts ----------
def make_prompt_nc(question: str) -> str:
    return BASE_RULE + f"Question: {question}\nAnswer:"


def make_prompt_cc(question: str, gold: str) -> str:
    return BASE_RULE + EVIDENCE_TMPL.format(ans=gold) + f"\nQuestion: {question}\nAnswer:"


def build_messages_multimodal(image: Image.Image, user_prompt: str) -> List[Dict[str, Any]]:
    # 兼容你当前 transformers：每条 message 的 content 都用 block list
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
    ]


# ---------- KV cache scoring (correct shift) ----------
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

    out = model(**base_inputs, use_cache=True)

    input_ids_prompt = base_inputs["input_ids"]
    attn_prompt = base_inputs.get("attention_mask", torch.ones_like(input_ids_prompt))
    prompt_last_logits = out.logits[:, -1, :].detach()

    return {
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
    prompt_last_logits = cache_pack["prompt_last_logits"]

    cont_len = int(cont_ids.shape[1])

    # first token from prompt_last_logits
    log_probs0 = torch.log_softmax(prompt_last_logits, dim=-1)
    total = float(log_probs0[0, int(cont_ids[0, 0])])

    if cont_len == 1:
        return total

    # remaining tokens: feed cont_ids[:-1] and score cont_ids[1:]
    inp = cont_ids[:, :-1]
    attn_full = torch.cat([attn_prompt, torch.ones_like(inp)], dim=1)

    out = model(
        input_ids=inp,
        attention_mask=attn_full,
        past_key_values=past_key_values,
        use_cache=False,
    )
    log_probs = torch.log_softmax(out.logits, dim=-1)  # [1, cont_len-1, V]

    for i in range(1, cont_len):
        tok_id = int(cont_ids[0, i])
        total += float(log_probs[0, i - 1, tok_id])

    return total


@torch.inference_mode()
def score_two_options_cached(model, processor, messages, gold: str, wrong: str) -> Dict[str, float]:
    cache_pack = build_prompt_cache_mm(model, processor, messages)
    # 为了速度：只用带前导空格的一种形式
    gold_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + gold)
    wrong_lp = logprob_from_cache_mm(model, processor, cache_pack, " " + wrong)
    return {gold: gold_lp, wrong: wrong_lp}


def argmax2(scores: Dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--resize_max_side", type=int, default=672, help="大图长边缩到该值；0 不缩")
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv)
    if "category" not in df.columns:
        df["category"] = "other"

    by_cat_sorted = build_by_cat_sorted(df)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    out_rows = []
    seen = used = miss = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filter NC & CC both-correct"):
        if args.limit > 0 and seen >= args.limit:
            break
        seen += 1

        img_id = str(row["img_id"])
        question = str(row["question"])
        gold = norm_text(row["answer"])
        cat = norm_text(row["category"])

        wrong = choose_wrong_answer_priority_A_strict(gold, cat, by_cat_sorted)

        img_path = Path(args.image_root) / img_id
        if not img_path.exists():
            miss += 1
            continue

        image = Image.open(img_path).convert("RGB")
        if args.resize_max_side and args.resize_max_side > 0:
            image.thumbnail((args.resize_max_side, args.resize_max_side))

        # NC
        nc_prompt = make_prompt_nc(question)
        nc_messages = build_messages_multimodal(image, nc_prompt)
        nc_scores = score_two_options_cached(model, processor, nc_messages, gold, wrong)
        nc_pred = argmax2(nc_scores)

        # CC (correct context)
        cc_prompt = make_prompt_cc(question, gold)
        cc_messages = build_messages_multimodal(image, cc_prompt)
        cc_scores = score_two_options_cached(model, processor, cc_messages, gold, wrong)
        cc_pred = argmax2(cc_scores)

        used += 1
        if (nc_pred == gold) and (cc_pred == gold):
            out_rows.append({
                "id": int(idx),
                "img_id": img_id,
                "image_path": str(img_path),
                "category": cat,
                "question": question,
                "gold": gold,
                "wrong": wrong,
                "nc_pred": nc_pred,
                "cc_pred": cc_pred,
                "nc_lp_gold": float(nc_scores[gold]),
                "nc_lp_wrong": float(nc_scores[wrong]),
                "cc_lp_gold": float(cc_scores[gold]),
                "cc_lp_wrong": float(cc_scores[wrong]),
            })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out_path, index=False)

    print("\nDone.")
    print(f"Seen rows: {seen}")
    print(f"Used rows (images found): {used}")
    print(f"Missing images: {miss}")
    print(f"NC&CC both-correct kept: {len(out_rows)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()