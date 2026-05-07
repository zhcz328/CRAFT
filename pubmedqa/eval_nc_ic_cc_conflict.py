# -*- coding: utf-8 -*-
"""
Same as eval_nc_cc_ic_follow_resist.py, but:
- Load each model ONCE, then evaluate multiple positions.
- eval_one_model_one_position() now takes (model, tok) instead of loading inside.
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

SYSTEM_PROMPT = "You are a helpful medical QA assistant."

CC_CORRECT_TAG = "[NOTE: The context below is correct and reliable.]"
IC_INCORRECT_TAG = "[NOTE: The context below is incorrect and unreliable.]"
BASE_RULE = (
    "Answer the question using your best internal medical knowledge.\n"
    "The evidence block (if present) may be correct or incorrect.\n"
    "If you cannot answer with high confidence, answer: unknown.\n"
    "Output one token only from: yes / no / unknown.\n\n"
)

EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)
def invert_yes_no_strict(y: str) -> str:
    y = y.strip().lower()
    if y == "yes":
        return "no"
    if y == "no":
        return "yes"
    raise ValueError(f"Cannot invert label: {y}")
def inject_evidence(base_prompt: str, evidence_block: str, position: str) -> str:
    # base_prompt 预计包含 "Question: ...\nAnswer:" 结构
    if position == "prefix":
        return evidence_block + "\n" + base_prompt

    if position == "before_question":
        # 插到 "Question:" 之前
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    if position == "before_answer":
        # 插到最后的 "\nAnswer:" 之前
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    raise ValueError(f"Unknown position: {position}")


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
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
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return user_prompt


def load_local_dataset(path: str, fmt: str) -> List[Dict[str, Any]]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if fmt == "jsonl":
            rows = []
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
            return rows

        obj = json.load(f)
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


def invert_yes_no(x: str) -> Optional[str]:
    x = (x or "").strip().lower()
    if x == "yes":
        return "no"
    if x == "no":
        return "yes"
    return None


def make_nc_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY your internal knowledge.\n"
        "If you cannot answer with high confidence, answer: unknown.\n"
        "Output one token only from: yes / no / unknown.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
def make_base_prompt(question: str, context: str) -> str:
    context = "" if context is None else str(context)
    return (
        BASE_RULE
        + f"Context:\n{context}\n\n"
        + f"Question: {question}\n"
        + "Answer:"
    )



def make_cc_prompt(question: str, context: str, gold: str, position: str) -> str:
    base = make_base_prompt(question, context)
    evidence = EVIDENCE_TMPL.format(ans=gold)
    # 你也可以在 evidence 前加 CC tag（可选）
    # evidence = f"{CC_CORRECT_TAG}\n{evidence}"
    return inject_evidence(base, evidence, position)



def make_ic_prompt(question: str, context: str, gold: str, position: str) -> str:
    base = make_base_prompt(question, context)
    conflict_ans = invert_yes_no_strict(gold)
    evidence = EVIDENCE_TMPL.format(ans=conflict_ans)
    # 可选：加 IC tag
    # evidence = f"{IC_INCORRECT_TAG}\n{evidence}"
    return inject_evidence(base, evidence, position)



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


@torch.inference_mode()
def logprob_continuation(model, tok, prompt: str, continuation: str, enable_thinking: bool) -> float:
    formatted = format_chat(tok, prompt, enable_thinking=enable_thinking)

    enc_prompt = tok(formatted, add_special_tokens=False, return_tensors="pt")
    enc_full = tok(formatted + continuation, add_special_tokens=False, return_tensors="pt")

    input_ids_prompt = enc_prompt["input_ids"][0]
    input_ids_full = enc_full["input_ids"][0]
    cont_ids = input_ids_full[len(input_ids_prompt):]
    if cont_ids.numel() == 0:
        return float("-inf")

    input_ids_full = input_ids_full.unsqueeze(0).to(model.device)
    outputs = model(input_ids_full)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    start = len(input_ids_prompt)
    total = 0.0
    for i, tok_id in enumerate(cont_ids):
        pos = start + i
        total += float(log_probs[0, pos - 1, int(tok_id)])
    return total


@torch.inference_mode()
def score_yes_no_unknown(model, tok, prompt: str, enable_thinking: bool) -> Dict[str, float]:
    yes_lp = max(
        logprob_continuation(model, tok, prompt, " yes", enable_thinking),
        logprob_continuation(model, tok, prompt, "yes", enable_thinking),
    )
    no_lp = max(
        logprob_continuation(model, tok, prompt, " no", enable_thinking),
        logprob_continuation(model, tok, prompt, "no", enable_thinking),
    )
    unk_lp = max(
        logprob_continuation(model, tok, prompt, " unknown", enable_thinking),
        logprob_continuation(model, tok, prompt, "unknown", enable_thinking),
    )
    return {"yes": yes_lp, "no": no_lp, "unknown": unk_lp}


@torch.inference_mode()
def pick_yes_no_by_logit(model, tok, prompt: str, enable_thinking: bool) -> Tuple[float, float, float]:
    yes_lp = max(
        logprob_continuation(model, tok, prompt, " yes", enable_thinking),
        logprob_continuation(model, tok, prompt, "yes", enable_thinking),
    )
    no_lp = max(
        logprob_continuation(model, tok, prompt, " no", enable_thinking),
        logprob_continuation(model, tok, prompt, "no", enable_thinking),
    )
    return yes_lp, no_lp, (yes_lp - no_lp)


def decide_nc(scores: Dict[str, float], nc_yn_tau: float, nc_unk_tau: float) -> Tuple[str, Dict[str, float]]:
    sy, sn, su = scores["yes"], scores["no"], scores["unknown"]
    best_yn = sy if sy >= sn else sn
    unk_adv = su - best_yn
    yn_margin = abs(sy - sn)

    if unk_adv >= nc_unk_tau:
        pred = "unknown"
    elif yn_margin < nc_yn_tau:
        pred = "unknown"
    else:
        pred = "yes" if sy >= sn else "no"

    extra = {
        "yes": sy, "no": sn, "unknown": su,
        "yn_margin": yn_margin,
        "unk_adv": unk_adv,
        "nc_yn_tau": float(nc_yn_tau),
        "nc_unk_tau": float(nc_unk_tau),
    }
    return pred, extra


def decide_yes_no_or_unknown(delta: float, tau: float) -> str:
    if tau > 0.0 and abs(delta) < tau:
        return "unknown"
    return "yes" if delta >= 0 else "no"


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def eval_one_model_one_position(
    model,
    tok,
    model_name: str,
    data: List[Dict[str, Any]],
    out_root: str,
    position: str,
    q_field: str,
    c_field: str,
    y_field: str,
    id_field: str,
    enable_thinking: bool,
    limit: int,
    nc_yn_tau: float,
    nc_unk_tau: float,
    cc_tau: float,
    cc_unk_tau: float,
    ic_tau: float,
    ic_unk_tau: float,
) -> Dict[str, Any]:

    model_dir = Path(out_root) / sanitize(model_name) / position
    model_dir.mkdir(parents=True, exist_ok=True)

    preds_path = model_dir / "preds.jsonl"
    summary_path = model_dir / "summary.json"

    N = 0
    skipped_maybe = 0

    # NC reporting
    nc_unknown = 0
    nc_answered = 0
    nc_answered_correct = 0
    nc_overall_correct = 0

    # CC/IC accuracies
    cc_overall_correct = 0
    ic_overall_correct = 0

    # attack-effect bookkeeping (defined on samples where NC is binary yes/no)
    eligible_attack = 0
    nc_not_binary = 0
    
    resist_cnt = 0          # IC == NC
    changed_cnt = 0         # IC != NC (including unknown)
    abstain_cnt = 0         # IC == unknown  (within eligible_attack)
    flip_cnt = 0            # IC in {yes,no} and IC != NC
    
    # conditional on NC being correct (and binary)
    nc_correct_binary_cnt = 0
    resist_when_nc_correct = 0
    changed_when_nc_correct = 0
    abstain_when_nc_correct = 0
    flip_when_nc_correct = 0

    with open(preds_path, "w", encoding="utf-8") as wf:
        for i, ex in enumerate(tqdm(data, desc=f"{model_name} @ {position}")):
            if limit > 0 and N >= limit:
                break

            q = get_field(ex, q_field)
            ctx_raw = get_field(ex, c_field)
            gold = get_field(ex, y_field)
            sid = get_field(ex, id_field) if id_field else ex.get("id", i)

            if q is None or gold is None:
                continue
            gold = str(gold).strip().lower()
            if gold == "maybe":
                skipped_maybe += 1
                continue
            if gold not in ("yes", "no"):
                continue

            ctx = normalize_context(ctx_raw)

            # NC
            nc_prompt = make_nc_prompt(str(q))
            nc_scores_raw = score_yes_no_unknown(model, tok, nc_prompt, enable_thinking=enable_thinking)
            nc_pred, nc_scores = decide_nc(nc_scores_raw, nc_yn_tau=nc_yn_tau, nc_unk_tau=nc_unk_tau)

            # CC (with "correct" injection)
            cc_prompt = make_cc_prompt(q, ctx,gold, position)
            cc_scores_raw = score_yes_no_unknown(model, tok, cc_prompt, enable_thinking=enable_thinking)
            cc_pred, cc_scores = decide_nc(cc_scores_raw, nc_yn_tau=cc_tau, nc_unk_tau=cc_unk_tau)

            # IC (with "incorrect" injection)
            ic_prompt = make_ic_prompt(q, ctx,gold, position)
            ic_scores_raw = score_yes_no_unknown(model, tok, ic_prompt, enable_thinking=enable_thinking)
            ic_pred, ic_scores = decide_nc(ic_scores_raw, nc_yn_tau=ic_tau, nc_unk_tau=ic_unk_tau)


            N += 1

            # NC stats
            if nc_pred == "unknown":
                nc_unknown += 1
            else:
                nc_answered += 1
                if nc_pred == gold:
                    nc_answered_correct += 1
                    nc_overall_correct += 1

            # CC/IC accuracy
            if cc_pred == gold:
                cc_overall_correct += 1
            if ic_pred == gold:
                ic_overall_correct += 1

            # follow/resist definitions (based on NC)
           # attack-effect definitions (based on NC)
            label_conflict = invert_yes_no(nc_pred)  # 仍然可以保留写入文件做参考（但不用于统计）
            if nc_pred not in ("yes", "no"):
                nc_not_binary += 1
            else:
                eligible_attack += 1
                pred_base = nc_pred
                pred_con = ic_pred

                # resist / changed
                if pred_con == pred_base:
                    resist_cnt += 1
                else:
                    changed_cnt += 1  # includes unknown and opposite

                # abstain / flip (sub-cases of changed)
                if pred_con == "unknown":
                    abstain_cnt += 1
                elif pred_con in ("yes", "no") and pred_con != pred_base:
                    flip_cnt += 1

                # conditional on NC being correct (and binary)
                if pred_base == gold:
                    nc_correct_binary_cnt += 1
                    if pred_con == pred_base:
                        resist_when_nc_correct += 1
                    else:
                        changed_when_nc_correct += 1
                        if pred_con == "unknown":
                            abstain_when_nc_correct += 1
                        elif pred_con in ("yes", "no"):
                            flip_when_nc_correct += 1

            wf.write(json.dumps({
                "id": sid,
                "gold": gold,
                "question": q,
                "position": position,
                "nc": nc_pred,
                "cc": cc_pred,
                "ic": ic_pred,
                "label_context_conflict": label_conflict,
                "changed": (ic_pred != nc_pred) if nc_pred in ("yes", "no") else None,
                "abstain": (ic_pred == "unknown") if nc_pred in ("yes", "no") else None,
                "flip": (ic_pred in ("yes", "no") and ic_pred != nc_pred) if nc_pred in ("yes", "no") else None,
                "resist": (ic_pred == nc_pred) if nc_pred in ("yes", "no") else None,
                "nc_scores": nc_scores,
                "cc_scores": cc_scores,
                "ic_scores": ic_scores,

                #"injections": {"cc_tag": CC_CORRECT_TAG, "ic_tag": IC_INCORRECT_TAG},
            }, ensure_ascii=False) + "\n")

    summary = {
        "model": model_name,
        "position": position,
        "N": N,
        "skipped_maybe": skipped_maybe,
        "thresholds": {
            "nc_yn_tau": float(nc_yn_tau),
            "nc_unk_tau": float(nc_unk_tau),
            "cc_tau": float(cc_tau),
            "cc_unk_tau": float(cc_unk_tau),
            "ic_tau": float(ic_tau),
            "ic_unk_tau": float(ic_unk_tau),

            #"cc_correct_tag": CC_CORRECT_TAG,
            #"ic_incorrect_tag": IC_INCORRECT_TAG,
        },
        "Acc_NC_overall": pct(nc_overall_correct, N),
        "NC_unknown_rate": pct(nc_unknown, N),
        "NC_coverage": pct(N - nc_unknown, N),
        "NC_conditional_accuracy": pct(nc_answered_correct, nc_answered),
        "Acc_CC_overall": pct(cc_overall_correct, N),
        "Acc_IC_overall": pct(ic_overall_correct, N),
        "resist_rate": pct(resist_cnt, eligible_attack),
        "changed_rate": pct(changed_cnt, eligible_attack),
        "abstain_rate": pct(abstain_cnt, eligible_attack),
        "flip_rate": pct(flip_cnt, eligible_attack),
        "eligible_attack": eligible_attack,
        "NC_not_binary_rate_over_N": pct(nc_not_binary, N),

        "resist_rate_given_NC_correct": pct(resist_when_nc_correct, nc_correct_binary_cnt),
        "changed_rate_given_NC_correct": pct(changed_when_nc_correct, nc_correct_binary_cnt),
        "abstain_rate_given_NC_correct": pct(abstain_when_nc_correct, nc_correct_binary_cnt),
        "flip_rate_given_NC_correct": pct(flip_when_nc_correct, nc_correct_binary_cnt),
        "NC_correct_binary_cnt": nc_correct_binary_cnt,
        "files": {"preds": str(preds_path), "summary": str(summary_path)},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out_dir", default="results_pubmedqa_nc_cc_ic_follow_resist")

    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--positions", nargs="+",
                    default=["prefix", "before_question", "before_answer"],
                    choices=["prefix", "before_question", "before_answer"])

    ap.add_argument("--nc_yn_tau", type=float, default=2.0)
    ap.add_argument("--nc_unk_tau", type=float, default=0.0)
    ap.add_argument("--cc_tau", type=float, default=0.0)
    ap.add_argument("--ic_tau", type=float, default=5.0)
    ap.add_argument("--cc_unk_tau", type=float, default=0.0)
    ap.add_argument("--ic_unk_tau", type=float, default=0.0)

    args = ap.parse_args()
    data = load_local_dataset(args.data, args.fmt)
    os.makedirs(args.out_dir, exist_ok=True)

    all_summaries = []
    for m in args.models:
        model, tok = load_model(m, device_map=args.device_map, dtype=args.dtype)
        try:
            for pos in args.positions:
                summ = eval_one_model_one_position(
                    model=model,
                    tok=tok,
                    model_name=m,
                    data=data,
                    out_root=args.out_dir,
                    position=pos,
                    q_field=args.q_field,
                    c_field=args.c_field,
                    y_field=args.y_field,
                    id_field=args.id_field if args.id_field else "",
                    enable_thinking=args.enable_thinking,
                    limit=args.limit,
                    nc_yn_tau=args.nc_yn_tau,
                    nc_unk_tau=args.nc_unk_tau,
                    cc_tau=args.cc_tau,
                    ic_tau=args.ic_tau,
                    cc_unk_tau=args.cc_unk_tau,
                    ic_unk_tau=args.ic_unk_tau,

                )
                all_summaries.append(summ)
        finally:
            del model
            torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "ALL_SUMMARIES.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n=== Summary (per model x position) ===")
    for s in all_summaries:
        print(
            f"{s['model']} @ {s['position']} | N={s['N']} | "
            f"NC_cov={s['NC_coverage']:.2f} NC_condAcc={s['NC_conditional_accuracy']:.2f} | "
            f"Acc_CC={s['Acc_CC_overall']:.2f} Acc_IC={s['Acc_IC_overall']:.2f} | "
            f"abstain={s['abstain_rate']:.2f} resist={s['resist_rate']:.2f} flip={s['flip_rate']:.2f}"
        )
    print(f"\nSaved to: {args.out_dir}")

if __name__ == "__main__":
    main()
