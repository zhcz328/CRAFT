# -*- coding: utf-8 -*-
"""
Filter PubMedQA-style dataset to the subset where a given model answers BOTH:
  - NC (No Context; internal knowledge) correctly
  - CC (Clean Context; uses provided context) correctly

Key changes vs eval_models_scheme1_flipctx_thinking_modes_ab.py:
  - Removes IC completely (no conflict construction).
  - NC in logit mode uses 3-way scoring (yes / no / unknown) with an abstention rule:
      1) If unknown is sufficiently preferred vs best(yes,no): abstain (unknown)
      2) Else if yes/no margin is too small: abstain (unknown)
      3) Else choose argmax(yes,no)
  - Reports and saves NC coverage + conditional accuracy (Scheme A style), per model.
  - Saves a filtered dataset file per model.

Recommended threshold presets (can be overridden by CLI args):
  - lenient:  nc_yn_tau=1.0, nc_unk_tau=0.0
  - balanced: nc_yn_tau=2.0, nc_unk_tau=0.0   (good default)
  - strict:   nc_yn_tau=3.0, nc_unk_tau=0.5

Output:
  results_pubmedqa_filter_nc_cc/
    <model_sanitized>.filtered.json        (filtered subset; JSON list of examples)
    <model_sanitized>.filtered.jsonl       (same subset; one example per line)
    <model_sanitized>.meta.json            (coverage/conditional-acc + counts)
    <model_sanitized>.preds.jsonl          (per-sample predictions + logits; for auditing)

Notes:
  - This script filters out gold=="maybe" by default (keeps only yes/no).
  - CC is scored as yes/no only (forced, unless you set --cc_tau > 0 to allow abstention).
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

LABELS = ("yes", "no", "unknown")


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def strip_think(text: str) -> str:
    # Kept for robustness; logit mode doesn't generate, but we may still store raw.
    t = text or ""
    t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<reasoning>.*?</reasoning>", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<think>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"<reasoning>.*", " ", t, flags=re.DOTALL | re.IGNORECASE)
    return t


def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    """Use chat template when available; pass enable_thinking if supported."""
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


def make_prompt(question: str, context: Optional[str], mode: str) -> str:
    """
    mode:
      - "NC": no context, internal knowledge only
      - "CC": must use provided context only
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
        "If the context is insufficient or contradictory, answer: unknown.\n"
        "Output one token only from: yes / no / unknown.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def load_local_dataset(path: str, fmt: Optional[str] = None) -> List[Dict[str, Any]]:
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
            # auto-detect
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


@torch.inference_mode()
def logprob_continuation(model, tok, prompt: str, continuation: str, enable_thinking: bool) -> float:
    """Compute log P(continuation | prompt) by teacher forcing."""
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
        if pos - 1 < 0 or pos - 1 >= log_probs.shape[1]:
            continue
        total += float(log_probs[0, pos - 1, int(tok_id)])
    return total


@torch.inference_mode()
def score_yes_no_unknown(model, tok, prompt: str, enable_thinking: bool) -> Dict[str, float]:
    """
    Return log-probs for yes/no/unknown (each is max over spaced & unspaced variants).
    """
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


def decide_nc_from_3way(scores: Dict[str, float], yn_tau: float, unk_tau: float) -> Tuple[str, Dict[str, float]]:
    """
    Two-step abstention:
      1) If unknown is preferred by >= unk_tau over best(yes,no): output unknown.
      2) Else if |yes-no| < yn_tau: output unknown (low-confidence yes/no).
      3) Else output argmax(yes,no).
    Returns (pred, extras) where extras include margins for auditing.
    """
    sy, sn, su = scores["yes"], scores["no"], scores["unknown"]
    best_yn = sy if sy >= sn else sn
    unk_adv = su - best_yn
    yn_margin = abs(sy - sn)

    if unk_tau is not None and unk_adv >= float(unk_tau):
        pred = "unknown"
    elif yn_tau is not None and yn_margin < float(yn_tau):
        pred = "unknown"
    else:
        pred = "yes" if sy >= sn else "no"

    extras = {
        "yes": sy, "no": sn, "unknown": su,
        "yn_margin": yn_margin,
        "unk_adv": unk_adv,
        "yn_tau": float(yn_tau),
        "unk_tau": float(unk_tau),
    }
    return pred, extras


def decide_yes_no_or_unknown_from_delta(delta: float, tau: float) -> str:
    """For CC: keep your existing tau-based abstention (optional)."""
    if tau is not None and tau > 0.0 and abs(delta) < tau:
        return "unknown"
    return "yes" if delta >= 0 else "no"


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


def eval_and_filter_one_model(
    model_name: str,
    data: List[Dict[str, Any]],
    out_dir: str,
    q_field: str,
    c_field: str,
    y_field: str,
    id_field: str,
    limit: int,
    device_map: str,
    dtype: str,
    enable_thinking: bool,
    # NC 3-way thresholds
    nc_yn_tau: float,
    nc_unk_tau: float,
    # CC thresholds (optional abstain)
    cc_tau: float,
    cc_unk_tau: float,

) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    base = sanitize(model_name)

    preds_path = os.path.join(out_dir, f"{base}.preds.jsonl")
    filtered_json_path = os.path.join(out_dir, f"{base}.filtered.json")
    filtered_jsonl_path = os.path.join(out_dir, f"{base}.filtered.jsonl")
    meta_path = os.path.join(out_dir, f"{base}.meta.json")

    model, tok = load_model(model_name, device_map=device_map, dtype=dtype)

    n_eval = 0
    skipped_maybe = 0

    # NC coverage/conditional-acc bookkeeping (over evaluated yes/no gold samples)
    nc_unknown = 0
    nc_answered = 0
    nc_answered_correct = 0

    # Filtering counts
    kept = 0
    skipped_not_both_correct = 0

    filtered_rows: List[Dict[str, Any]] = []

    total_cap = limit if (limit and limit > 0) else len(data)

    with open(preds_path, "w", encoding="utf-8") as wf:
        for i, ex in enumerate(tqdm(data, desc=f"Filter {model_name}", total=total_cap)):
            if limit > 0 and n_eval >= limit:
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

            ctx_cc = normalize_context(ctx_raw)

            # Prompts
            nc_prompt = make_prompt(str(q), None, "NC")
            cc_prompt = make_prompt(str(q), ctx_cc, "CC")

            # NC: 3-way scoring + abstention
            nc_scores = score_yes_no_unknown(model, tok, nc_prompt, enable_thinking=enable_thinking)
            nc_pred, nc_extra = decide_nc_from_3way(nc_scores, yn_tau=nc_yn_tau, unk_tau=nc_unk_tau)

            # CC: score yes/no (optionally abstain by cc_tau)
            cc_scores = score_yes_no_unknown(model, tok, cc_prompt, enable_thinking=enable_thinking)
            cc_pred, cc_extra = decide_nc_from_3way(cc_scores, yn_tau=cc_tau, unk_tau=cc_unk_tau)


            # Update NC coverage stats
            n_eval += 1
            if nc_pred == "unknown":
                nc_unknown += 1
            else:
                nc_answered += 1
                nc_answered_correct += int(nc_pred == gold)

            both_correct = (nc_pred == gold) and (cc_pred == gold)

            rec = {
                "id": sid,
                "gold": gold,
                "question": q,
                "nc": nc_pred,
                "cc": cc_pred,
                "nc_scores": nc_extra,
                "cc_scores": cc_extra,
                "both_correct": bool(both_correct),
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if both_correct:
                kept += 1
                # Save the ORIGINAL example, plus optional audit fields
                row = dict(ex)
                row["_filter_meta"] = {
                    "id": sid,
                    "gold": gold,
                    "nc": nc_pred,
                    "cc": cc_pred,
                    "nc_scores": nc_extra,
                    "cc_scores": cc_extra,
                    "thresholds": {"nc_yn_tau": float(nc_yn_tau), "nc_unk_tau": float(nc_unk_tau), "cc_tau": float(cc_tau), "cc_unk_tau": float(cc_unk_tau)},

                }
                filtered_rows.append(row)
            else:
                skipped_not_both_correct += 1

    # Save filtered dataset
    with open(filtered_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered_rows, f, ensure_ascii=False, indent=2)
    with open(filtered_jsonl_path, "w", encoding="utf-8") as f:
        for row in filtered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def pct(x: int) -> float:
        return 100.0 * x / n_eval if n_eval else 0.0

    meta = {
        "model": model_name,
        "N_eval": n_eval,
        "skipped_maybe": skipped_maybe,
        "kept_both_nc_cc_correct": kept,
        "skipped_not_both_correct": skipped_not_both_correct,
        # NC Scheme A reporting
        "NC_unknown_rate": pct(nc_unknown),
        "NC_coverage": 100.0 - pct(nc_unknown),
        "NC_conditional_accuracy": (100.0 * nc_answered_correct / nc_answered) if nc_answered else 0.0,
        # thresholds used
        "thresholds": {
            "nc_yn_tau": float(nc_yn_tau),
            "nc_unk_tau": float(nc_unk_tau),
            "cc_tau": float(cc_tau),
            "cc_unk_tau": float(cc_unk_tau),

        },
        "files": {
            "preds": preds_path,
            "filtered_json": filtered_json_path,
            "filtered_jsonl": filtered_jsonl_path,
            "meta": meta_path,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    del model
    torch.cuda.empty_cache()

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Local dataset file path")
    ap.add_argument("--fmt", default="json", choices=["jsonl", "json"], help="Input format")
    ap.add_argument("--out_dir", default="results_pubmedqa_filter_nc_cc")

    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="", help="Optional. e.g. id or PMID")

    ap.add_argument("--models", nargs="+", required=True, help="HF model names/paths, space-separated")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")

    # Threshold presets
    ap.add_argument("--preset", default="balanced", choices=["lenient", "balanced", "strict"],
                    help="Recommended NC 3-way thresholds preset. You can still override with explicit args below.")

    # NC 3-way thresholds
    ap.add_argument("--nc_yn_tau", type=float, default=None,
                    help="NC: if |logp_yes-logp_no| < nc_yn_tau => unknown (low-confidence yes/no).")
    ap.add_argument("--nc_unk_tau", type=float, default=None,
                    help="NC: if logp_unknown - max(logp_yes,logp_no) >= nc_unk_tau => unknown (prefer abstain).")

    # CC abstention (optional)
    ap.add_argument("--cc_tau", type=float, default=3.0,
                    help="CC: if >0, abstain to unknown when |logp_yes-logp_no| < cc_tau. Default 0 (forced yes/no).")
    ap.add_argument("--cc_unk_tau", type=float, default=0.0,
                help="CC: if logp_unknown - max(logp_yes,logp_no) >= cc_unk_tau => unknown.")


    args = ap.parse_args()

    # Preset defaults
    presets = {
        "lenient":  {"nc_yn_tau": 1.0, "nc_unk_tau": 0.0},
        "balanced": {"nc_yn_tau": 2.0, "nc_unk_tau": 0.0},
        "strict":   {"nc_yn_tau": 3.0, "nc_unk_tau": 0.5},
    }
    p = presets[args.preset]
    nc_yn_tau = float(args.nc_yn_tau) if args.nc_yn_tau is not None else p["nc_yn_tau"]
    nc_unk_tau = float(args.nc_unk_tau) if args.nc_unk_tau is not None else p["nc_unk_tau"]

    data = load_local_dataset(args.data, args.fmt)

    all_meta = []
    for m in args.models:
        meta = eval_and_filter_one_model(
            model_name=m,
            data=data,
            out_dir=args.out_dir,
            q_field=args.q_field,
            c_field=args.c_field,
            y_field=args.y_field,
            id_field=args.id_field if args.id_field else "",
            limit=args.limit,
            device_map=args.device_map,
            dtype=args.dtype,
            enable_thinking=args.enable_thinking,
            nc_yn_tau=nc_yn_tau,
            nc_unk_tau=nc_unk_tau,
            cc_tau=float(args.cc_tau),
            cc_unk_tau=float(args.cc_unk_tau),

        )
        all_meta.append(meta)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ALL_META.json"), "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)

    print("\n=== Filter Summary (per model) ===")
    for m in all_meta:
        th = m["thresholds"]
        print(
            f"{m['model']}\tN_eval={m['N_eval']}\tkept={m['kept_both_nc_cc_correct']}"
            f"\tNC_cov={m['NC_coverage']:.2f}\tNC_condAcc={m['NC_conditional_accuracy']:.2f}"
            f"\t(th: nc_yn_tau={th['nc_yn_tau']}, nc_unk_tau={th['nc_unk_tau']}, cc_tau={th['cc_tau']})"
        )
    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()
