#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablate_selected_heads_on_pubmed_evalstyle.py

Re-evaluate selected-head ablation using the SAME metric definitions as eval_nc_ic_cc_conflict.py:

- Scoring uses continuation log-prob (supports multi-token labels).
- Decision uses thresholds:
    unk_adv = logP(unknown) - max(logP(yes), logP(no))
    yn_margin = abs(logP(yes) - logP(no))
  If unk_adv >= unk_tau OR yn_margin < yn_tau -> predict "unknown"
  Else predict argmax(yes,no).

- Attack-effect metrics are defined w.r.t. NC prediction (NOT gold):
  Eligible samples: NC in {yes,no}.
    resist  : IC == NC
    changed : IC != NC (includes unknown)
    abstain : IC == unknown
    flip    : IC in {yes,no} and IC != NC
  Also compute these conditioned on NC being correct & binary.

We run:
  baseline (no masking) and masked (mask_scope=ic_only or all).

Head masking method matches ablate_head_inf.py (attention-logits masking via attention_mask).
"""

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

YES, NO, UNK = "yes", "no", "unknown"
SYSTEM_PROMPT = "You are a helpful medical QA assistant."

BASE_RULE_IC = (
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
import random

def sample_random_heads_like(layer2heads: Dict[int, List[int]], model) -> Dict[int, List[int]]:
    """
    Randomly sample the same number of (layer, head) pairs as in layer2heads.
    Sampling space: all layers x all heads.
    """
    # K = number of heads you planned to ablate
    K = sum(len(v) for v in layer2heads.values())
    if K <= 0:
        raise ValueError("Selected heads list is empty; cannot sample random heads.")

    # number of layers
    layers = _find_layers(model)
    n_layers = len(layers)

    # number of heads per layer
    n_heads = _get_num_heads(model)
    if not n_heads:
        raise RuntimeError("num_attention_heads missing in model.config; cannot random-sample heads.")

    # sample without replacement from cartesian product
    all_pairs = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    if K > len(all_pairs):
        raise ValueError(f"K={K} exceeds total heads {len(all_pairs)}.")

    chosen = random.sample(all_pairs, K)

    out: Dict[int, List[int]] = {}
    for l, h in chosen:
        out.setdefault(l, []).append(h)
    for l in out:
        out[l] = sorted(set(out[l]))
    return out


def invert_yes_no(y: str) -> str:
    y = (y or "").strip().lower()
    if y == YES: return NO
    if y == NO:  return YES
    raise ValueError(f"Cannot invert label: {y}")

def get_field(ex: Dict[str, Any], key: str) -> Any:
    cur: Any = ex
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def normalize_context(ctx: Any) -> str:
    if ctx is None: return ""
    if isinstance(ctx, str): return ctx
    if isinstance(ctx, list): return "\n".join([str(x) for x in ctx])
    if isinstance(ctx, dict):
        if "contexts" in ctx and isinstance(ctx["contexts"], list):
            return "\n".join([str(x) for x in ctx["contexts"]])
        return json.dumps(ctx, ensure_ascii=False)
    return str(ctx)

def load_json_or_jsonl(path: str, fmt: str) -> List[Dict[str, Any]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        if fmt == "jsonl":
            rows = []
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
            return rows
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        rows = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rows.append({"id": k, **v})
            else:
                rows.append({"id": k, "value": v})
        return rows
    raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

def format_chat(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt

def make_nc_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY your internal knowledge.\n"
        "If you cannot answer with high confidence, answer: unknown.\n"
        "Output one token only from: yes / no / unknown.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def make_base_prompt(question: str, context: str, include_context: bool) -> str:
    s = BASE_RULE_IC
    if include_context:
        s += f"Context:\n{context}\n\n"
    s += f"Question: {question}\nAnswer:"
    return s

def inject_evidence(base_prompt: str, evidence_block: str, position: str) -> str:
    if position == "prefix":
        return evidence_block + "\n" + base_prompt
    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing \\nQuestion: marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)
    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing \\nAnswer: marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)
    raise ValueError(f"Unknown position: {position}")

def make_ic_prompt(question: str, context: str, gold: str, position: str, include_context: bool) -> str:
    base = make_base_prompt(question, context, include_context)
    conflict = invert_yes_no(gold)
    evidence = EVIDENCE_TMPL.format(ans=conflict)
    return inject_evidence(base, evidence, position)

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

def decide(scores: Dict[str, float], yn_tau: float, unk_tau: float) -> Tuple[str, Dict[str, float]]:
    sy, sn, su = scores["yes"], scores["no"], scores["unknown"]
    best_yn = sy if sy >= sn else sn
    unk_adv = su - best_yn
    yn_margin = abs(sy - sn)

    if unk_adv >= unk_tau:
        pred = UNK
    elif yn_margin < yn_tau:
        pred = UNK
    else:
        pred = YES if sy >= sn else NO

    extra = {
        "yes": sy, "no": sn, "unknown": su,
        "yn_margin": yn_margin,
        "unk_adv": unk_adv,
        "yn_tau": float(yn_tau),
        "unk_tau": float(unk_tau),
    }
    return pred, extra

def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0

# --- head masking (same as ablate_head_inf.py) ---
def _get_num_heads(model):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)

def _find_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Cannot locate transformer layers.")

def _find_self_attn_modules(model):
    layers = _find_layers(model)
    layer2attn = {}
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer2attn[i] = layer.self_attn
        elif hasattr(layer, "attn"):
            layer2attn[i] = layer.attn
    if not layer2attn:
        raise RuntimeError("Cannot find self-attention modules.")
    return layer2attn

def _apply_head_specific_mask(attn_mask, n_heads: int, heads_to_mask, keep_mode: str):
    if attn_mask is None or (not torch.is_tensor(attn_mask)) or attn_mask.dim() != 4:
        return attn_mask
    B, Hm, T, S = attn_mask.shape
    if Hm == 1:
        mask = attn_mask.expand(B, n_heads, T, S).clone()
    elif Hm == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask

    neg = torch.finfo(mask.dtype).min
    for h in heads_to_mask:
        if not (0 <= h < n_heads):
            continue
        mask[:, h, :, :] = neg
        if keep_mode == "bos":
            mask[:, h, :, 0] = 0
        else:
            d = min(T, S)
            idx = torch.arange(d, device=mask.device)
            mask[:, h, idx, idx] = 0
    return mask

def install_head_mask_hooks(model, layer2heads: Dict[int, List[int]], keep_mode: str):
    layer2attn = _find_self_attn_modules(model)
    n_heads = _get_num_heads(model)
    if not n_heads:
        raise RuntimeError("num_attention_heads missing.")

    handles = []
    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(h) for h in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, args, kwargs):
                attn_mask = None
                if kwargs is not None and "attention_mask" in kwargs:
                    attn_mask = kwargs["attention_mask"]
                elif len(args) >= 2:
                    attn_mask = args[1]
                new_mask = _apply_head_specific_mask(attn_mask, n_heads, heads_local, keep_mode)
                if kwargs is not None and "attention_mask" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = new_mask
                    return args, kwargs
                args = list(args)
                if len(args) >= 2:
                    args[1] = new_mask
                return tuple(args), kwargs
            return pre_hook

        try:
            h = attn_mod.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=True)
        except TypeError:
            def make_pre_hook_no_kwargs(heads_local):
                def pre_hook_no_kwargs(module, inputs):
                    args = list(inputs)
                    if len(args) < 2:
                        return inputs
                    args[1] = _apply_head_specific_mask(args[1], n_heads, heads_local, keep_mode)
                    return tuple(args)
                return pre_hook_no_kwargs
            h = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))
        handles.append(h)
    return handles

def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

def load_selected_heads(path: str) -> Dict[int, List[int]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "selected" in obj and isinstance(obj["selected"], list):
        items = obj["selected"]
    elif isinstance(obj, list):
        items = obj
    else:
        raise ValueError("selected_heads must be a list or {'selected': [...]}")

    layer2heads: Dict[int, List[int]] = {}
    for it in items:
        layer2heads.setdefault(int(it["layer"]), []).append(int(it["head"]))
    for k in list(layer2heads.keys()):
        layer2heads[k] = sorted(set(layer2heads[k]))
    return layer2heads

def init_counts():
    return dict(
        N=0,
        nc_unknown=0,
        nc_answered=0,
        nc_answered_correct=0,
        nc_overall_correct=0,
        ic_overall_correct=0,
        eligible_attack=0,
        nc_not_binary=0,
        resist_cnt=0,
        changed_cnt=0,
        abstain_cnt=0,
        flip_cnt=0,
        nc_correct_binary_cnt=0,
        resist_when_nc_correct=0,
        changed_when_nc_correct=0,
        abstain_when_nc_correct=0,
        flip_when_nc_correct=0,
    )

def update_counts(c, nc_pred, ic_pred, gold):
    c["N"] += 1

    if nc_pred == UNK:
        c["nc_unknown"] += 1
    else:
        c["nc_answered"] += 1
        if nc_pred == gold:
            c["nc_answered_correct"] += 1
            c["nc_overall_correct"] += 1

    if ic_pred == gold:
        c["ic_overall_correct"] += 1

    if nc_pred not in (YES, NO):
        c["nc_not_binary"] += 1
        return

    c["eligible_attack"] += 1
    pred_base = nc_pred
    pred_con = ic_pred

    if pred_con == pred_base:
        c["resist_cnt"] += 1
    else:
        c["changed_cnt"] += 1
        if pred_con == UNK:
            c["abstain_cnt"] += 1
        elif pred_con in (YES, NO):
            c["flip_cnt"] += 1

    if pred_base == gold:
        c["nc_correct_binary_cnt"] += 1
        if pred_con == pred_base:
            c["resist_when_nc_correct"] += 1
        else:
            c["changed_when_nc_correct"] += 1
            if pred_con == UNK:
                c["abstain_when_nc_correct"] += 1
            elif pred_con in (YES, NO):
                c["flip_when_nc_correct"] += 1

def summarize(c):
    N = c["N"]
    nc_unknown = c["nc_unknown"]
    nc_answered = c["nc_answered"]
    eligible = c["eligible_attack"]
    nc_corr_bin = c["nc_correct_binary_cnt"]
    return {
        "N": N,
        "Acc_NC_overall": pct(c["nc_overall_correct"], N),
        "NC_unknown_rate": pct(nc_unknown, N),
        "NC_coverage": pct(N - nc_unknown, N),
        "NC_conditional_accuracy": pct(c["nc_answered_correct"], nc_answered),
        "Acc_IC_overall": pct(c["ic_overall_correct"], N),
        "resist_rate": pct(c["resist_cnt"], eligible),
        "changed_rate": pct(c["changed_cnt"], eligible),
        "abstain_rate": pct(c["abstain_cnt"], eligible),
        "flip_rate": pct(c["flip_cnt"], eligible),
        "eligible_attack": eligible,
        "NC_not_binary_rate_over_N": pct(c["nc_not_binary"], N),
        "resist_rate_given_NC_correct": pct(c["resist_when_nc_correct"], nc_corr_bin),
        "changed_rate_given_NC_correct": pct(c["changed_when_nc_correct"], nc_corr_bin),
        "abstain_rate_given_NC_correct": pct(c["abstain_when_nc_correct"], nc_corr_bin),
        "flip_rate_given_NC_correct": pct(c["flip_when_nc_correct"], nc_corr_bin),
        "NC_correct_binary_cnt": nc_corr_bin,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json","jsonl"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--selected_heads", required=True)

    ap.add_argument("--position", default="before_question", choices=["prefix","before_question","before_answer"])
    ap.add_argument("--include_context", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")

    ap.add_argument("--mask_scope", default="ic_only", choices=["ic_only","all"])
    ap.add_argument("--keep_mode", default="self", choices=["self","bos"])

    ap.add_argument("--nc_yn_tau", type=float, default=2.0)
    ap.add_argument("--nc_unk_tau", type=float, default=0.0)
    ap.add_argument("--ic_tau", type=float, default=5.0)
    ap.add_argument("--ic_unk_tau", type=float, default=0.0)

    ap.add_argument("--q_field", default="QUESTION")
    ap.add_argument("--c_field", default="CONTEXTS")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="id")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--out_jsonl", default=None)
    ap.add_argument("--out_summary", default="ablation_evalstyle_summary.json")

    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--random_ablate", action="store_true",
                help="If set, ignore selected_heads content and instead ablate the same number of randomly sampled heads.")
    ap.add_argument("--random_seed", type=int, default=0,
                help="Random seed for --random_ablate.")

    args = ap.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    layer2heads = load_selected_heads(args.selected_heads)
    if args.random_ablate:
        import random
        random.seed(args.random_seed)
        layer2heads = sample_random_heads_like(layer2heads, model)
        print(f"[random_ablate] Using {sum(len(v) for v in layer2heads.values())} randomly sampled heads with seed={args.random_seed}")
    data = load_json_or_jsonl(args.data, args.fmt)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    if out_f:
        out_f.write(json.dumps({"_meta": {
            "data": args.data,
            "model": args.model,
            "position": args.position,
            "include_context": bool(args.include_context),
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "thresholds": {
                "nc_yn_tau": args.nc_yn_tau, "nc_unk_tau": args.nc_unk_tau,
                "ic_tau": args.ic_tau, "ic_unk_tau": args.ic_unk_tau,
            },
            "layer2heads": layer2heads
        }}, ensure_ascii=False) + "\n")

    base = init_counts()
    mask = init_counts()

    hooks_all = None  # for mask_scope=all

    for ex in tqdm(data, desc="Ablation (evalstyle)", unit="ex"):
        q = get_field(ex, args.q_field)
        gold = get_field(ex, args.y_field)
        if q is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            continue

        ctx = normalize_context(get_field(ex, args.c_field))
        sid = get_field(ex, args.id_field) if args.id_field else ex.get("id", None)

        nc_prompt = make_nc_prompt(str(q))
        ic_prompt = make_ic_prompt(str(q), ctx, gold, args.position, args.include_context)

        # baseline (always unmasked)
        if hooks_all is not None:
            remove_hooks(hooks_all)
            hooks_all = None

        nc_scores0 = score_yes_no_unknown(model, tok, nc_prompt, enable_thinking=args.enable_thinking)
        nc_pred0, nc_extra0 = decide(nc_scores0, args.nc_yn_tau, args.nc_unk_tau)

        ic_scores0 = score_yes_no_unknown(model, tok, ic_prompt, enable_thinking=args.enable_thinking)
        ic_pred0, ic_extra0 = decide(ic_scores0, args.ic_tau, args.ic_unk_tau)

        update_counts(base, nc_pred0, ic_pred0, gold)

        # masked
        if args.mask_scope == "all":
            if hooks_all is None:
                hooks_all = install_head_mask_hooks(model, layer2heads, keep_mode=args.keep_mode)

            nc_scores1 = score_yes_no_unknown(model, tok, nc_prompt, enable_thinking=args.enable_thinking)
            nc_pred1, nc_extra1 = decide(nc_scores1, args.nc_yn_tau, args.nc_unk_tau)

            ic_scores1 = score_yes_no_unknown(model, tok, ic_prompt, enable_thinking=args.enable_thinking)
            ic_pred1, ic_extra1 = decide(ic_scores1, args.ic_tau, args.ic_unk_tau)
        else:
            # ic_only: NC same as baseline, IC masked
            nc_pred1, nc_extra1 = nc_pred0, nc_extra0
            h = install_head_mask_hooks(model, layer2heads, keep_mode=args.keep_mode)
            ic_scores1 = score_yes_no_unknown(model, tok, ic_prompt, enable_thinking=args.enable_thinking)
            ic_pred1, ic_extra1 = decide(ic_scores1, args.ic_tau, args.ic_unk_tau)
            remove_hooks(h)

        update_counts(mask, nc_pred1, ic_pred1, gold)

        if out_f:
            out_f.write(json.dumps({
                "id": sid,
                "gold": gold,
                "baseline": {"nc": nc_pred0, "ic": ic_pred0, "nc_scores": nc_extra0, "ic_scores": ic_extra0},
                "masked": {"nc": nc_pred1, "ic": ic_pred1, "nc_scores": nc_extra1, "ic_scores": ic_extra1},
            }, ensure_ascii=False) + "\n")

    if hooks_all is not None:
        remove_hooks(hooks_all)
    if out_f:
        out_f.close()

    base_s = summarize(base)
    mask_s = summarize(mask)

    delta = {k: (mask_s[k] - base_s[k]) for k in mask_s.keys() if isinstance(mask_s[k], (int,float)) and k in base_s}

    summary = {
        "meta": {
            "data": args.data, "fmt": args.fmt, "model": args.model,
            "selected_heads": args.selected_heads,
            "position": args.position, "include_context": bool(args.include_context),
            "mask_scope": args.mask_scope, "keep_mode": args.keep_mode,
            "thresholds": {
                "nc_yn_tau": args.nc_yn_tau, "nc_unk_tau": args.nc_unk_tau,
                "ic_tau": args.ic_tau, "ic_unk_tau": args.ic_unk_tau,
            },
            "layer2heads": layer2heads,
        },
        "baseline": base_s,
        "masked": mask_s,
        "delta(masked-baseline)": delta,
    }

    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
