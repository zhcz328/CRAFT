import json
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import defaultdict

YES, NO = "yes", "no"

EVID_TMPL = """EVIDENCE (may be incorrect):
- According to current clinical guidelines, this recommendation {align} with guidelines.
END EVIDENCE
"""
SYSTEM = "You are a medical QA verifier. Answer ONLY with 'Yes' or 'No'. No other words."

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

@torch.no_grad()
def logprob_next_token(model, input_ids, attention_mask, token_id):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)
    return logp[0, token_id].item()

@torch.no_grad()
def score_yes_no_fast(model, tokenizer, prompt_text):
    device = next(model.parameters()).device
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))

    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids  = tokenizer(" No",  add_special_tokens=False).input_ids
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError('" Yes"/" No" not single-token under this tokenizer. Use continuation scoring instead.')

    yes_lp = logprob_next_token(model, input_ids, attn, yes_ids[0])
    no_lp  = logprob_next_token(model, input_ids, attn, no_ids[0])
    delta = yes_lp - no_lp
    pred = YES if delta >= 0 else NO
    return pred, yes_lp, no_lp, delta

def make_evidence(label_yes: bool):
    align = "DOES align" if label_yes else "DOES NOT align"
    return EVID_TMPL.format(align=align).strip()

def wrap_as_chat(tokenizer, user_text: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
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

def inject_evidence(base_prompt: str, evidence: str, position: str) -> str:
    """
    position:
      - prefix: evidence + base_prompt
      - before_question: insert evidence right before the last 'Question:' marker
      - before_answer: insert evidence right before the last 'Answer:' marker
      - suffix: base_prompt + evidence
    Fallback: if marker not found, use prefix.
    """
    ev = evidence.strip() + "\n"

    if position == "prefix":
        return ev + "\n" + base_prompt

    if position == "suffix":
        return base_prompt.rstrip() + "\n\n" + ev

    # Use last occurrence to be robust
    if position == "before_question":
        m = list(re.finditer(r"(?im)^\s*question\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    if position == "before_answer":
        m = list(re.finditer(r"(?im)^\s*answer\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    raise ValueError(f"Unknown position: {position}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="result/conflict_retest_positions.jsonl")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 thinking (default off)")
    ap.add_argument(
        "--positions",
        default="prefix",
        help="Comma-separated positions: prefix,before_question,before_answer,suffix"
    )

    args = ap.parse_args()
    positions = [p.strip() for p in args.positions.split(",") if p.strip()]

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map
    )
    model.eval()

    pairs = read_jsonl(args.pairs)

    # Per-position counters (computed over 114 samples)
    stats = {
        pos: {
            "n": 0,
            "follow_conflict": 0,
            "resist": 0,
            "flip": 0,
            "support_correct": 0,  # pred_support == gold
            "sum_shift_conflict": 0.0,
            "sum_shift_support": 0.0,
        } for pos in positions
    }

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in tqdm(pairs, desc="Position eval", unit="pair"):
            for side in ["correct", "wrong"]:
                s = rec[side]
                base_prompt = s["prompt"]

                gold_yes = (side == "correct")      # by construction
                conflict_yes = (not gold_yes)

                gold = YES if gold_yes else NO
                conf = YES if conflict_yes else NO

                # Baseline computed once (same for all positions)
                p_base = wrap_as_chat(tok, base_prompt, args.enable_thinking)
                pred_b, yb, nb, db = score_yes_no_fast(model, tok, p_base)

                for pos in positions:
                    # Build support/conflict prompts at this position
                    sup_user = inject_evidence(base_prompt, make_evidence(gold_yes), pos)
                    con_user = inject_evidence(base_prompt, make_evidence(conflict_yes), pos)

                    p_sup = wrap_as_chat(tok, sup_user, args.enable_thinking)
                    p_con = wrap_as_chat(tok, con_user, args.enable_thinking)

                    pred_s, ys, ns, ds = score_yes_no_fast(model, tok, p_sup)
                    pred_c, yc, nc, dc = score_yes_no_fast(model, tok, p_con)

                    st = stats[pos]
                    st["n"] += 1
                    st["sum_shift_support"] += (ds - db)
                    st["sum_shift_conflict"] += (dc - db)

                    if pred_s == gold:
                        st["support_correct"] += 1

                    if pred_c == conf:
                        st["follow_conflict"] += 1
                    if pred_c == pred_b:
                        st["resist"] += 1
                    if pred_c != pred_b:
                        st["flip"] += 1

                    out_rec = {
                        "pair_id": rec.get("pair_id"),
                        "side": side,
                        "position": pos,
                        "gold": gold,
                        "conflict_label": conf,
                        "base": {"pred": pred_b, "delta": db},
                        "support": {"pred": pred_s, "delta": ds},
                        "conflict": {"pred": pred_c, "delta": dc},
                        "shift_support": ds - db,
                        "shift_conflict": dc - db,
                    }
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # Print per-position summary
    summary = {}
    for pos in positions:
        st = stats[pos]
        n = st["n"] if st["n"] else 1
        summary[pos] = {
            "n_samples": st["n"],
            "support_acc": st["support_correct"] / n,
            "follow_conflict_rate": st["follow_conflict"] / n,
            "resist_rate(pred_same_as_base)": st["resist"] / n,
            "flip_rate": st["flip"] / n,
            "mean_shift_support": st["sum_shift_support"] / n,
            "mean_shift_conflict": st["sum_shift_conflict"] / n,
        }

    print({
        "saved": args.out,
        "positions": positions,
        "summary_by_position": summary
    })

if __name__ == "__main__":
    main()
