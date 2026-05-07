import json
import argparse
from tqdm import tqdm
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def load_conflict_specific_heads(head_groups_json_path):
    """
    head_groups.json schema has:
      - conflict_specific: [{layer:int, head:int, ...}, ...]
    """
    with open(head_groups_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    heads = obj.get("conflict_specific", [])
    layer2heads = {}
    for h in heads:
        layer = int(h["layer"])
        head = int(h["head"])
        layer2heads.setdefault(layer, []).append(head)
    # de-dup + sort
    for k in list(layer2heads.keys()):
        layer2heads[k] = sorted(set(layer2heads[k]))
    return layer2heads, obj

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
    ev = evidence.strip() + "\n"

    if position == "prefix":
        return ev + "\n" + base_prompt

    if position == "suffix":
        return base_prompt.rstrip() + "\n\n" + ev

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

# --------------------------
# Head masking (ablation) via o_proj forward hook
# --------------------------

def _get_num_heads_and_hidden(model):
    cfg = getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    nheads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    return nheads, hidden

def _find_o_proj_modules(model):
    """
    Try to find per-layer attention output projection modules.
    Returns a dict: layer_index -> module
    Supports common HF model structures + trust_remote_code models (best-effort).
    """
    # Candidate containers for layers
    layer_lists = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_lists.append(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_lists.append(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layer_lists.append(model.gpt_neox.layers)

    if not layer_lists:
        raise RuntimeError("Cannot locate transformer layers (model.model.layers / transformer.h / gpt_neox.layers).")

    layers = layer_lists[0]  # pick first match
    layer2oproj = {}

    for i, layer in enumerate(layers):
        mod = None

        # Common patterns
        # LLaMA/Qwen-like: layer.self_attn.o_proj
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            mod = layer.self_attn.o_proj

        # Some variants: layer.attn.o_proj or layer.attn.out_proj
        if mod is None and hasattr(layer, "attn"):
            if hasattr(layer.attn, "o_proj"):
                mod = layer.attn.o_proj
            elif hasattr(layer.attn, "out_proj"):
                mod = layer.attn.out_proj

        # GPT2-like: layer.attn.c_proj (not ideal, but acts as output projection)
        if mod is None and hasattr(layer, "attn") and hasattr(layer.attn, "c_proj"):
            mod = layer.attn.c_proj

        if mod is not None:
            layer2oproj[i] = mod

    if not layer2oproj:
        raise RuntimeError("Cannot find any o_proj/out_proj modules in layers.")

    return layer2oproj, len(layers)

import torch

def install_head_mask_hooks(model, layer2heads):
    layer2oproj, _ = _find_o_proj_modules(model)
    n_heads_cfg, _ = _get_num_heads_and_hidden(model)

    handles = []

    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2oproj:
            continue

        oproj = layer2oproj[layer_idx]
        heads = sorted(set(int(h) for h in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, inputs):
                x = inputs[0]  # [B, T, H]
                if not torch.is_tensor(x) or x.dim() < 2:
                    return inputs

                H = x.size(-1)
                if n_heads_cfg is None or n_heads_cfg <= 0 or H % n_heads_cfg != 0:
                    return inputs

                head_dim = H // n_heads_cfg
                x2 = x.clone()

                for h in heads_local:
                    if 0 <= h < n_heads_cfg:
                        s = h * head_dim
                        e = (h + 1) * head_dim
                        x2[..., s:e] = 0

                # 返回新的 inputs tuple（不调用 module）
                return (x2,) + tuple(inputs[1:])
            return pre_hook

        # PyTorch 不同版本对 with_kwargs 支持不同，这里做兼容
        try:
            handle = oproj.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=False)
        except TypeError:
            handle = oproj.register_forward_pre_hook(make_pre_hook(heads))

        handles.append(handle)

    return handles


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

# --------------------------
# Main eval
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="same jsonl as your second script uses")
    ap.add_argument("--model", required=True)
    ap.add_argument("--head_groups", required=True, help="path to head_groups.json")
    ap.add_argument("--out", default="result/conflict_retest_masked_heads.jsonl")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16","bfloat16","float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--positions", default="prefix",
                    help="Comma-separated positions: prefix,before_question,before_answer,suffix")
    ap.add_argument("--max_pairs", type=int, default=None, help="optional cap for pairs")
    ap.add_argument("--mask_scope", default="all", choices=["all", "conflict_only"],
                    help="mask heads for: all prompts (base/support/conflict) OR conflict prompt only")

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

    layer2heads, groups_obj = load_conflict_specific_heads(args.head_groups)
    # install hooks once (fast)
    hook_handles = install_head_mask_hooks(model, layer2heads)

    pairs = read_jsonl(args.pairs)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    # Per-position counters (computed over samples = 2 * n_pairs)
    stats = {
        pos: {
            "n": 0,
            "follow_conflict": 0,
            "resist": 0,
            "flip": 0,
            "support_correct": 0,
            "sum_shift_conflict": 0.0,
            "sum_shift_support": 0.0,
        } for pos in positions
    }

    # Save metadata header (optional, first line)
    meta = {
        "head_groups_path": args.head_groups,
        "masked_conflict_specific": [{"layer": k, "heads": v} for k, v in sorted(layer2heads.items())],
        "fixed_thresholds": groups_obj.get("fixed_thresholds"),
        "mask_scope": args.mask_scope,
        "positions": positions,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")

        for rec in tqdm(pairs, desc="Masked-head retest", unit="pair"):
            for side in ["correct", "wrong"]:
                s = rec[side]
                base_prompt = s["prompt"]

                gold_yes = (side == "correct")      # by construction
                conflict_yes = (not gold_yes)

                gold = YES if gold_yes else NO
                conf = YES if conflict_yes else NO

                # Baseline computed once
                p_base = wrap_as_chat(tok, base_prompt, args.enable_thinking)

                if args.mask_scope == "conflict_only":
                    # base without masking: temporarily remove hooks
                    remove_hooks(hook_handles)
                    pred_b, yb, nb, db = score_yes_no_fast(model, tok, p_base)
                    # re-install for support/conflict
                    hook_handles = install_head_mask_hooks(model, layer2heads)
                else:
                    pred_b, yb, nb, db = score_yes_no_fast(model, tok, p_base)

                for pos in positions:
                    sup_user = inject_evidence(base_prompt, make_evidence(gold_yes), pos)
                    con_user = inject_evidence(base_prompt, make_evidence(conflict_yes), pos)

                    p_sup = wrap_as_chat(tok, sup_user, args.enable_thinking)
                    p_con = wrap_as_chat(tok, con_user, args.enable_thinking)

                    if args.mask_scope == "conflict_only":
                        # support without masking (optional, but keep consistent with "conflict_only")
                        remove_hooks(hook_handles)
                        pred_s, ys, ns, ds = score_yes_no_fast(model, tok, p_sup)
                        hook_handles = install_head_mask_hooks(model, layer2heads)

                        pred_c, yc, nc, dc = score_yes_no_fast(model, tok, p_con)
                    else:
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
        "head_groups": args.head_groups,
        "mask_scope": args.mask_scope,
        "positions": positions,
        "summary_by_position": summary
    })

    remove_hooks(hook_handles)

if __name__ == "__main__":
    main()
