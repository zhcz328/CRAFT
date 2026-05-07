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
# Head masking (ablation) via attention-logits masking (pre-softmax)
#
# IMPORTANT:
#   - We do NOT set head output tensors to -inf (that would blow up residual stream).
#   - Instead, we modify the *attention mask* passed into each layer's self-attention so that,
#     for selected heads, almost all key positions get a very large negative bias BEFORE softmax.
#   - To avoid NaNs (softmax over all -inf), we keep at least one key position unmasked:
#       default: keep "self" position (diagonal), so the ablated head attends only to itself.
# --------------------------

def _get_num_heads_and_hidden(model):
    cfg = getattr(model, "config", None)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    nheads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    return nheads, hidden

def _find_layers(model):
    layer_lists = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_lists.append(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_lists.append(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        layer_lists.append(model.gpt_neox.layers)
    if not layer_lists:
        raise RuntimeError("Cannot locate transformer layers (model.model.layers / transformer.h / gpt_neox.layers).")
    return layer_lists[0]

def _find_self_attn_modules(model):
    """
    Returns:
      layer2attn: dict[layer_index -> self-attention module]
      n_layers: int
    """
    layers = _find_layers(model)
    layer2attn = {}
    for i, layer in enumerate(layers):
        mod = None
        if hasattr(layer, "self_attn"):
            mod = layer.self_attn
        elif hasattr(layer, "attn"):
            mod = layer.attn
        if mod is not None:
            layer2attn[i] = mod
    if not layer2attn:
        raise RuntimeError("Cannot find any self-attention modules in layers.")
    return layer2attn, len(layers)

def _apply_head_specific_mask(attn_mask, n_heads: int, heads_to_mask, keep_mode: str = "self"):
    """
    attn_mask: Tensor of shape [B, 1, T, S] or [B, n_heads, T, S] (float additive mask)
    heads_to_mask: list[int]
    keep_mode:
      - "self": keep diagonal (query t can still attend to key t)
      - "bos": keep key=0 for all queries
    """
    if attn_mask is None:
        return None

    if not torch.is_tensor(attn_mask):
        return attn_mask

    if attn_mask.dim() != 4:
        # Many HF models pass a 2D attention_mask at the top-level, but by the time it
        # reaches the attention module it is typically expanded to 4D. If it's still not 4D,
        # we cannot do head-specific masking reliably here.
        return attn_mask

    B, Hm, T, S = attn_mask.shape

    # Expand to per-head mask if needed
    if Hm == 1:
        mask = attn_mask.expand(B, n_heads, T, S).clone()
    elif Hm == n_heads:
        mask = attn_mask.clone()
    else:
        # Unexpected head/broadcast dimension
        return attn_mask

    # Use a very negative number (acts like -inf for softmax) in the same dtype
    neg = torch.finfo(mask.dtype).min

    for h in heads_to_mask:
        if not (0 <= h < n_heads):
            continue
        # Mask everything for this head...
        mask[:, h, :, :] = neg

        # ...but keep at least one key position to avoid all -inf => NaN in softmax
        if keep_mode == "bos":
            mask[:, h, :, 0] = 0
        else:  # "self" default
            # Keep diagonal positions where query index == key index (only valid up to min(T,S))
            diag_len = min(T, S)
            idx = torch.arange(diag_len, device=mask.device)
            mask[:, h, idx, idx] = 0

    return mask

def install_head_mask_hooks(model, layer2heads, keep_mode: str = "self"):
    """
    Install forward-pre-hooks on each layer's self-attention module.
    For selected heads in each layer, we modify the attention_mask (additive mask)
    so that those heads are effectively suppressed BEFORE softmax.

    keep_mode:
      - "self" (default): ablated head attends only to itself (diagonal)
      - "bos": ablated head attends only to BOS (key=0)
    """
    layer2attn, _ = _find_self_attn_modules(model)
    n_heads_cfg, _ = _get_num_heads_and_hidden(model)

    if n_heads_cfg is None or n_heads_cfg <= 0:
        raise RuntimeError("Cannot read num_attention_heads from model.config; cannot apply head-specific logits mask.")

    handles = []

    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(h) for h in heads))

        def make_pre_hook(heads_local):
            def pre_hook(module, args, kwargs):
                # Find attention_mask in kwargs or args (common HF signature: (hidden_states, attention_mask, ...))
                attn_mask = None
                if kwargs is not None and "attention_mask" in kwargs:
                    attn_mask = kwargs["attention_mask"]
                elif len(args) >= 2:
                    attn_mask = args[1]

                new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)

                if new_mask is attn_mask:
                    return args, kwargs

                # Write back
                if kwargs is not None and "attention_mask" in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["attention_mask"] = new_mask
                    return args, kwargs
                else:
                    args = list(args)
                    if len(args) >= 2:
                        args[1] = new_mask
                    return tuple(args), kwargs
            return pre_hook

        # Prefer with_kwargs=True so we can safely edit kwargs
        try:
            handle = attn_mod.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=True)
        except TypeError:
            # Older PyTorch: no with_kwargs support; we can only see args.
            def make_pre_hook_no_kwargs(heads_local):
                def pre_hook_no_kwargs(module, inputs):
                    args = list(inputs)
                    if len(args) < 2:
                        return inputs
                    attn_mask = args[1]
                    new_mask = _apply_head_specific_mask(attn_mask, n_heads_cfg, heads_local, keep_mode=keep_mode)
                    args[1] = new_mask
                    return tuple(args)
                return pre_hook_no_kwargs
            handle = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))

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
        device_map=args.device_map,
        attn_implementation="eager",
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
