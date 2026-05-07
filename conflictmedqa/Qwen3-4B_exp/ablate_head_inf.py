import json
import argparse
from pathlib import Path
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

def load_ablation_heads(head_file_path):
    """
    Supported schemas:
    1) selected_heads.json from select_heads.py:
       {"meta": {...}, "selected_heads": [{"layer": int, "head": int, ...}, ...]}
    2) legacy head_groups.json:
       {"conflict_specific": [{"layer": int, "head": int, ...}, ...], ...}
    """
    with open(head_file_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    heads = obj.get("selected_heads")
    if heads is None:
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


def load_conflict_specific_heads(head_groups_json_path):
    return load_ablation_heads(head_groups_json_path)


def infer_default_head_file(model_name: str, model_path: str, pairs_path: str, positions: list[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    pos = positions[0] if positions else "prefix"
    model_l = (model_name or "").lower()
    model_path_l = (model_path or "").lower()
    pairs_l = (pairs_path or "").lower()

    base_dirs = []
    # Highest-priority hint: dataset path tells us which experiment branch is being evaluated.
    if "/llama32_3b/" in pairs_l:
        base_dirs.append(script_dir / "llama32_3b")
    if "llama32_3b" in model_l or "llama32_3b" in model_path_l:
        base_dirs.append(script_dir / "llama32_3b")
    # Always keep both branches as fallback.
    base_dirs.extend([script_dir / "llama32_3b", script_dir])

    dedup_base_dirs = []
    for b in base_dirs:
        if b not in dedup_base_dirs:
            dedup_base_dirs.append(b)

    tried = []
    for base_dir in dedup_base_dirs:
        candidate_roots = [
            base_dir / "result" / pos / "headscan_rounds_top30_inf",
            base_dir / "result" / pos / "headscan_rounds_top30",
            base_dir / "result" / pos / "headscan_rounds_v2",
        ]

        for root in candidate_roots:
            cand = root / "selected_heads.json"
            tried.append(cand)
            if cand.exists():
                return cand
        for root in candidate_roots:
            cand = root / "head_groups.json"
            tried.append(cand)
            if cand.exists():
                return cand

    tried_msg = "\n".join(f"- {p}" for p in tried)
    raise FileNotFoundError(
        "Could not auto-resolve head file. Tried:\n"
        f"{tried_msg}\n"
        "Please pass --head_file explicitly."
    )

@torch.no_grad()
def logprob_continuation(model, tokenizer, prompt_text: str, continuation: str) -> float:
    """
    Match eval scoring: teacher-forced continuation log-prob sum.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt_text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if len(cont_ids) == 0:
        return float("-inf")

    cur_ids = enc["input_ids"]
    cur_attn = enc.get("attention_mask", torch.ones_like(cur_ids))
    total = 0.0

    for tid in cont_ids:
        out = model(input_ids=cur_ids, attention_mask=cur_attn)
        logits = out.logits[:, -1, :]
        lp = torch.log_softmax(logits, dim=-1)[0, tid].item()
        total += lp

        tid_t = torch.tensor([[tid]], device=cur_ids.device)
        cur_ids = torch.cat([cur_ids, tid_t], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(tid_t)], dim=1)

    return total

@torch.no_grad()
def score_yes_no_fast(model, tokenizer, prompt_text):
    # Keep name for compatibility, but scoring is eval-aligned continuation scoring.
    yes_lp = max(
        logprob_continuation(model, tokenizer, prompt_text, " Yes"),
        logprob_continuation(model, tokenizer, prompt_text, "Yes"),
    )
    no_lp = max(
        logprob_continuation(model, tokenizer, prompt_text, " No"),
        logprob_continuation(model, tokenizer, prompt_text, "No"),
    )
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
    ap.add_argument(
        "--head_file",
        "--head_groups",
        dest="head_file",
        default=None,
        help="path to selected_heads.json (preferred) or legacy head_groups.json; if omitted, auto-resolved from model+position",
    )
    ap.add_argument("--out", default="result/conflict_retest_masked_heads.jsonl")
    ap.add_argument("--summary_out", default=None, help="optional path to save summary json")
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
    head_file_path = Path(args.head_file) if args.head_file else infer_default_head_file(
        args.model, args.model, args.pairs, positions
    )
    if args.head_file is None:
        print(f"[info] --head_file not set, auto-selected: {head_file_path}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map,
        attn_implementation="eager",
    )
    model.eval()

    layer2heads, groups_obj = load_ablation_heads(str(head_file_path))
    # install hooks once (fast)
    hook_handles = install_head_mask_hooks(model, layer2heads)

    pairs = read_jsonl(args.pairs)
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]

    # Per-position counters (computed over samples = 2 * n_pairs)
    stats = {
        pos: {
            "n": 0,
            "base_correct_dataset_ref": 0,
            "base_correct_before": 0,
            "base_correct_after": 0,
            "support_correct_before": 0,
            "support_correct_after": 0,
            "follow_conflict_before": 0,
            "follow_conflict_after": 0,
            "resist_after": 0,
            "flip_after": 0,
            "sum_shift_support_before": 0.0,
            "sum_shift_support_after": 0.0,
            "sum_shift_conflict_before": 0.0,
            "sum_shift_conflict_after": 0.0,
        } for pos in positions
    }

    # Save metadata header (optional, first line)
    meta = {
        "head_file_path": str(head_file_path),
        "masked_conflict_specific": [{"layer": k, "heads": v} for k, v in sorted(layer2heads.items())],
        "selection_meta": groups_obj.get("meta"),
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
                dataset_base_pred = (s.get("pred") or "").strip().lower()

                # Baseline computed once
                p_base = wrap_as_chat(tok, base_prompt, args.enable_thinking)

                # BEFORE (no ablation): always evaluate with hooks removed.
                remove_hooks(hook_handles)
                pred_b_before, yb0, nb0, db_before = score_yes_no_fast(model, tok, p_base)
                # Re-install hooks for AFTER path.
                hook_handles = install_head_mask_hooks(model, layer2heads)

                if args.mask_scope == "all":
                    pred_b_after, yb1, nb1, db_after = score_yes_no_fast(model, tok, p_base)
                else:
                    # conflict_only: base is not ablated in AFTER view
                    pred_b_after, db_after = pred_b_before, db_before

                for pos in positions:
                    sup_user = inject_evidence(base_prompt, make_evidence(gold_yes), pos)
                    con_user = inject_evidence(base_prompt, make_evidence(conflict_yes), pos)

                    p_sup = wrap_as_chat(tok, sup_user, args.enable_thinking)
                    p_con = wrap_as_chat(tok, con_user, args.enable_thinking)

                    # BEFORE (no ablation) for support/conflict
                    remove_hooks(hook_handles)
                    pred_s_before, ys0, ns0, ds_before = score_yes_no_fast(model, tok, p_sup)
                    pred_c_before, yc0, nc0, dc_before = score_yes_no_fast(model, tok, p_con)
                    hook_handles = install_head_mask_hooks(model, layer2heads)

                    # AFTER (ablation policy)
                    if args.mask_scope == "all":
                        pred_s_after, ys1, ns1, ds_after = score_yes_no_fast(model, tok, p_sup)
                        pred_c_after, yc1, nc1, dc_after = score_yes_no_fast(model, tok, p_con)
                    else:
                        # conflict_only: support stays unablated; conflict is ablated
                        pred_s_after, ds_after = pred_s_before, ds_before
                        pred_c_after, yc1, nc1, dc_after = score_yes_no_fast(model, tok, p_con)

                    st = stats[pos]
                    st["n"] += 1
                    st["sum_shift_support_before"] += (ds_before - db_before)
                    st["sum_shift_support_after"] += (ds_after - db_after)
                    st["sum_shift_conflict_before"] += (dc_before - db_before)
                    st["sum_shift_conflict_after"] += (dc_after - db_after)

                    if pred_b_before == gold:
                        st["base_correct_before"] += 1
                    if pred_b_after == gold:
                        st["base_correct_after"] += 1
                    if dataset_base_pred == gold:
                        st["base_correct_dataset_ref"] += 1
                    if pred_s_before == gold:
                        st["support_correct_before"] += 1
                    if pred_s_after == gold:
                        st["support_correct_after"] += 1

                    if pred_c_before == conf:
                        st["follow_conflict_before"] += 1
                    if pred_c_after == conf:
                        st["follow_conflict_after"] += 1
                    if pred_c_after == pred_b_after:
                        st["resist_after"] += 1
                    if pred_c_after != pred_b_after:
                        st["flip_after"] += 1

                    out_rec = {
                        "pair_id": rec.get("pair_id"),
                        "side": side,
                        "position": pos,
                        "gold": gold,
                        "conflict_label": conf,
                        "base_before": {"pred": pred_b_before, "delta": db_before},
                        "base_after": {"pred": pred_b_after, "delta": db_after},
                        "support_before": {"pred": pred_s_before, "delta": ds_before},
                        "support_after": {"pred": pred_s_after, "delta": ds_after},
                        "conflict_before": {"pred": pred_c_before, "delta": dc_before},
                        "conflict_after": {"pred": pred_c_after, "delta": dc_after},
                        "shift_support_before": ds_before - db_before,
                        "shift_support_after": ds_after - db_after,
                        "shift_conflict_before": dc_before - db_before,
                        "shift_conflict_after": dc_after - db_after,
                    }
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    # Print per-position summary
    summary = {}
    for pos in positions:
        st = stats[pos]
        n = st["n"] if st["n"] else 1
        base_acc_before = st["base_correct_before"] / n
        base_acc_before_ref = st["base_correct_dataset_ref"] / n
        base_acc_after = st["base_correct_after"] / n
        support_acc_before = st["support_correct_before"] / n
        support_acc_after = st["support_correct_after"] / n
        follow_before = st["follow_conflict_before"] / n
        follow_after = st["follow_conflict_after"] / n
        summary[pos] = {
            "n_samples": st["n"],
            # "before" is aligned to kept_pairs filtering reference (original no-ablation pred in file).
            "base_acc_before": base_acc_before_ref,
            "base_acc_before_recomputed": base_acc_before,
            "base_acc_after": base_acc_after,
            "base_acc_delta": base_acc_after - base_acc_before_ref,
            "support_acc_before": support_acc_before,
            "support_acc_after": support_acc_after,
            "support_acc_delta": support_acc_after - support_acc_before,
            "follow_conflict_rate_before": follow_before,
            "follow_conflict_rate_after": follow_after,
            "follow_conflict_rate_delta": follow_after - follow_before,
            "resist_rate_after(pred_same_as_base_after)": st["resist_after"] / n,
            "flip_rate_after": st["flip_after"] / n,
            "mean_shift_support_before": st["sum_shift_support_before"] / n,
            "mean_shift_support_after": st["sum_shift_support_after"] / n,
            "mean_shift_support_delta": (st["sum_shift_support_after"] - st["sum_shift_support_before"]) / n,
            "mean_shift_conflict_before": st["sum_shift_conflict_before"] / n,
            "mean_shift_conflict_after": st["sum_shift_conflict_after"] / n,
            "mean_shift_conflict_delta": (st["sum_shift_conflict_after"] - st["sum_shift_conflict_before"]) / n,
        }

    summary_payload = {
        "meta": meta,
        "summary_by_position": summary,
    }
    if args.summary_out:
        summary_out_path = Path(args.summary_out)
    else:
        out_path = Path(args.out)
        summary_out_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out_path, "w", encoding="utf-8") as sf:
        json.dump(summary_payload, sf, ensure_ascii=False, indent=2)

    print({
        "saved": args.out,
        "summary_saved": str(summary_out_path),
        "head_file": str(head_file_path),
        "mask_scope": args.mask_scope,
        "positions": positions,
        "summary_by_position": summary
    })

    remove_hooks(hook_handles)

if __name__ == "__main__":
    main()
