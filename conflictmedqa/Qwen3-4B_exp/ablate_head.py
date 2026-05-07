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


def infer_default_head_file(model_name: str, positions: list[str]) -> Path:
    script_dir = Path(__file__).resolve().parent
    pos = positions[0] if positions else "prefix"
    model_l = (model_name or "").lower()

    base_dir = script_dir / "llama32_3b" if "llama32_3b" in model_l else script_dir
    candidate_roots = [
        base_dir / "result" / pos / "headscan_rounds_top30_inf",
        base_dir / "result" / pos / "headscan_rounds_top30",
        base_dir / "result" / pos / "headscan_rounds_v2",
    ]

    for root in candidate_roots:
        cand = root / "selected_heads.json"
        if cand.exists():
            return cand
    for root in candidate_roots:
        cand = root / "head_groups.json"
        if cand.exists():
            return cand

    return candidate_roots[0] / "selected_heads.json"

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

    # Align with eval scoring: compare both with/without leading space and
    # take the better-scoring single-token variant for each label.
    yes_tids = []
    no_tids = []
    for s in (" Yes", "Yes"):
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            yes_tids.append(ids[0])
    for s in (" No", "No"):
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            no_tids.append(ids[0])

    if not yes_tids or not no_tids:
        raise RuntimeError('Cannot score yes/no with single-token fast path under this tokenizer.')

    yes_lp = max(logprob_next_token(model, input_ids, attn, tid) for tid in set(yes_tids))
    no_lp  = max(logprob_next_token(model, input_ids, attn, tid) for tid in set(no_tids))
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
    ap.add_argument(
        "--head_file",
        "--head_groups",
        dest="head_file",
        default=None,
        help="path to selected_heads.json (preferred) or legacy head_groups.json; if omitted, auto-resolved from model+position",
    )
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
    head_file_path = Path(args.head_file) if args.head_file else infer_default_head_file(args.model, positions)
    if args.head_file is None:
        print(f"[info] --head_file not set, auto-selected: {head_file_path}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device_map
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
        "head_file": str(head_file_path),
        "mask_scope": args.mask_scope,
        "positions": positions,
        "summary_by_position": summary
    })

    remove_hooks(hook_handles)

if __name__ == "__main__":
    main()
