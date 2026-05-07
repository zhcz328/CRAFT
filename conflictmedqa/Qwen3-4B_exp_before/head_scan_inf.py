# head_scan.py
import json, re, argparse, os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

YES, NO = "yes", "no"
SYSTEM = "You are a medical QA verifier. Answer ONLY with 'Yes' or 'No'. No other words."
EVID_TMPL = """EVIDENCE (may be incorrect):
- According to current clinical guidelines, this recommendation {align} with guidelines.
END EVIDENCE
"""

def read_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs

def make_evidence(label_yes: bool):
    align = "DOES align" if label_yes else "DOES NOT align"
    return EVID_TMPL.format(align=align).strip()

def inject_evidence(base_prompt: str, evidence: str, position: str):
    ev = evidence.strip() + "\n"
    if position == "before_question":
        m = list(re.finditer(r"(?im)^\s*question\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]
    if position == "prefix":
        return ev + "\n" + base_prompt
    if position == "before_answer":
        m = list(re.finditer(r"(?im)^\s*answer\s*:", base_prompt))
        if not m:
            return ev + "\n" + base_prompt
        idx = m[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]
    raise ValueError(position)

def wrap_as_chat(tok, user_text: str, enable_thinking: bool):
    msgs = [{"role":"system","content":SYSTEM},{"role":"user","content":user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def delta_yes_no(model, tok, prompt_text: str):
    dev = next(model.parameters()).device
    enc = tok(prompt_text, return_tensors="pt")
    enc = {k:v.to(dev) for k,v in enc.items()}
    out = model(**enc)
    logits = out.logits[:, -1, :]
    logp = torch.log_softmax(logits, dim=-1)
    yid = tok(" Yes", add_special_tokens=False).input_ids
    nid = tok(" No",  add_special_tokens=False).input_ids
    if len(yid)!=1 or len(nid)!=1:
        raise RuntimeError('" Yes"/" No" not single-token; need continuation scoring.')
    return (logp[0, yid[0]] - logp[0, nid[0]]).item()

def oproj_head_ablate_hook(head_idx: int, head_dim: int):
    """DEPRECATED: kept for backward compatibility.

    Previously, head ablation was implemented by zeroing the corresponding head slice
    in the input to o_proj. We now use an attention-logits masking method (pre-softmax)
    implemented via attention_mask modification (see install_single_head_mask_hook).
    """
    def hook(module, inputs):
        return inputs
    return hook


def _apply_head_specific_mask(attn_mask, n_heads: int, heads_to_mask, keep_mode: str = "self"):
    """Apply head-specific additive mask BEFORE softmax, matching ablate_head_inf.py.

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
        # Cannot do head-specific masking if mask is not already expanded to 4D.
        return attn_mask

    B, Hm, T, S = attn_mask.shape

    # Expand to per-head mask if needed
    if Hm == 1:
        mask = attn_mask.expand(B, n_heads, T, S).clone()
    elif Hm == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask

    neg = torch.finfo(mask.dtype).min

    for h in heads_to_mask:
        if not (0 <= int(h) < n_heads):
            continue
        h = int(h)
        mask[:, h, :, :] = neg

        # Keep at least one key position to avoid all -inf => NaN
        if keep_mode == "bos":
            mask[:, h, :, 0] = 0
        else:  # "self"
            diag_len = min(T, S)
            idx = torch.arange(diag_len, device=mask.device)
            mask[:, h, idx, idx] = 0

    return mask


def install_single_head_mask_hook(attn_mod, n_heads: int, head_idx: int, keep_mode: str = "self"):
    """Install a forward-pre-hook on a layer's self-attn module to ablate one head.

    This matches the masking method used in ablate_head_inf.py:
      - modify attention_mask additive bias before softmax
      - for the selected head, mask almost all keys with a very negative bias
      - keep at least one allowed key (self diagonal or BOS) to avoid NaNs
    """
    heads_local = [int(head_idx)]

    def pre_hook(module, args, kwargs):
        attn_mask = None
        if kwargs is not None and "attention_mask" in kwargs:
            attn_mask = kwargs["attention_mask"]
        elif len(args) >= 2:
            attn_mask = args[1]

        new_mask = _apply_head_specific_mask(attn_mask, n_heads, heads_local, keep_mode=keep_mode)

        if new_mask is attn_mask:
            return args, kwargs

        if kwargs is not None and "attention_mask" in kwargs:
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = new_mask
            return args, kwargs
        else:
            args = list(args)
            if len(args) >= 2:
                args[1] = new_mask
            return tuple(args), kwargs

    try:
        return attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
    except TypeError:
        # Older PyTorch without with_kwargs
        def pre_hook_no_kwargs(module, inputs):
            args = list(inputs)
            if len(args) < 2:
                return inputs
            args[1] = _apply_head_specific_mask(args[1], n_heads, heads_local, keep_mode=keep_mode)
            return tuple(args)
        return attn_mod.register_forward_pre_hook(pre_hook_no_kwargs)

def parse_layers_arg(layers_str: str):
    # supports "1,2,3" only (keep consistent with your original)
    return [int(x.strip()) for x in layers_str.split(",") if x.strip()]

def load_plan_layers(plan_path, round_name=None):
    """
    plan schema expected:
      {
        "rounds": [{"name": "...", "layers": [..], ...}, ...],
        ...
      }
    """
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    rounds = plan.get("rounds", [])
    if not rounds:
        raise ValueError("plan file missing 'rounds' or rounds is empty")

    if round_name is None:
        return plan, [(r["name"], r["layers"]) for r in rounds]
    for r in rounds:
        if r.get("name") == round_name:
            return plan, [(r["name"], r["layers"])]
    raise ValueError(f"round name not found in plan: {round_name}")

def run_head_scan_round(model, tok, layers_mod, n_heads,
                        samples, scan_layers, out_path):
    """
    samples: list of tuples (d_base, d_con, p_base, p_con)
    """
    results = []

    for layer_idx in scan_layers:
        attn = layers_mod[layer_idx].self_attn

        for h in tqdm(range(n_heads), desc=f"Scan L{layer_idx}", unit="head"):
            handle = install_single_head_mask_hook(attn, n_heads=n_heads, head_idx=h, keep_mode="self")

            eff_red_sum = 0.0
            base_damage_sum = 0.0
            n = 0

            for d_base, d_con, p_base, p_con in samples:
                # ablated runs
                d_con_ab = delta_yes_no(model, tok, p_con)
                d_base_ab = delta_yes_no(model, tok, p_base)

                # original effect
                effect = d_con - d_base
                # ablated effect: keep BASE (unablated) as reference for effect reduction, consistent with your design
                effect_ab = d_con_ab - d_base

                eff_red_sum += (abs(effect) - abs(effect_ab))
                base_damage_sum += abs(d_base_ab - d_base)
                n += 1

            handle.remove()

            results.append({
                "layer": layer_idx,
                "head": h,
                "mean_abs_effect_reduction": eff_red_sum / max(n,1),
                "mean_abs_base_delta_change": base_damage_sum / max(n,1),
            })

    results.sort(key=lambda r: (-(r["mean_abs_effect_reduction"]), r["mean_abs_base_delta_change"]))

    out = {
        "scan_layers": scan_layers,
        "n_samples": len(samples),
        "top20": results[:20],
        "saved": out_path,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _quantile(vals, q: float):
    """Simple inclusive quantile without numpy."""
    if not vals:
        return None
    xs = sorted(vals)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize_head_groups(round_outputs, out_path: str, topk: int = 10,
                          cs_eff_min: float | None = None,
                          cs_base_max: float | None = None,
                          bb_eff_min: float | None = None,
                          bb_base_min: float | None = None):
    """Summarize heads into conflict-specific vs backbone groups from round outputs.

    round_outputs schema:
      [{"round": <name>, "result": {"top20": [...]}}, ...]

    If cs_eff_min/cs_base_max are provided, we use fixed thresholds:
      - conflict_specific: effect_reduction > cs_eff_min AND base_delta_change < cs_base_max
      - backbone:         effect_reduction > bb_eff_min AND base_delta_change > bb_base_min
        (bb_eff_min defaults to cs_eff_min)

    Otherwise, we fall back to a quantile-based heuristic on the candidate set.
    """
    # Collect candidates from union of per-round top20
    raw = []
    for r in round_outputs:
        rname = r.get("round")
        res = r.get("result") or {}
        for it in (res.get("top20") or []):
            er = float(it.get("mean_abs_effect_reduction", 0.0))
            bc = float(it.get("mean_abs_base_delta_change", 0.0))
            raw.append({
                "round": rname,
                "layer": int(it.get("layer")),
                "head": int(it.get("head")),
                "mean_abs_effect_reduction": er,
                "mean_abs_base_delta_change": bc,
            })

    if not raw:
        out = {
            "note": "no candidates found (top20 empty); nothing to summarize",
            "conflict_specific": [],
            "backbone": [],
            "saved": out_path,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    # De-dup by (layer, head): keep the best entry (highest effect_reduction; tie -> lower base change)
    dedup = {}
    for x in raw:
        k = (x["layer"], x["head"])
        if k not in dedup:
            dedup[k] = {**x, "rounds": [x["round"]]}
        else:
            dedup[k]["rounds"].append(x["round"])
            cur = dedup[k]
            better = (
                x["mean_abs_effect_reduction"] > cur["mean_abs_effect_reduction"]
                or (
                    x["mean_abs_effect_reduction"] == cur["mean_abs_effect_reduction"]
                    and x["mean_abs_base_delta_change"] < cur["mean_abs_base_delta_change"]
                )
            )
            if better:
                rounds = cur["rounds"]
                dedup[k] = {**x, "rounds": rounds}

    entries = list(dedup.values())

    # Helper scores for ranking
    for e in entries:
        er = e["mean_abs_effect_reduction"]
        bc = e["mean_abs_base_delta_change"]
        e["conflict_specific_score"] = er / (bc + 1e-6)   # larger => stronger + less base damage
        e["backbone_score"] = er * bc                     # larger => strong and base-involved

    fixed_mode = (cs_eff_min is not None) and (cs_base_max is not None)

    if fixed_mode:
        if bb_eff_min is None:
            bb_eff_min = cs_eff_min

        conflict = [
            e for e in entries
            if e["mean_abs_effect_reduction"] > cs_eff_min and e["mean_abs_base_delta_change"] < cs_base_max and e["mean_abs_effect_reduction"] > e["mean_abs_base_delta_change"]
        ]
        conflict_sorted = sorted(conflict, key=lambda x: (-x["conflict_specific_score"],
                                                          -x["mean_abs_effect_reduction"],
                                                          x["mean_abs_base_delta_change"]))

        backbone_sorted = []
        if bb_base_min is not None:
            backbone = [
                e for e in entries
                if e["mean_abs_effect_reduction"] > bb_eff_min and e["mean_abs_base_delta_change"] > bb_base_min
            ]
            backbone_sorted = sorted(backbone, key=lambda x: (-x["backbone_score"],
                                                             -x["mean_abs_base_delta_change"],
                                                             -x["mean_abs_effect_reduction"]))

        out = {
            "candidates_source": "union_of_per_round_top20_dedup_by_(layer,head)",
            "n_candidates": len(entries),
            "fixed_thresholds": {
                "conflict_specific": {
                    "effect_reduction_gt": cs_eff_min,
                    "base_delta_change_lt": cs_base_max,
                },
                "backbone": None if bb_base_min is None else {
                    "effect_reduction_gt": bb_eff_min,
                    "base_delta_change_gt": bb_base_min,
                }
            },
            "topk": topk,
            "conflict_specific": conflict_sorted[:topk],
            "backbone": backbone_sorted[:topk],
            "saved": out_path,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return out

    # --------- fallback: quantile heuristic (previous behavior) ----------
    er_vals = [e["mean_abs_effect_reduction"] for e in entries]
    bc_vals = [e["mean_abs_base_delta_change"] for e in entries]
    er_p75 = _quantile(er_vals, 0.75)
    bc_p25 = _quantile(bc_vals, 0.25)
    bc_p75 = _quantile(bc_vals, 0.75)

    conflict = [e for e in entries if e["mean_abs_effect_reduction"] >= er_p75 and e["mean_abs_base_delta_change"] <= bc_p25]
    backbone = [e for e in entries if e["mean_abs_effect_reduction"] >= er_p75 and e["mean_abs_base_delta_change"] >= bc_p75]

    conflict_sorted = sorted(conflict, key=lambda x: (-x["conflict_specific_score"],
                                                      -x["mean_abs_effect_reduction"],
                                                      x["mean_abs_base_delta_change"]))
    backbone_sorted = sorted(backbone, key=lambda x: (-x["backbone_score"],
                                                     -x["mean_abs_base_delta_change"],
                                                     -x["mean_abs_effect_reduction"]))

    if len(conflict_sorted) < min(5, topk):
        conflict_sorted = sorted(entries, key=lambda x: (-x["conflict_specific_score"],
                                                         -x["mean_abs_effect_reduction"],
                                                         x["mean_abs_base_delta_change"]))
    if len(backbone_sorted) < min(5, topk):
        backbone_sorted = sorted(entries, key=lambda x: (-x["backbone_score"],
                                                        -x["mean_abs_base_delta_change"],
                                                        -x["mean_abs_effect_reduction"]))

    out = {
        "candidates_source": "union_of_per_round_top20_dedup_by_(layer,head)",
        "n_candidates": len(entries),
        "thresholds": {
            "effect_reduction_p75": er_p75,
            "base_delta_change_p25": bc_p25,
            "base_delta_change_p75": bc_p75,
        },
        "topk": topk,
        "conflict_specific": conflict_sorted[:topk],
        "backbone": backbone_sorted[:topk],
        "saved": out_path,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--position", default="before_question", choices=["prefix","before_question","before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")

    # OLD MODE (manual)
    ap.add_argument("--layers", default=None, help="Comma-separated layer indices, e.g. 32,33,34,35 (used if --plan not set)")

    ap.add_argument("--max_pairs", type=int, default=106)

    # OLD single output (manual mode) OR used as default root in plan mode
    ap.add_argument("--out", default="head_scan.json")

    # PLAN MODE
    ap.add_argument("--plan", default=None, help="scan_plan.json from layer_trace.py")
    ap.add_argument("--round", default=None, help="run only this round name from plan; default runs all rounds")
    ap.add_argument("--plan_out_dir", default="result/headscan_rounds", help="save per-round outputs here")

    # Summary of conflict-specific vs backbone heads
    ap.add_argument("--groups_out", default=None, help="save head group summary json here (default: alongside summary)")
    ap.add_argument("--groups_topk", type=int, default=50, help="how many heads to keep per group")

    # Fixed-threshold grouping (optional; if not set, falls back to quantile heuristic)
    ap.add_argument("--cs_eff_min", type=float, default=None, help="conflict-specific: mean_abs_effect_reduction > X")
    ap.add_argument("--cs_base_max", type=float, default=None, help="conflict-specific: mean_abs_base_delta_change < Y")
    ap.add_argument("--bb_eff_min", type=float, default=None, help="backbone: mean_abs_effect_reduction > X (default: cs_eff_min)")
    ap.add_argument("--bb_base_min", type=float, default=None, help="backbone: mean_abs_base_delta_change > Z")

    # dtype
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()

    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None)
    hidden  = getattr(cfg, "hidden_size", None)
    if n_heads is None or hidden is None:
        raise RuntimeError("Missing num_attention_heads/hidden_size in config")
    head_dim = hidden // n_heads

    layers_mod = model.model.layers if hasattr(model, "model") else model.layers

    # ---------- build samples (baseline/conflict deltas cached) ----------
    pairs = read_pairs(args.pairs)[:args.max_pairs]

    samples = []
    for rec in pairs:
        for side in ["correct","wrong"]:
            base_prompt = rec[side]["prompt"]
            gold_yes = (side == "correct")
            conflict_yes = (not gold_yes)

            p_base = wrap_as_chat(tok, base_prompt, args.enable_thinking)
            conflict_user = inject_evidence(base_prompt, make_evidence(conflict_yes), args.position)
            p_con = wrap_as_chat(tok, conflict_user, args.enable_thinking)

            d_base = delta_yes_no(model, tok, p_base)
            d_con  = delta_yes_no(model, tok, p_con)
            samples.append((d_base, d_con, p_base, p_con))

    # ---------- choose scan layers: plan mode OR manual ----------
    if args.plan:
        plan, round_layers = load_plan_layers(args.plan, args.round)
        os.makedirs(args.plan_out_dir, exist_ok=True)

        all_rounds = []
        for rname, scan_layers in round_layers:
            out_path = os.path.join(args.plan_out_dir, f"head_scan_{rname}.json")
            r_out = run_head_scan_round(
                model=model,
                tok=tok,
                layers_mod=layers_mod,
                n_heads=n_heads,
                samples=samples,
                scan_layers=scan_layers,
                out_path=out_path
            )
            all_rounds.append({"round": rname, "result": r_out})

        summary_path = os.path.join(args.plan_out_dir, "head_scan_summary.json")
        summary = {
            "position": args.position,
            "enable_thinking": bool(args.enable_thinking),
            "dtype": args.dtype,
            "n_pairs": len(pairs),
            "n_samples": len(samples),
            "plan": args.plan,
            "rounds_ran": [x["round"] for x in all_rounds],
            "round_outputs": all_rounds,
            "saved": summary_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        groups_path = args.groups_out or os.path.join(args.plan_out_dir, "head_groups.json")
        summarize_head_groups(all_rounds, groups_path, topk=args.groups_topk, cs_eff_min=args.cs_eff_min, cs_base_max=args.cs_base_max, bb_eff_min=args.bb_eff_min, bb_base_min=args.bb_base_min)

        print(summary)
        return

    # manual mode
    if not args.layers:
        raise ValueError("Either --plan must be provided, or --layers must be set in manual mode.")
    scan_layers = parse_layers_arg(args.layers)

    out = run_head_scan_round(
        model=model,
        tok=tok,
        layers_mod=layers_mod,
        n_heads=n_heads,
        samples=samples,
        scan_layers=scan_layers,
        out_path=args.out
    )
    out.update({
        "position": args.position,
        "enable_thinking": bool(args.enable_thinking),
        "dtype": args.dtype,
        "n_pairs": len(pairs),
    })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # also write head group summary in manual mode
    if args.groups_out is None:
        root, ext = os.path.splitext(args.out)
        groups_path = root + "_head_groups.json"
    else:
        groups_path = args.groups_out
    summarize_head_groups([{"round": "manual", "result": out}], groups_path, topk=args.groups_topk, cs_eff_min=args.cs_eff_min, cs_base_max=args.cs_base_max, bb_eff_min=args.bb_eff_min, bb_base_min=args.bb_base_min)

    print(out)

if __name__ == "__main__":
    main()
