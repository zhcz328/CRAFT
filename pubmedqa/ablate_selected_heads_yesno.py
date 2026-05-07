import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from yesno_utils import (
    NO,
    YES,
    derive_follow_label,
    get_field,
    get_prompt_profile,
    invert_yes_no,
    load_records,
    make_conflict_prompt,
    make_nc_prompt,
    normalize_context,
    score_yes_no,
)


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def _get_num_heads(model):
    cfg = getattr(model, "config", None)
    return getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)


def _find_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate transformer layers.")


def _find_self_attn_modules(model):
    layers = _find_layers(model)
    layer2attn = {}
    for idx, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer2attn[idx] = layer.self_attn
        elif hasattr(layer, "attn"):
            layer2attn[idx] = layer.attn
    if not layer2attn:
        raise RuntimeError("Cannot find self-attention modules.")
    return layer2attn


def _apply_head_specific_mask(attn_mask, n_heads, heads_to_mask, keep_mode):
    if attn_mask is None or (not torch.is_tensor(attn_mask)) or attn_mask.dim() != 4:
        return attn_mask
    bsz, hmask, tgt, src = attn_mask.shape
    if hmask == 1:
        mask = attn_mask.expand(bsz, n_heads, tgt, src).clone()
    elif hmask == n_heads:
        mask = attn_mask.clone()
    else:
        return attn_mask
    neg = torch.finfo(mask.dtype).min
    for head_idx in heads_to_mask:
        mask[:, head_idx, :, :] = neg
        if keep_mode == "bos":
            mask[:, head_idx, :, 0] = 0
        else:
            diag_len = min(tgt, src)
            diag = torch.arange(diag_len, device=mask.device)
            mask[:, head_idx, diag, diag] = 0
    return mask


def install_head_mask_hooks(model, layer2heads, keep_mode):
    layer2attn = _find_self_attn_modules(model)
    n_heads = _get_num_heads(model)
    handles = []
    for layer_idx, heads in layer2heads.items():
        if layer_idx not in layer2attn:
            continue
        attn_mod = layer2attn[layer_idx]
        heads = sorted(set(int(head) for head in heads))

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
            handle = attn_mod.register_forward_pre_hook(make_pre_hook(heads), with_kwargs=True)
        except TypeError:
            def make_pre_hook_no_kwargs(heads_local):
                def pre_hook_no_kwargs(module, inputs):
                    args = list(inputs)
                    if len(args) < 2:
                        return inputs
                    args[1] = _apply_head_specific_mask(args[1], n_heads, heads_local, keep_mode)
                    return tuple(args)
                return pre_hook_no_kwargs
            handle = attn_mod.register_forward_pre_hook(make_pre_hook_no_kwargs(heads))
        handles.append(handle)
    return handles


def remove_hooks(handles):
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


def load_selected_heads(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    items = obj.get("selected", obj) if isinstance(obj, dict) else obj
    layer2heads = {}
    for item in items:
        layer2heads.setdefault(int(item["layer"]), []).append(int(item["head"]))
    for key in list(layer2heads.keys()):
        layer2heads[key] = sorted(set(layer2heads[key]))
    return layer2heads


def sample_random_heads_like(layer2heads, model):
    total = sum(len(heads) for heads in layer2heads.values())
    all_pairs = [(layer_idx, head_idx) for layer_idx in range(len(_find_layers(model))) for head_idx in range(_get_num_heads(model))]
    chosen = random.sample(all_pairs, total)
    out = {}
    for layer_idx, head_idx in chosen:
        out.setdefault(layer_idx, []).append(head_idx)
    return {key: sorted(set(value)) for key, value in out.items()}


def init_counts():
    return {
        "N": 0,
        "base_correct": 0,
        "conflict_correct": 0,
        "resist": 0,
        "follow_conflict": 0,
        "base_correct_total": 0,
        "resist_given_base_correct": 0,
        "follow_given_base_correct": 0,
    }


def update_counts(counts, base_pred, conflict_pred, gold):
    follow_label = derive_follow_label(conflict_pred, gold)
    counts["N"] += 1
    counts["base_correct"] += int(base_pred == gold)
    counts["conflict_correct"] += int(conflict_pred == gold)
    counts["resist"] += int(follow_label == "resist")
    counts["follow_conflict"] += int(follow_label == "follow_conflict")
    if base_pred == gold:
        counts["base_correct_total"] += 1
        counts["resist_given_base_correct"] += int(follow_label == "resist")
        counts["follow_given_base_correct"] += int(follow_label == "follow_conflict")


def summarize(counts):
    total = counts["N"]
    base_correct_total = counts["base_correct_total"]
    return {
        "N": total,
        "Acc_base": pct(counts["base_correct"], total),
        "Acc_conflict": pct(counts["conflict_correct"], total),
        "resist_rate": pct(counts["resist"], total),
        "follow_conflict_rate": pct(counts["follow_conflict"], total),
        "resist_rate_given_base_correct": pct(counts["resist_given_base_correct"], base_correct_total),
        "follow_conflict_rate_given_base_correct": pct(counts["follow_given_base_correct"], base_correct_total),
        "base_correct_total": base_correct_total,
    }


def main():
    ap = argparse.ArgumentParser(description="Re-evaluate selected-head ablation on PubMedQA yes/no prompts.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet", "auto"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--selected_heads", required=True)
    ap.add_argument("--position", default="before_question", choices=["prefix", "before_question", "before_answer"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--mask_scope", default="ic_only", choices=["ic_only", "all"])
    ap.add_argument("--keep_mode", default="self", choices=["self", "bos"])
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="pubid")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_jsonl", default="")
    ap.add_argument("--out_summary", default="ablation_yesno_summary.json")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--random_ablate", action="store_true")
    ap.add_argument("--random_seed", type=int, default=0)
    args = ap.parse_args()
    prompt_profile = get_prompt_profile(args.model)

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
        random.seed(args.random_seed)
        layer2heads = sample_random_heads_like(layer2heads, model)

    data = load_records(args.data, fmt=args.fmt)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    out_file = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    if out_file:
        out_file.write(json.dumps({"_meta": {"layer2heads": layer2heads, "mask_scope": args.mask_scope}}, ensure_ascii=False) + "\n")

    baseline = init_counts()
    masked = init_counts()
    hooks_all = None

    for idx, ex in enumerate(tqdm(data, desc="Ablation yes/no", unit="ex")):
        question = get_field(ex, args.q_field)
        context_raw = get_field(ex, args.c_field)
        gold = get_field(ex, args.y_field)
        if question is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            continue

        context = normalize_context(context_raw)
        sample_id = get_field(ex, args.id_field) if args.id_field else ex.get("id", idx)
        base_prompt = make_nc_prompt(str(question), allow_unknown=False)
        conflict_prompt = make_conflict_prompt(
            str(question),
            context,
            gold,
            args.position,
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )

        if hooks_all is not None:
            remove_hooks(hooks_all)
            hooks_all = None

        base_scores = score_yes_no(model, tok, base_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        base_pred = YES if base_scores[YES] >= base_scores[NO] else NO
        conflict_scores = score_yes_no(model, tok, conflict_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
        conflict_pred = YES if conflict_scores[YES] >= conflict_scores[NO] else NO
        update_counts(baseline, base_pred, conflict_pred, gold)

        if args.mask_scope == "all":
            if hooks_all is None:
                hooks_all = install_head_mask_hooks(model, layer2heads, keep_mode=args.keep_mode)
            base_scores_masked = score_yes_no(model, tok, base_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
            base_pred_masked = YES if base_scores_masked[YES] >= base_scores_masked[NO] else NO
            conflict_scores_masked = score_yes_no(model, tok, conflict_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
            conflict_pred_masked = YES if conflict_scores_masked[YES] >= conflict_scores_masked[NO] else NO
        else:
            base_scores_masked = base_scores
            base_pred_masked = base_pred
            handles = install_head_mask_hooks(model, layer2heads, keep_mode=args.keep_mode)
            conflict_scores_masked = score_yes_no(model, tok, conflict_prompt, enable_thinking=args.enable_thinking, prompt_profile=prompt_profile)
            conflict_pred_masked = YES if conflict_scores_masked[YES] >= conflict_scores_masked[NO] else NO
            remove_hooks(handles)

        update_counts(masked, base_pred_masked, conflict_pred_masked, gold)

        if out_file:
            out_file.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "gold": gold,
                        "conflict_label": invert_yes_no(gold),
                        "baseline": {
                            "base": {"pred": base_pred, "scores": base_scores},
                            "conflict": {"pred": conflict_pred, "scores": conflict_scores},
                        },
                        "masked": {
                            "base": {"pred": base_pred_masked, "scores": base_scores_masked},
                            "conflict": {"pred": conflict_pred_masked, "scores": conflict_scores_masked},
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if hooks_all is not None:
        remove_hooks(hooks_all)
    if out_file:
        out_file.close()

    baseline_summary = summarize(baseline)
    masked_summary = summarize(masked)
    delta = {
        key: masked_summary[key] - baseline_summary[key]
        for key in masked_summary
        if isinstance(masked_summary[key], (int, float))
    }
    summary = {
        "meta": {
            "data": args.data,
            "model": args.model,
            "selected_heads": args.selected_heads,
            "position": args.position,
            "mask_scope": args.mask_scope,
            "keep_mode": args.keep_mode,
            "layer2heads": layer2heads,
        },
        "baseline": baseline_summary,
        "masked": masked_summary,
        "delta(masked-baseline)": delta,
    }
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
