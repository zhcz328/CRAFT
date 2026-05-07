import argparse
from pathlib import Path

from tqdm import tqdm

from yesno_utils import (
    NO,
    YES,
    decide_yes_no,
    get_field,
    get_prompt_profile,
    load_model,
    load_records,
    make_nc_prompt,
    make_support_prompt,
    normalize_context,
    sanitize_path_component,
    score_yes_no,
    write_json,
    write_jsonl,
)


def eval_and_filter_one_model(
    model_name,
    model_label,
    data,
    out_dir,
    q_field,
    c_field,
    y_field,
    id_field,
    limit,
    device_map,
    dtype,
    enable_thinking,
    nc_yn_tau,
    nc_unk_tau,
    cc_yn_tau,
    cc_unk_tau,
    prompt_profile,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_path_component(model_label or model_name)

    preds_path = out_dir / f"{base}.preds.jsonl"
    filtered_json_path = out_dir / f"{base}.filtered.json"
    filtered_jsonl_path = out_dir / f"{base}.filtered.jsonl"
    meta_path = out_dir / f"{base}.meta.json"

    model, tok = load_model(model_name, device_map=device_map, dtype_name=dtype)

    n_eval = 0
    skipped_non_yesno = 0
    nc_answered_correct = 0
    kept = 0
    filtered_rows = []
    pred_rows = []

    total_cap = limit if (limit and limit > 0) else len(data)
    for idx, ex in enumerate(tqdm(data, desc=f"Filter {model_label or model_name}", total=total_cap)):
        if limit > 0 and n_eval >= limit:
            break

        question = get_field(ex, q_field)
        context_raw = get_field(ex, c_field)
        gold = get_field(ex, y_field)
        sample_id = get_field(ex, id_field) if id_field else ex.get("id", idx)

        if question is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            skipped_non_yesno += 1
            continue

        context = normalize_context(context_raw)
        nc_prompt = make_nc_prompt(str(question), allow_unknown=False)
        support_prompt = make_support_prompt(
            str(question),
            context,
            gold,
            position="before_question",
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )

        nc_scores = score_yes_no(model, tok, nc_prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
        nc_pred, nc_extra = decide_yes_no(nc_scores)

        support_scores = score_yes_no(model, tok, support_prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
        support_pred, support_extra = decide_yes_no(support_scores)

        n_eval += 1
        nc_answered_correct += int(nc_pred == gold)

        keep = nc_pred == gold and support_pred == gold
        pred_rows.append(
            {
                "id": sample_id,
                "gold": gold,
                "question": question,
                "nc": nc_pred,
                "support": support_pred,
                "nc_scores": nc_extra,
                "support_scores": support_extra,
                "both_correct": bool(keep),
            }
        )
        if keep:
            kept += 1
            row = dict(ex)
            row["_filter_meta"] = {
                "id": sample_id,
                "gold": gold,
                "nc": nc_pred,
                "support": support_pred,
                "nc_scores": nc_extra,
                "support_scores": support_extra,
                "scoring_mode": "yes_no_only",
                "prompt_profile": prompt_profile,
                "unused_thresholds": {
                    "nc_yn_tau": float(nc_yn_tau),
                    "nc_unk_tau": float(nc_unk_tau),
                    "cc_yn_tau": float(cc_yn_tau),
                    "cc_unk_tau": float(cc_unk_tau),
                },
            }
            filtered_rows.append(row)

    write_jsonl(preds_path, pred_rows)
    write_json(filtered_json_path, filtered_rows)
    write_jsonl(filtered_jsonl_path, filtered_rows)

    meta = {
        "model": model_name,
        "N_eval": n_eval,
        "skipped_non_yesno": skipped_non_yesno,
        "kept_both_nc_support_correct": kept,
        "NC_unknown_rate": 0.0,
        "NC_coverage": 100.0 if n_eval else 0.0,
        "NC_conditional_accuracy": (100.0 * nc_answered_correct / n_eval) if n_eval else 0.0,
        "scoring_mode": "yes_no_only",
        "prompt_profile": prompt_profile,
        "unused_thresholds": {
            "nc_yn_tau": float(nc_yn_tau),
            "nc_unk_tau": float(nc_unk_tau),
            "cc_yn_tau": float(cc_yn_tau),
            "cc_unk_tau": float(cc_unk_tau),
        },
        "files": {
            "preds": str(preds_path),
            "filtered_json": str(filtered_json_path),
            "filtered_jsonl": str(filtered_jsonl_path),
            "meta": str(meta_path),
        },
    }
    write_json(meta_path, meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Filter PubMedQA data to yes/no rows where NC and support-context are both correct.")
    ap.add_argument("--data", default="data/train-00000-of-00001.parquet")
    ap.add_argument("--fmt", default="auto", choices=["auto", "json", "jsonl", "parquet"])
    ap.add_argument("--out_dir", default="data/filter_yesno")
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="pubid")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--model_labels", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--nc_yn_tau", type=float, default=2.0)
    ap.add_argument("--nc_unk_tau", type=float, default=0.0)
    ap.add_argument("--cc_yn_tau", type=float, default=2.0)
    ap.add_argument("--cc_unk_tau", type=float, default=0.0)
    args = ap.parse_args()

    data = load_records(args.data, fmt=args.fmt)
    all_meta = []
    if args.model_labels and len(args.model_labels) != len(args.models):
        raise SystemExit("--model_labels count must match --models count.")

    for idx, model_name in enumerate(args.models):
        model_label = args.model_labels[idx] if args.model_labels else model_name
        prompt_profile = get_prompt_profile(model_name, model_label)
        meta = eval_and_filter_one_model(
            model_name=model_name,
            model_label=model_label,
            data=data,
            out_dir=args.out_dir,
            q_field=args.q_field,
            c_field=args.c_field,
            y_field=args.y_field,
            id_field=args.id_field,
            limit=args.limit,
            device_map=args.device_map,
            dtype=args.dtype,
            enable_thinking=args.enable_thinking,
            nc_yn_tau=args.nc_yn_tau,
            nc_unk_tau=args.nc_unk_tau,
            cc_yn_tau=args.cc_yn_tau,
            cc_unk_tau=args.cc_unk_tau,
            prompt_profile=prompt_profile,
        )
        all_meta.append(meta)

    write_json(Path(args.out_dir) / "ALL_META.json", all_meta)
    for meta in all_meta:
        print(
            f"{meta['model']}\tN_eval={meta['N_eval']}\tkept={meta['kept_both_nc_support_correct']}"
            f"\tNC_cov={meta['NC_coverage']:.2f}\tNC_condAcc={meta['NC_conditional_accuracy']:.2f}"
        )


if __name__ == "__main__":
    main()
