import argparse
from pathlib import Path

from tqdm import tqdm

from yesno_utils import (
    NO,
    YES,
    decide_yes_no,
    derive_follow_label,
    get_field,
    get_prompt_profile,
    invert_yes_no,
    load_model,
    load_records,
    make_conflict_prompt,
    make_nc_prompt,
    make_support_prompt,
    normalize_context,
    sanitize_path_component,
    score_yes_no,
    write_json,
    write_jsonl,
)


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def eval_one_model_one_position(
    model,
    tok,
    model_name,
    model_label,
    data,
    out_root,
    position,
    q_field,
    c_field,
    y_field,
    id_field,
    enable_thinking,
    limit,
    prompt_profile,
):
    model_dir = Path(out_root) / sanitize_path_component(model_label or model_name) / position
    model_dir.mkdir(parents=True, exist_ok=True)
    preds_path = model_dir / "preds.jsonl"
    summary_path = model_dir / "summary.json"

    records = []
    base_correct = 0
    support_correct = 0
    conflict_correct = 0
    resist = 0
    follow_conflict = 0
    follow_when_base_correct = 0
    resist_when_base_correct = 0
    n = 0

    total_cap = limit if (limit and limit > 0) else len(data)
    for idx, ex in enumerate(tqdm(data, desc=f"{model_label or model_name} @ {position}", total=total_cap)):
        if limit > 0 and n >= limit:
            break

        question = get_field(ex, q_field)
        context_raw = get_field(ex, c_field)
        gold = get_field(ex, y_field)
        pair_id = get_field(ex, id_field) if id_field else ex.get("id", idx)
        if question is None or gold is None:
            continue
        gold = str(gold).strip().lower()
        if gold not in (YES, NO):
            continue

        context = normalize_context(context_raw)
        conflict_label = invert_yes_no(gold)
        base_prompt = make_nc_prompt(str(question), allow_unknown=False)
        support_prompt = make_support_prompt(
            str(question),
            context,
            gold,
            position,
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )
        conflict_prompt = make_conflict_prompt(
            str(question),
            context,
            gold,
            position,
            allow_unknown=False,
            prompt_profile=prompt_profile,
        )

        base_scores = score_yes_no(model, tok, base_prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
        base_pred, base_extra = decide_yes_no(base_scores)
        support_scores = score_yes_no(model, tok, support_prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
        support_pred, support_extra = decide_yes_no(support_scores)
        conflict_scores = score_yes_no(model, tok, conflict_prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
        conflict_pred, conflict_extra = decide_yes_no(conflict_scores)

        follow_label = derive_follow_label(conflict_pred, gold)
        base_is_correct = base_pred == gold
        support_is_correct = support_pred == gold
        conflict_is_correct = conflict_pred == gold

        n += 1
        base_correct += int(base_is_correct)
        support_correct += int(support_is_correct)
        conflict_correct += int(conflict_is_correct)
        resist += int(follow_label == "resist")
        follow_conflict += int(follow_label == "follow_conflict")
        if base_is_correct:
            resist_when_base_correct += int(follow_label == "resist")
            follow_when_base_correct += int(follow_label == "follow_conflict")

        records.append(
            {
                "pair_id": pair_id,
                "question": question,
                "gold": gold,
                "conflict_label": conflict_label,
                "position": position,
                "base_prompt": base_prompt,
                "support_prompt": support_prompt,
                "conflict_prompt": conflict_prompt,
                "base": {"pred": base_pred, "scores": base_extra},
                "support": {"pred": support_pred, "scores": support_extra},
                "conflict": {"pred": conflict_pred, "scores": conflict_extra},
                "follow_label": follow_label,
                "base_correct": base_is_correct,
                "support_correct": support_is_correct,
                "conflict_correct": conflict_is_correct,
            }
        )

    write_jsonl(preds_path, records)
    summary = {
        "model": model_name,
        "prompt_profile": prompt_profile,
        "position": position,
        "N": n,
        "Acc_base": pct(base_correct, n),
        "Acc_support": pct(support_correct, n),
        "Acc_conflict": pct(conflict_correct, n),
        "resist_rate": pct(resist, n),
        "follow_conflict_rate": pct(follow_conflict, n),
        "resist_rate_given_base_correct": pct(resist_when_base_correct, base_correct),
        "follow_conflict_rate_given_base_correct": pct(follow_when_base_correct, base_correct),
        "files": {"preds": str(preds_path), "summary": str(summary_path)},
    }
    write_json(summary_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate PubMedQA yes/no NC/support/conflict prompts across positions.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--fmt", default="json", choices=["json", "jsonl", "parquet", "auto"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--model_labels", nargs="*", default=[])
    ap.add_argument("--out_dir", default="result_all_positions_yesno")
    ap.add_argument("--q_field", default="question")
    ap.add_argument("--c_field", default="context")
    ap.add_argument("--y_field", default="final_decision")
    ap.add_argument("--id_field", default="pubid")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--positions", nargs="+", default=["prefix", "before_question", "before_answer"])
    args = ap.parse_args()

    data = load_records(args.data, fmt=args.fmt)
    all_summaries = []
    if args.model_labels and len(args.model_labels) != len(args.models):
        raise SystemExit("--model_labels count must match --models count.")

    for idx, model_name in enumerate(args.models):
        model_label = args.model_labels[idx] if args.model_labels else model_name
        prompt_profile = get_prompt_profile(model_name, model_label)
        model, tok = load_model(model_name, device_map=args.device_map, dtype_name=args.dtype)
        try:
            for position in args.positions:
                summary = eval_one_model_one_position(
                    model=model,
                    tok=tok,
                    model_name=model_name,
                    model_label=model_label,
                    data=data,
                    out_root=args.out_dir,
                    position=position,
                    q_field=args.q_field,
                    c_field=args.c_field,
                    y_field=args.y_field,
                    id_field=args.id_field,
                    enable_thinking=args.enable_thinking,
                    limit=args.limit,
                    prompt_profile=prompt_profile,
                )
                all_summaries.append(summary)
        finally:
            del model

    write_json(Path(args.out_dir) / "ALL_SUMMARIES.json", all_summaries)
    for summary in all_summaries:
        print(
            f"{summary['model']} @ {summary['position']} | N={summary['N']} | "
            f"Acc_base={summary['Acc_base']:.2f} | Acc_support={summary['Acc_support']:.2f} | "
            f"Acc_conflict={summary['Acc_conflict']:.2f} | follow={summary['follow_conflict_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
