import json
import re
from pathlib import Path

YES = "yes"
NO = "no"
SYSTEM = "You are a medical QA verifier. Answer ONLY with 'Yes' or 'No'. No other words."
EVID_TMPL = """EVIDENCE (may be incorrect):
- According to current clinical guidelines, this recommendation {align} with guidelines.
END EVIDENCE
"""
FOLLOW_MAP = {
    "resist": 0,
    "follow_conflict": 1,
    "unknown": 2,
}


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_padding_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token or eos_token to use for padding.")
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def gold_label_for_side(side):
    if side == "correct":
        return YES
    if side == "wrong":
        return NO
    raise ValueError(f"Unexpected side: {side}")


def opposite_label(label):
    if label == YES:
        return NO
    if label == NO:
        return YES
    raise ValueError(f"Unexpected label: {label}")


def make_evidence(label_yes):
    align = "DOES align" if label_yes else "DOES NOT align"
    return EVID_TMPL.format(align=align).strip()


def inject_evidence(base_prompt, evidence, position):
    ev = evidence.strip() + "\n"
    if position == "prefix":
        return ev + "\n" + base_prompt

    if position == "suffix":
        return base_prompt.rstrip() + "\n\n" + ev

    if position == "before_question":
        matches = list(re.finditer(r"(?im)^\s*question\s*:", base_prompt))
        if not matches:
            return ev + "\n" + base_prompt
        idx = matches[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    if position == "before_answer":
        matches = list(re.finditer(r"(?im)^\s*answer\s*:", base_prompt))
        if not matches:
            return ev + "\n" + base_prompt
        idx = matches[-1].start()
        return base_prompt[:idx].rstrip() + "\n\n" + ev + "\n" + base_prompt[idx:]

    raise ValueError(f"Unknown position: {position}")


def wrap_as_chat(tokenizer, user_text, enable_thinking=False):
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


def derive_follow_label(position_record):
    if not position_record:
        return "unknown"
    conflict_pred = position_record["conflict"]["pred"]
    gold = position_record["gold"]
    conflict_label = position_record["conflict_label"]
    if conflict_pred == conflict_label:
        return "follow_conflict"
    if conflict_pred == gold:
        return "resist"
    return "unknown"


def load_outcome_map(conflict_positions_path, position):
    if not conflict_positions_path:
        return {}
    rows = read_jsonl(conflict_positions_path)
    outcome_map = {}
    for row in rows:
        if row.get("position") != position:
            continue
        key = (row.get("pair_id"), row.get("side"))
        outcome_map[key] = row
    return outcome_map
