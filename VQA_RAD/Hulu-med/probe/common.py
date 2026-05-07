import csv
import json
from pathlib import Path

FOLLOW_MAP = {
    "resist": 0,
    "follow_conflict": 1,
    "unknown": 2,
}

SYSTEM_PROMPT = "You are a helpful medical QA assistant."
BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)
EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {ans}.\n"
    "END EVIDENCE\n"
)


def norm_text(text):
    return str(text or "").strip().lower()


def sample_key(img_id, question):
    return f"{str(img_id).strip()}||{norm_text(question)}"


def read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


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


def make_base_prompt(question):
    return BASE_RULE + f"Question: {question}\nAnswer:"


def inject_evidence(base_prompt, evidence_block, position):
    if position == "prefix":
        return evidence_block + "\n" + base_prompt

    if position == "before_question":
        marker = "\nQuestion:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nQuestion:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    if position == "before_answer":
        marker = "\nAnswer:"
        if marker not in base_prompt:
            raise ValueError("base_prompt missing '\\nAnswer:' marker")
        return base_prompt.replace(marker, "\n" + evidence_block + marker, 1)

    raise ValueError(f"Unknown position: {position}")


def make_prompt_no_evidence(question):
    return make_base_prompt(question)


def make_prompt_with_evidence(question, evidence_answer, position):
    base_prompt = make_base_prompt(question)
    evidence_block = EVIDENCE_TMPL.format(ans=str(evidence_answer).strip())
    return inject_evidence(base_prompt, evidence_block, position)


def build_messages_multimodal(image, prompt_text):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def derive_follow_label(outcome_record, gold_answer, wrong_answer):
    if not outcome_record:
        return "unknown"
    conflict_pred = norm_text(outcome_record.get("conflict_pred"))
    gold = norm_text(gold_answer)
    wrong = norm_text(wrong_answer)
    if conflict_pred == wrong:
        return "follow_conflict"
    if conflict_pred == gold:
        return "resist"
    return "unknown"


def default_preds_path(project_root, position):
    candidates = {
        "prefix": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_root_autodl-tmp_hulumed-4B_Hulu-Med-4B"
        / "prefix"
        / "preds.jsonl",
        "before_question": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_archive_zengjiaqi_Medical_LLM_Hulu-Med-4B"
        / "before_question"
        / "preds.jsonl",
        "before_answer": project_root
        / "eval-results_vqarad_hulumed4b_all"
        / "_root_autodl-tmp_hulumed-4B_Hulu-Med-4B"
        / "before_answer"
        / "preds.jsonl",
    }
    return candidates[position]


def load_outcome_map(preds_jsonl_path):
    if not preds_jsonl_path or not Path(preds_jsonl_path).exists():
        return {}
    rows = read_jsonl(preds_jsonl_path)
    outcome_map = {}
    for row in rows:
        key = sample_key(row.get("img_id", ""), row.get("question", ""))
        outcome_map[key] = {
            "img_id": row.get("img_id"),
            "question": row.get("question"),
            "gold": norm_text(row.get("gold")),
            "wrong": norm_text(row.get("conflict")),
            "nc_pred": norm_text(row.get("nc_pred")),
            "support_pred": norm_text(row.get("support_pred")),
            "conflict_pred": norm_text(row.get("conflict_pred")),
            "follow_conflict": bool(row.get("follow_conflict", False)),
            "resist": bool(row.get("resist", False)),
            "changed_vs_nc": bool(row.get("changed_vs_nc", False)),
            "position": row.get("position"),
        }
    return outcome_map

