import csv
import json
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parents[1]
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from mask_utils import UNKNOWN_ANSWER, is_unknown_text


HALLUCINATION_MAP = {
    "unknown": 0,
    "hallucination": 1,
}

# Backward-compatible alias for older code paths.
FOLLOW_MAP = HALLUCINATION_MAP

SYSTEM_PROMPT = "You are a helpful medical QA assistant."
BASE_RULE = (
    "Answer the question using your best internal medical knowledge AND the image.\n"
    "If the image does not contain enough information to answer the question, output ONLY unknown.\n"
    "Output ONLY the final answer.\n\n"
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


def make_prompt_no_evidence(question):
    return make_base_prompt(question)


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
    ic_pred = norm_text(outcome_record.get("conflict_pred"))
    wrong = norm_text(wrong_answer)
    # For image-conflict, hallucination means the masked-image answer is not unknown.
    if outcome_record.get("ic_is_unknown") or ic_pred == wrong or is_unknown_text(ic_pred):
        return "unknown"
    return "hallucination"


def default_preds_path(project_root, position, model_name="", model="hulumed-4b"):
    from model_registry import resolve_model_selection

    _, spec = resolve_model_selection(model_name=model_name, model=model)
    root = Path(project_root) / spec.slug / "eval-results_slake_image_conflict"
    direct = root / "preds.jsonl"
    if direct.exists():
        return direct
    recursive_matches = sorted(root.glob("**/preds.jsonl"), key=lambda p: (len(str(p)), str(p)))
    if recursive_matches:
        return recursive_matches[0]
    return direct


def load_outcome_map(preds_jsonl_path):
    if not preds_jsonl_path or not Path(preds_jsonl_path).exists():
        return {}
    rows = read_jsonl(preds_jsonl_path)
    outcome_map = {}
    for row in rows:
        key = sample_key(row.get("img_id", ""), row.get("question", ""))
        ic_pred = norm_text(row.get("ic_pred_norm") or row.get("ic_pred_raw"))
        outcome_map[key] = {
            "img_id": row.get("img_id"),
            "question": row.get("question"),
            "gold": norm_text(row.get("gold")),
            "wrong": norm_text(row.get("unknown_target", UNKNOWN_ANSWER)),
            "nc_pred": norm_text(row.get("nc_pred_norm") or row.get("nc_pred_raw")),
            "support_pred": norm_text(row.get("nc_pred_norm") or row.get("nc_pred_raw")),
            "conflict_pred": ic_pred,
            "ic_pred": ic_pred,
            "follow_conflict": bool(row.get("ic_is_unknown", False)),
            "ic_is_unknown": bool(row.get("ic_is_unknown", False)),
            "resist": bool(row.get("ic_matches_gold", False)),
            "ic_matches_gold": bool(row.get("ic_matches_gold", False)),
            "changed_vs_nc": bool(row.get("changed_vs_nc", False)),
            "position": row.get("position", "image_conflict"),
        }
    return outcome_map
