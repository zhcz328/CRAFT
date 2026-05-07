import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

YES = "yes"
NO = "no"
UNKNOWN = "unknown"
LABELS = (YES, NO, UNKNOWN)
SYSTEM_PROMPT = "You are a biomedical QA assistant. Answer exactly with yes or no when possible."
LLAMA_SYSTEM_PROMPT = "You are a helpful medical QA assistant."

NC_RULE_YESNO = (
    "Answer the biomedical yes/no question using ONLY your internal knowledge.\n"
    "Output one token only from: yes / no.\n\n"
)
NC_RULE_FILTER = (
    "Answer the biomedical yes/no question using ONLY your internal knowledge.\n"
    "If you cannot answer with high confidence, answer: unknown.\n"
    "Output one token only from: yes / no / unknown.\n\n"
)
CTX_RULE_YESNO = (
    "Answer the biomedical yes/no question using the question, the provided context, and the evidence block.\n"
    "The evidence block may be correct or incorrect.\n"
    "Output one token only from: yes / no.\n\n"
)
LLAMA_CTX_RULE_YESNO = (
    "Answer the question using your best internal medical knowledge AND the context.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "Output ONLY the final answer.\n\n"
)
CTX_RULE_FILTER = (
    "Answer the biomedical yes/no question using the question, the provided context, and the evidence block.\n"
    "The evidence block may be correct or incorrect.\n"
    "If you cannot answer with high confidence, answer: unknown.\n"
    "Output one token only from: yes / no / unknown.\n\n"
)
LLAMA_CTX_RULE_FILTER = (
    "Answer the question using your best internal medical knowledge AND the context.\n"
    "The EVIDENCE block (if present) may be correct or incorrect.\n"
    "If you cannot answer with high confidence, answer: unknown.\n"
    "Output ONLY the final answer.\n\n"
)
EVIDENCE_TMPL = (
    "EVIDENCE (may be incorrect):\n"
    "- Based on the provided context, the answer should be: {answer}.\n"
    "END EVIDENCE\n"
)
LLAMA_EVIDENCE_TMPL = (
    "EVIDENCE:\n"
    "- According to the provided context, the correct answer is: {answer}.\n"
    "END EVIDENCE\n"
)


def get_prompt_profile(model_name: str = "", model_label: str = "") -> str:
    text = f"{model_name} {model_label}".lower()
    return "llama" if "llama" in text else "default"


def get_system_prompt(prompt_profile: str = "default") -> str:
    return LLAMA_SYSTEM_PROMPT if prompt_profile == "llama" else SYSTEM_PROMPT


def sanitize_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_field(ex: dict[str, Any], key: str) -> Any:
    cur: Any = ex
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_context(ctx: Any) -> str:
    if ctx is None:
        return ""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, list):
        return "\n".join(str(x) for x in ctx)
    if isinstance(ctx, dict):
        if "contexts" in ctx and isinstance(ctx["contexts"], list):
            return "\n".join(str(x) for x in ctx["contexts"])
        return json.dumps(ctx, ensure_ascii=False)
    return str(ctx)


def load_records(path: str, fmt: str = "auto") -> list[dict[str, Any]]:
    input_path = Path(path)
    actual_fmt = fmt
    if actual_fmt == "auto":
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            actual_fmt = "json"
        elif suffix == ".jsonl":
            actual_fmt = "jsonl"
        elif suffix == ".parquet":
            actual_fmt = "parquet"
        else:
            raise ValueError(f"Cannot infer format from suffix: {input_path}")

    if actual_fmt == "jsonl":
        return read_jsonl(input_path)

    if actual_fmt == "json":
        obj = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            rows = []
            for key, value in obj.items():
                if isinstance(value, dict):
                    rows.append({"id": key, **value})
                else:
                    rows.append({"id": key, "value": value})
            return rows
        raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")

    if actual_fmt == "parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(input_path)
        return table.to_pylist()

    raise ValueError(f"Unsupported format: {actual_fmt}")


def invert_yes_no(label: str) -> str:
    token = str(label).strip().lower()
    if token == YES:
        return NO
    if token == NO:
        return YES
    raise ValueError(f"Cannot invert label: {label}")


def make_evidence(answer: str, prompt_profile: str = "default") -> str:
    tmpl = LLAMA_EVIDENCE_TMPL if prompt_profile == "llama" else EVIDENCE_TMPL
    return tmpl.format(answer=str(answer).strip().lower()).strip()


def inject_evidence(base_prompt: str, evidence: str, position: str) -> str:
    ev = evidence.strip() + "\n"
    if position == "prefix":
        return ev + "\n" + base_prompt

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


def make_nc_prompt(question: str, allow_unknown: bool) -> str:
    rule = NC_RULE_FILTER if allow_unknown else NC_RULE_YESNO
    return rule + f"Question: {question}\nAnswer:"


def make_context_base_prompt(
    question: str,
    context: str,
    allow_unknown: bool,
    prompt_profile: str = "default",
) -> str:
    if prompt_profile == "llama":
        rule = LLAMA_CTX_RULE_FILTER if allow_unknown else LLAMA_CTX_RULE_YESNO
    else:
        rule = CTX_RULE_FILTER if allow_unknown else CTX_RULE_YESNO
    return rule + f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def make_support_prompt(
    question: str,
    context: str,
    gold_answer: str,
    position: str,
    allow_unknown: bool,
    prompt_profile: str = "default",
) -> str:
    base = make_context_base_prompt(question, context, allow_unknown=allow_unknown, prompt_profile=prompt_profile)
    return inject_evidence(base, make_evidence(gold_answer, prompt_profile=prompt_profile), position)


def make_conflict_prompt(
    question: str,
    context: str,
    gold_answer: str,
    position: str,
    allow_unknown: bool,
    prompt_profile: str = "default",
) -> str:
    base = make_context_base_prompt(question, context, allow_unknown=allow_unknown, prompt_profile=prompt_profile)
    return inject_evidence(base, make_evidence(invert_yes_no(gold_answer), prompt_profile=prompt_profile), position)


def wrap_as_chat(tokenizer, user_text: str, enable_thinking: bool = False, prompt_profile: str = "default") -> str:
    messages = [
        {"role": "system", "content": get_system_prompt(prompt_profile)},
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


def load_model(model_name: str, device_map: str, dtype_name: str):
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


@torch.inference_mode()
def logprob_continuation(
    model,
    tok,
    prompt: str,
    continuation: str,
    enable_thinking: bool,
    prompt_profile: str = "default",
) -> float:
    formatted = wrap_as_chat(tok, prompt, enable_thinking=enable_thinking, prompt_profile=prompt_profile)
    enc_prompt = tok(formatted, add_special_tokens=False, return_tensors="pt")
    enc_full = tok(formatted + continuation, add_special_tokens=False, return_tensors="pt")
    input_ids_prompt = enc_prompt["input_ids"][0]
    input_ids_full = enc_full["input_ids"][0]
    cont_ids = input_ids_full[len(input_ids_prompt) :]
    if cont_ids.numel() == 0:
        return float("-inf")

    input_ids_full = input_ids_full.unsqueeze(0).to(model.device)
    outputs = model(input_ids_full)
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    start = len(input_ids_prompt)
    total = 0.0
    for i, tok_id in enumerate(cont_ids):
        pos = start + i
        total += float(log_probs[0, pos - 1, int(tok_id)])
    return total


@torch.inference_mode()
def score_yes_no_unknown(
    model,
    tok,
    prompt: str,
    enable_thinking: bool,
    prompt_profile: str = "default",
) -> dict[str, float]:
    yes_lp = max(
        logprob_continuation(model, tok, prompt, " yes", enable_thinking, prompt_profile=prompt_profile),
        logprob_continuation(model, tok, prompt, "yes", enable_thinking, prompt_profile=prompt_profile),
    )
    no_lp = max(
        logprob_continuation(model, tok, prompt, " no", enable_thinking, prompt_profile=prompt_profile),
        logprob_continuation(model, tok, prompt, "no", enable_thinking, prompt_profile=prompt_profile),
    )
    unk_lp = max(
        logprob_continuation(model, tok, prompt, " unknown", enable_thinking, prompt_profile=prompt_profile),
        logprob_continuation(model, tok, prompt, "unknown", enable_thinking, prompt_profile=prompt_profile),
    )
    return {YES: yes_lp, NO: no_lp, UNKNOWN: unk_lp}


@torch.inference_mode()
def score_yes_no(
    model,
    tok,
    prompt: str,
    enable_thinking: bool,
    prompt_profile: str = "default",
) -> dict[str, float]:
    yes_lp = max(
        logprob_continuation(model, tok, prompt, " yes", enable_thinking, prompt_profile=prompt_profile),
        logprob_continuation(model, tok, prompt, "yes", enable_thinking, prompt_profile=prompt_profile),
    )
    no_lp = max(
        logprob_continuation(model, tok, prompt, " no", enable_thinking, prompt_profile=prompt_profile),
        logprob_continuation(model, tok, prompt, "no", enable_thinking, prompt_profile=prompt_profile),
    )
    return {YES: yes_lp, NO: no_lp}


def decide_with_unknown(scores: dict[str, float], yn_tau: float, unk_tau: float):
    sy, sn, su = scores[YES], scores[NO], scores[UNKNOWN]
    best_yn = max(sy, sn)
    yn_margin = abs(sy - sn)
    unk_adv = su - best_yn
    if unk_adv >= unk_tau or yn_margin < yn_tau:
        pred = UNKNOWN
    else:
        pred = YES if sy >= sn else NO
    return pred, {
        YES: sy,
        NO: sn,
        UNKNOWN: su,
        "yn_margin": yn_margin,
        "unk_adv": unk_adv,
        "yn_tau": float(yn_tau),
        "unk_tau": float(unk_tau),
    }


def decide_yes_no(scores: dict[str, float]):
    sy, sn = scores[YES], scores[NO]
    pred = YES if sy >= sn else NO
    return pred, {
        YES: sy,
        NO: sn,
        "yesno_margin": sy - sn,
    }


def derive_follow_label(conflict_pred: str, gold: str) -> str:
    if conflict_pred == invert_yes_no(gold):
        return "follow_conflict"
    if conflict_pred == gold:
        return "resist"
    return "unknown"
