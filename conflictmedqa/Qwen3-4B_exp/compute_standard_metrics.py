import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


YES = "yes"
NO = "no"
SPLITS = ("all", "train", "val")
POSITIONS = ("prefix", "before_question", "before_answer")
EPS = 1e-12


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def example_key(rec: dict) -> Tuple[int, str, str]:
    return int(rec["pair_id"]), str(rec["side"]), str(rec["position"])


def load_baseline_by_split(base_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    all_rows = read_jsonl(base_dir / "conflict_positions.jsonl")
    val_rows = read_jsonl(base_dir / "conflict_positions_val.jsonl")

    val_keys = {example_key(r) for r in val_rows}
    train_rows = [r for r in all_rows if example_key(r) not in val_keys]

    out = {
        "all": {pos: [] for pos in POSITIONS},
        "train": {pos: [] for pos in POSITIONS},
        "val": {pos: [] for pos in POSITIONS},
    }
    for split, rows in (("all", all_rows), ("train", train_rows), ("val", val_rows)):
        for row in rows:
            pos = row["position"]
            if pos in out[split]:
                out[split][pos].append(row)
    return out


def resolve_ablation_path(pos_dir: Path, prefix: str, split: str) -> Optional[Path]:
    candidates = []
    if split == "all":
        candidates = [
            pos_dir / f"{prefix}_all.jsonl",
            pos_dir / f"{prefix}.jsonl",
        ]
    else:
        candidates = [
            pos_dir / f"{prefix}_{split}.jsonl",
            pos_dir / f"{prefix}_all_{split}.jsonl",
        ]

    for path in candidates:
        if path.exists():
            return path
    return None


def load_eval_rows(path: Path, position: str) -> List[dict]:
    rows = read_jsonl(path)
    clean_rows = []
    for row in rows:
        if "_meta" in row:
            continue
        if row.get("position") != position:
            continue
        clean_rows.append(row)
    return clean_rows


def to_binary_label(label: str) -> int:
    if label == YES:
        return 1
    if label == NO:
        return 0
    raise ValueError(f"Unsupported label: {label}")


def f1_for_positive(y_true: List[int], y_pred: List[int], positive: int) -> float:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive and yp == positive)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive and yp == positive)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive and yp != positive)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def expected_calibration_error(probs: List[float], preds: List[int], golds: List[int], n_bins: int = 10) -> float:
    n = len(probs)
    if n == 0:
        return 0.0

    ece = 0.0
    confidences = [p if pred == 1 else 1.0 - p for p, pred in zip(probs, preds)]
    correctness = [1.0 if pred == gold else 0.0 for pred, gold in zip(preds, golds)]

    for bin_idx in range(n_bins):
        lo = bin_idx / n_bins
        hi = (bin_idx + 1) / n_bins
        if bin_idx == n_bins - 1:
            idxs = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            idxs = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not idxs:
            continue
        bin_acc = sum(correctness[i] for i in idxs) / len(idxs)
        bin_conf = sum(confidences[i] for i in idxs) / len(idxs)
        ece += abs(bin_acc - bin_conf) * (len(idxs) / n)
    return ece


def compute_condition_metrics(rows: Iterable[dict], field: str) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "brier_score": 0.0,
            "nll": 0.0,
        }

    golds: List[int] = []
    preds: List[int] = []
    probs_yes: List[float] = []

    for row in rows:
        gold = to_binary_label(row["gold"])
        delta = float(row[field]["delta"])
        prob_yes = min(max(sigmoid(delta), EPS), 1.0 - EPS)
        pred = 1 if prob_yes >= 0.5 else 0

        golds.append(gold)
        preds.append(pred)
        probs_yes.append(prob_yes)

    acc = sum(1 for yt, yp in zip(golds, preds) if yt == yp) / len(rows)
    f1_yes = f1_for_positive(golds, preds, 1)
    f1_no = f1_for_positive(golds, preds, 0)
    macro_f1 = (f1_yes + f1_no) / 2.0
    brier = sum((p - y) ** 2 for p, y in zip(probs_yes, golds)) / len(rows)
    nll = -sum(
        y * math.log(p) + (1 - y) * math.log(1.0 - p)
        for p, y in zip(probs_yes, golds)
    ) / len(rows)
    ece = expected_calibration_error(probs_yes, preds, golds)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece": ece,
        "brier_score": brier,
        "nll": nll,
    }


def compute_attack_success_rate(rows: Iterable[dict]) -> Optional[float]:
    rows = list(rows)
    if not rows:
        return None

    clean_correct = [row for row in rows if row["base"]["pred"] == row["gold"]]
    if not clean_correct:
        return None
    attack_success = sum(1 for row in clean_correct if row["conflict"]["pred"] != row["gold"])
    return attack_success / len(clean_correct)


def summarize_rows(rows: List[dict]) -> Dict[str, object]:
    clean = compute_condition_metrics(rows, "base")
    conflict = compute_condition_metrics(rows, "conflict")
    support = compute_condition_metrics(rows, "support")
    attack_success_rate = compute_attack_success_rate(rows)

    return {
        "n_samples": len(rows),
        "clean_accuracy": clean["accuracy"],
        "conflict_accuracy": conflict["accuracy"],
        "robust_accuracy": conflict["accuracy"],
        "attack_success_rate": attack_success_rate,
        "accuracy_drop": clean["accuracy"] - conflict["accuracy"],
        "macro_f1": conflict["macro_f1"],
        "ece": conflict["ece"],
        "brier_score": conflict["brier_score"],
        "nll": conflict["nll"],
        "clean_macro_f1": clean["macro_f1"],
        "conflict_macro_f1": conflict["macro_f1"],
        "support_accuracy": support["accuracy"],
        "clean_ece": clean["ece"],
        "conflict_ece": conflict["ece"],
        "clean_brier_score": clean["brier_score"],
        "conflict_brier_score": conflict["brier_score"],
        "clean_nll": clean["nll"],
        "conflict_nll": conflict["nll"],
    }


def compare_to_baseline(candidate: Dict[str, object], baseline: Dict[str, object]) -> Dict[str, Optional[float]]:
    def delta(name: str) -> Optional[float]:
        a = candidate.get(name)
        b = baseline.get(name)
        if a is None or b is None:
            return None
        return float(a) - float(b)

    return {
        "delta_robust_accuracy": delta("robust_accuracy"),
        "delta_conflict_accuracy": delta("conflict_accuracy"),
        "delta_clean_accuracy": delta("clean_accuracy"),
        "delta_attack_success_rate": delta("attack_success_rate"),
        "delta_accuracy_drop": delta("accuracy_drop"),
    }


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Qwen3-4B_exp root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    result_all_positions = root / "result_all_positions"
    result_train = root / "result_train"

    baseline_by_split = load_baseline_by_split(result_all_positions)

    for position in POSITIONS:
        pos_dir = result_train / position
        for split in SPLITS:
            baseline_rows = baseline_by_split[split][position]
            baseline_metrics = summarize_rows(baseline_rows)

            selected_path = resolve_ablation_path(pos_dir, "ablate_heads", split)
            random_path = resolve_ablation_path(pos_dir, "ablate_random", split)

            selected_metrics = None
            random_metrics = None

            if selected_path is not None:
                selected_rows = load_eval_rows(selected_path, position)
                selected_metrics = summarize_rows(selected_rows)

            if random_path is not None:
                random_rows = load_eval_rows(random_path, position)
                random_metrics = summarize_rows(random_rows)

            payload = {
                "position": position,
                "split": split,
                "baseline_source": str(result_all_positions),
                "baseline": baseline_metrics,
                "selected_ablation": {
                    "source": str(selected_path) if selected_path else None,
                    "metrics": selected_metrics,
                    "vs_baseline": compare_to_baseline(selected_metrics, baseline_metrics) if selected_metrics else None,
                },
                "random_ablation": {
                    "source": str(random_path) if random_path else None,
                    "metrics": random_metrics,
                    "vs_baseline": compare_to_baseline(random_metrics, baseline_metrics) if random_metrics else None,
                },
                "notes": {
                    "probability_reconstruction": "Binary probabilities are reconstructed as sigmoid(delta), where delta = logP(Yes) - logP(No).",
                    "attack_success_rate_definition": "Computed on clean-correct examples only: P(conflict prediction != gold | clean prediction == gold).",
                    "ece_bins": 10,
                    "train_baseline_recovery": "Recovered as (all baseline rows) minus (val baseline rows) by (pair_id, side, position).",
                },
            }

            out_path = pos_dir / f"standard_metrics_{split}.json"
            save_json(out_path, payload)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
