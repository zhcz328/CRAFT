#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor


DEFAULT_SELECTED_HEADS = Path(
    "/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/"
    "headscan_slake_mm_accel/head_scan_core_layers.json"
)
DEFAULT_CSVS = [
    Path("/root/logit_lens/Slake_vqa/image_conflict/data/hulumed4b/slake_nc_correct_ic_ready_train.csv"),
    Path("/root/logit_lens/Slake_vqa/image_conflict/data/hulumed4b/slake_nc_correct_ic_ready_val.csv"),
]
DEFAULT_PREDS = Path("/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/eval-results_slake_image_conflict/preds.jsonl")
DEFAULT_LENS_CKPT = Path(
    "/root/autodl-tmp/image_conflict/tuned_lens/hulumed4b/results/image_conflict/train_idreg/tuned_lens.pt"
)
DEFAULT_MODEL = "/root/autodl-tmp/Hulu-Med-4B"
DEFAULT_OUT_DIR = Path("/root/logit_lens/PIC/tuned_lens/head_distribution/image/hulumed-4B/data")
DEFAULT_PROJECT_DIR = Path("/root/logit_lens/Slake_vqa/image_conflict")


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROJECT_DIR = DEFAULT_PROJECT_DIR
COMMON = load_module("image_conflict_tuned_lens_common", PROJECT_DIR / "tuned_lens/common.py")
MODELING = load_module("image_conflict_tuned_lens_modeling", PROJECT_DIR / "tuned_lens/modeling.py")
ABLATE = load_module("image_conflict_ablate_head", PROJECT_DIR / "ablate_head.py")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export per-head tuned-lens distributions for image-conflict samples, "
            "using margin = logprob(unknown) - logprob(best competing answer)."
        )
    )
    ap.add_argument("--selected-heads", type=Path, default=DEFAULT_SELECTED_HEADS)
    ap.add_argument("--selected-heads-key", type=str, default="results")
    ap.add_argument("--selected-heads-topk", type=int, default=0)
    ap.add_argument("--csv", type=Path, action="append", default=None)
    ap.add_argument("--preds-jsonl", type=Path, default=DEFAULT_PREDS)
    ap.add_argument("--lens-ckpt", type=Path, default=DEFAULT_LENS_CKPT)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--lens-device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=672)
    ap.add_argument("--mask-scale", type=float, default=1.0)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--lens-mode", type=str, default="tuned", choices=["raw", "tuned"])
    ap.add_argument("--require-nc-correct", action="store_true", default=True)
    ap.add_argument("--require-ic-unknown", action="store_true" )
    return ap.parse_args()


def pick_device(name: str, fallback: torch.device) -> torch.device:
    if name == "auto":
        return fallback
    return torch.device(name)


def load_selected_pairs(path: Path, key: str, topk: int) -> list[tuple[int, int]]:
    if not key and topk <= 0:
        return ABLATE.parse_selected_heads(str(path))

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data
    if key:
        if not isinstance(data, dict) or key not in data or not isinstance(data[key], list):
            raise ValueError(f"Cannot find list field '{key}' in {path}")
        items = data[key]
    elif isinstance(data, dict):
        for candidate_key in ("selected", "selected_heads", "results", "heads", "topk", "top20"):
            if candidate_key in data and isinstance(data[candidate_key], list):
                items = data[candidate_key]
                break
    if not isinstance(items, list):
        raise ValueError(f"Selected heads source is not a list: {path}")

    pairs = []
    for item in items[: topk or None]:
        if isinstance(item, dict) and "layer" in item and "head" in item:
            pairs.append((int(item["layer"]), int(item["head"])))
    if not pairs:
        raise ValueError(f"No head pairs parsed from {path}")
    return pairs


def chunked(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def first_token_logprob_from_hidden(hidden_seq, prompt_len: int, token_id: int, output_head) -> float:
    pred_hidden = hidden_seq[prompt_len - 1]
    logits = output_head(pred_hidden.to(device=output_head.weight.device, dtype=output_head.weight.dtype))
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(log_probs[int(token_id)].item())


class OProjInputCatcher:
    def __init__(self, layer2attn: dict[int, Any], target_layers: list[int]):
        self.layer2attn = layer2attn
        self.target_layers = target_layers
        self.cache: dict[int, torch.Tensor] = {}
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def _hook(module, args):
            if args:
                self.cache[layer_idx] = args[0].detach()
        return _hook

    def __enter__(self):
        for layer_idx in self.target_layers:
            attn = self.layer2attn[layer_idx]
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def project_single_head(attn_module, head_input: torch.Tensor, head_idx: int, head_dim: int) -> torch.Tensor:
    start = head_idx * head_dim
    end = start + head_dim
    weight = attn_module.o_proj.weight[:, start:end]
    return F.linear(head_input.to(device=weight.device, dtype=weight.dtype), weight, bias=None)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def best_competing_answer(scores_json: str | dict[str, float], unknown_answer: str) -> str | None:
    if isinstance(scores_json, str):
        if not scores_json.strip():
            return None
        score_map = json.loads(scores_json)
    else:
        score_map = dict(scores_json or {})
    unknown_norm = str(unknown_answer).strip().lower()
    candidates = [(ans, float(score)) for ans, score in score_map.items() if str(ans).strip().lower() != unknown_norm]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def prepare_image_rows(
    csv_paths: list[Path],
    preds_jsonl: Path,
    max_records: int,
    tokenizer,
    require_nc_correct: bool,
    require_ic_unknown: bool,
) -> list[dict]:
    preds_by_id = {str(row.get("id")): row for row in read_jsonl(preds_jsonl)}
    eligible = []
    for csv_path in csv_paths:
        for row in read_csv_rows(csv_path):
            pred_row = preds_by_id.get(str(row.get("id")))
            if not pred_row:
                continue
            if require_nc_correct and not bool(pred_row.get("nc_correct", False)):
                continue
            if require_ic_unknown and not bool(pred_row.get("ic_is_unknown", False)):
                continue

            unknown_answer = str(pred_row.get("unknown_target") or row.get("unknown_target") or "unknown").strip()
            nc_best = best_competing_answer(pred_row.get("nc_scores_json", "{}"), unknown_answer)
            ic_best = best_competing_answer(pred_row.get("ic_scores_json", "{}"), unknown_answer)
            if not nc_best or not ic_best:
                continue

            prompt_text = COMMON.make_prompt_no_evidence(str(row.get("question", "")).strip())
            base_item = dict(row)
            base_item.update(
                {
                    "source_csv": str(csv_path),
                    "prompt_type": "base",
                    "condition": "NC",
                    "prompt_text": prompt_text,
                    "image_variant": "nc",
                    "unknown_answer": unknown_answer,
                    "best_competing_answer": nc_best,
                    "unknown_token_ids": COMMON.answer_token_ids(tokenizer, unknown_answer),
                    "best_competing_token_ids": COMMON.answer_token_ids(tokenizer, nc_best),
                    "nc_scores_json": pred_row.get("nc_scores_json", "{}"),
                    "ic_scores_json": pred_row.get("ic_scores_json", "{}"),
                    "pred_row": pred_row,
                }
            )
            conflict_item = dict(row)
            conflict_item.update(
                {
                    "source_csv": str(csv_path),
                    "prompt_type": "conflict",
                    "condition": "IC",
                    "prompt_text": prompt_text,
                    "image_variant": "ic",
                    "unknown_answer": unknown_answer,
                    "best_competing_answer": ic_best,
                    "unknown_token_ids": COMMON.answer_token_ids(tokenizer, unknown_answer),
                    "best_competing_token_ids": COMMON.answer_token_ids(tokenizer, ic_best),
                    "nc_scores_json": pred_row.get("nc_scores_json", "{}"),
                    "ic_scores_json": pred_row.get("ic_scores_json", "{}"),
                    "pred_row": pred_row,
                }
            )
            if not base_item["unknown_token_ids"] or not base_item["best_competing_token_ids"]:
                continue
            if not conflict_item["unknown_token_ids"] or not conflict_item["best_competing_token_ids"]:
                continue
            base_item["unknown_first_token_id"] = int(base_item["unknown_token_ids"][0])
            base_item["best_competing_first_token_id"] = int(base_item["best_competing_token_ids"][0])
            conflict_item["unknown_first_token_id"] = int(conflict_item["unknown_token_ids"][0])
            conflict_item["best_competing_first_token_id"] = int(conflict_item["best_competing_token_ids"][0])
            eligible.extend([base_item, conflict_item])
            if max_records > 0 and len(eligible) >= max_records * 2:
                return eligible
    return eligible


def compute_head_samples(
    selected_pairs: list[tuple[int, int]],
    rows: list[dict],
    translators,
    lens_mode: str,
    model,
    output_head,
    model_dtype,
    model_device: torch.device,
    lens_device: torch.device,
    batch_size: int,
    image_size: int,
    mask_scale: float,
    project_root: Path,
) -> list[dict]:
    samples: list[dict] = []
    layer2attn, _ = ABLATE._find_self_attn_modules(model)
    num_heads, hidden_size = ABLATE._get_num_heads_and_hidden(model)
    if num_heads is None or hidden_size is None:
        raise RuntimeError("Cannot infer attention head layout.")
    head_dim = hidden_size // num_heads
    target_layers = sorted({layer for layer, _ in selected_pairs})
    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in selected_pairs:
        heads_by_layer[int(layer)].append(int(head))

    batches = list(chunked(rows, batch_size))
    batch_pbar = tqdm(batches, total=len(batches), desc="extract image head reps", unit="batch")
    for batch_rows in batch_pbar:
        batch_inputs = COMMON.build_batch_inputs(
            PROCESSOR,
            batch_rows,
            project_root,
            image_size=image_size,
            mask_scale=mask_scale,
        )
        batch_inputs.pop("token_type_ids", None)
        prompt_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
        batch_inputs = COMMON.move_to_device(batch_inputs, model_device, model_dtype)

        with torch.no_grad():
            with OProjInputCatcher(layer2attn, target_layers) as prompt_catcher:
                model(**batch_inputs, output_hidden_states=False, use_cache=False)
                prompt_o_proj_inputs = {layer: tensor.clone() for layer, tensor in prompt_catcher.cache.items()}

        for item_idx, row in enumerate(batch_rows):
            prompt_len = int(prompt_lengths[item_idx])
            unknown_first_token_id = int(row["unknown_first_token_id"])
            competitor_first_token_id = int(row["best_competing_first_token_id"])
            for layer_idx in target_layers:
                translator = translators[str(layer_idx)]
                translator_dtype = next(translator.parameters()).dtype
                attn_module = layer2attn[layer_idx]
                prompt_layer_tensor = prompt_o_proj_inputs[layer_idx][item_idx]

                for head_idx in heads_by_layer[layer_idx]:
                    start = head_idx * head_dim
                    end = start + head_dim
                    head_hidden = project_single_head(attn_module, prompt_layer_tensor[:, start:end], head_idx, head_dim)
                    if lens_mode == "tuned":
                        score_hidden = translator(head_hidden.to(device=lens_device, dtype=translator_dtype))
                    else:
                        score_hidden = head_hidden

                    unknown_logprob = first_token_logprob_from_hidden(
                        score_hidden,
                        prompt_len,
                        unknown_first_token_id,
                        output_head,
                    )
                    competitor_logprob = first_token_logprob_from_hidden(
                        score_hidden,
                        prompt_len,
                        competitor_first_token_id,
                        output_head,
                    )
                    samples.append(
                        {
                            "sample_id": row.get("id"),
                            "record_id": f"{row.get('id')}:{row.get('condition')}",
                            "sample_key": f"{row.get('img_id')}||{row.get('question')}",
                            "img_id": row.get("img_id"),
                            "question": row.get("question"),
                            "source_csv": row.get("source_csv"),
                            "prompt_type": row.get("prompt_type"),
                            "condition": row["condition"],
                            "layer": layer_idx,
                            "head": head_idx,
                            "head_label": f"Head {layer_idx * 32 + head_idx}",
                            "unknown_answer": row.get("unknown_answer"),
                            "best_competing_answer": row.get("best_competing_answer"),
                            "unknown_logprob": unknown_logprob,
                            "best_competing_logprob": competitor_logprob,
                            "margin": unknown_logprob - competitor_logprob,
                        }
                    )
        batch_pbar.set_postfix(samples=len(samples))
    return samples


def summarize(samples: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in samples:
        grouped[(row["head_label"], row["condition"])].append(float(row["margin"]))
    heads = sorted({row["head_label"] for row in samples}, key=lambda x: int(x.split()[1]))
    out = []
    for head_label in heads:
        head_rows = [row for row in samples if row["head_label"] == head_label]
        layer = int(head_rows[0]["layer"])
        head = int(head_rows[0]["head"])
        row = {
            "head_label": head_label,
            "layer": layer,
            "head": head,
            "count_nc": len(grouped.get((head_label, "NC"), [])),
            "count_ic": len(grouped.get((head_label, "IC"), [])),
        }
        for cond in ("NC", "IC"):
            values = grouped.get((head_label, cond), [])
            if values:
                mu = sum(values) / len(values)
                row[f"{cond.lower()}_mean_margin"] = mu
                row[f"{cond.lower()}_std_margin"] = (
                    math.sqrt(sum((x - mu) ** 2 for x in values) / max(1, len(values) - 1)) if len(values) > 1 else 0.0
                )
            else:
                row[f"{cond.lower()}_mean_margin"] = None
                row[f"{cond.lower()}_std_margin"] = None
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    csv_paths = args.csv if args.csv else DEFAULT_CSVS
    selected_pairs = load_selected_pairs(args.selected_heads, args.selected_heads_key, args.selected_heads_topk)
    ckpt = torch.load(args.lens_ckpt, map_location="cpu")

    model = COMMON.load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=MODELING.resolve_dtype(args.dtype),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_head = COMMON.get_output_head(model)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    lens_device = pick_device(args.lens_device, model_device)
    translators = MODELING.build_translators(
        layer_indices=ckpt["layer_indices"],
        hidden_dim=ckpt["hidden_dim"],
        dtype=MODELING.resolve_dtype(ckpt["translator_dtype"]),
        rank=ckpt["translator_rank"],
    ).to(lens_device)
    translators.load_state_dict(ckpt["state_dict"], strict=True)
    translators.eval()

    global PROCESSOR, TOKENIZER
    PROCESSOR = processor
    TOKENIZER = tokenizer

    rows = prepare_image_rows(
        csv_paths=csv_paths,
        preds_jsonl=args.preds_jsonl,
        max_records=args.max_records,
        tokenizer=tokenizer,
        require_nc_correct=args.require_nc_correct,
        require_ic_unknown=args.require_ic_unknown,
    )

    all_samples = compute_head_samples(
        selected_pairs=selected_pairs,
        rows=rows,
        translators=translators,
        lens_mode=args.lens_mode,
        model=model,
        output_head=output_head,
        model_dtype=model_dtype,
        model_device=model_device,
        lens_device=lens_device,
        batch_size=args.batch_size,
        image_size=args.image_size,
        mask_scale=args.mask_scale,
        project_root=args.project_dir,
    )

    samples_path = args.out_dir / "head_margin_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in all_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "selected_heads_path": str(args.selected_heads),
        "selected_heads_key": args.selected_heads_key,
        "csvs": [str(path) for path in csv_paths],
        "preds_jsonl": str(args.preds_jsonl),
        "lens_ckpt": str(args.lens_ckpt),
        "model": args.model,
        "lens_mode": args.lens_mode,
        "position": "image_conflict",
        "margin_definition": "next_token_logprob(unknown)-next_token_logprob(best_competing)",
        "n_selected_heads": len(selected_pairs),
        "n_records": len(rows) // 2,
        "require_nc_correct": args.require_nc_correct,
        "require_ic_unknown": args.require_ic_unknown,
        "head_summaries": summarize(all_samples),
    }
    summary_path = args.out_dir / "head_margin_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": str(samples_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


PROCESSOR = None
TOKENIZER = None


if __name__ == "__main__":
    COMMON.ensure_video_import_compat()
    main()
