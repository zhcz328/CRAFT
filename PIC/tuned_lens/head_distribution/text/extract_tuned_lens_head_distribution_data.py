#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor


DEFAULT_SELECTED_HEADS = Path(
    "/root/logit_lens/VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
    "headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json"
)
DEFAULT_MANIFESTS = [
    Path("/root/logit_lens/VQA_RAD/Hulu-med/tuned_lens/data/train_before_question.jsonl"),
    Path("/root/logit_lens/VQA_RAD/Hulu-med/tuned_lens/data/val_before_question.jsonl"),
]
DEFAULT_LENS_CKPT = Path(
    "/root/logit_lens/VQA_RAD/Hulu-med/tuned_lens/results/train_before_question_idreg_tmp_ok/tuned_lens.pt"
)
DEFAULT_MODEL = "/root/autodl-tmp/Hulu-Med-4B"
DEFAULT_OUT_DIR = Path("/root/logit_lens/PIC/tuned_lens/head_distribution/text/data")


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path("/root/logit_lens/VQA_RAD/Hulu-med")
TUNED_LENS_DIR = PROJECT_DIR / "tuned_lens"

COMMON = load_module("hulu_tuned_lens_common", TUNED_LENS_DIR / "common.py")
MODELING = load_module("hulu_tuned_lens_modeling", TUNED_LENS_DIR / "modeling.py")
ABLATE = load_module("hulu_ablate_head", PROJECT_DIR / "ablate_head.py")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Export per-head tuned-lens margin distributions for selected Hulu-med "
            "head-scan heads on the VQA-RAD val split."
        )
    )
    ap.add_argument("--selected-heads", type=Path, default=DEFAULT_SELECTED_HEADS)
    ap.add_argument(
        "--selected-heads-key",
        type=str,
        default="",
        help="Optional list field inside --selected-heads to use, e.g. top20 or results.",
    )
    ap.add_argument(
        "--selected-heads-topk",
        type=int,
        default=0,
        help="Optional top-k limit after parsing selected heads.",
    )
    ap.add_argument("--manifest", type=Path, action="append", default=None)
    ap.add_argument("--lens-ckpt", type=Path, default=DEFAULT_LENS_CKPT)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument("--lens-device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=672)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument(
        "--prompt-types",
        type=str,
        default="base,conflict",
        help="Comma-separated prompt types to export. base->NC, conflict->IC.",
    )
    ap.add_argument(
        "--follow-labels",
        type=str,
        default="follow_conflict",
        help="Comma-separated follow_label filter. Empty means keep all.",
    )
    ap.add_argument(
        "--require-nc-correct",
        action="store_true",
        help="Keep only samples whose base/NC prediction matches the gold answer.",
    )
    ap.add_argument(
        "--require-ic-wrong",
        action="store_true",
        help="Keep only samples whose conflict/IC prediction matches the wrong answer.",
    )
    ap.add_argument(
        "--lens-mode",
        type=str,
        default="raw",
        choices=["raw", "tuned"],
        help="Use raw logit lens or tuned lens translator on per-head representations.",
    )
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

    pairs: list[tuple[int, int]] = []
    for item in items[: topk or None]:
        if not isinstance(item, dict):
            continue
        if "layer" in item and "head" in item:
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
            if not hasattr(attn, "o_proj"):
                raise RuntimeError(f"Layer {layer_idx} attention module missing o_proj")
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def project_single_head(attn_module, head_input: torch.Tensor, head_idx: int, num_heads: int, head_dim: int) -> torch.Tensor:
    start = head_idx * head_dim
    end = start + head_dim
    weight = attn_module.o_proj.weight[:, start:end]
    return F.linear(
        head_input.to(device=weight.device, dtype=weight.dtype),
        weight,
        bias=None,
    )


def prepare_rows(
    manifests: list[Path],
    prompt_types: set[str],
    follow_labels: set[str],
    max_records: int,
    tokenizer,
    require_nc_correct: bool,
    require_ic_wrong: bool,
) -> list[dict]:
    eligible = []
    for manifest in manifests:
        rows = COMMON.read_jsonl(manifest)
        for row in rows:
            if row.get("prompt_type") not in prompt_types:
                continue
            if follow_labels and row.get("follow_label") not in follow_labels:
                continue
            if require_nc_correct and row.get("baseline_pred") != row.get("gold_answer"):
                continue
            if require_ic_wrong and row.get("conflict_pred") != row.get("wrong_answer"):
                continue
            gold_ids = COMMON.answer_token_ids(tokenizer, row["gold_answer"])
            wrong_ids = COMMON.answer_token_ids(tokenizer, row["wrong_answer"])
            if not gold_ids or not wrong_ids:
                continue
            item = dict(row)
            item["source_manifest"] = str(manifest)
            item["gold_token_ids"] = gold_ids
            item["wrong_token_ids"] = wrong_ids
            item["gold_first_token_id"] = int(gold_ids[0])
            item["wrong_first_token_id"] = int(wrong_ids[0])
            item["condition"] = "NC" if row["prompt_type"] == "base" else "IC"
            eligible.append(item)
            if max_records > 0 and len(eligible) >= max_records:
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
) -> list[dict]:
    samples: list[dict] = []
    project_root = PROJECT_DIR
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
    batch_pbar = tqdm(batches, total=len(batches), desc="extract head reps", unit="batch")
    for batch_rows in batch_pbar:
        batch_inputs = COMMON.build_batch_inputs(PROCESSOR, batch_rows, project_root, image_size=image_size)
        batch_inputs.pop("token_type_ids", None)
        prompt_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
        batch_inputs = COMMON.move_to_device(batch_inputs, model_device, model_dtype)

        with torch.no_grad():
            with OProjInputCatcher(layer2attn, target_layers) as prompt_catcher:
                model(**batch_inputs, output_hidden_states=False, use_cache=False)
                prompt_o_proj_inputs = {layer: tensor.clone() for layer, tensor in prompt_catcher.cache.items()}

        for item_idx, row in enumerate(batch_rows):
            prompt_len = int(prompt_lengths[item_idx])
            gold_first_token_id = int(row["gold_first_token_id"])
            wrong_first_token_id = int(row["wrong_first_token_id"])

            for layer_idx in target_layers:
                translator = translators[str(layer_idx)]
                translator_dtype = next(translator.parameters()).dtype
                attn_module = layer2attn[layer_idx]
                prompt_layer_tensor = prompt_o_proj_inputs[layer_idx][item_idx]

                for head_idx in heads_by_layer[layer_idx]:
                    start = head_idx * head_dim
                    end = start + head_dim
                    head_hidden = project_single_head(
                        attn_module,
                        prompt_layer_tensor[:, start:end],
                        head_idx,
                        num_heads,
                        head_dim,
                    )
                    if lens_mode == "tuned":
                        score_hidden = translator(head_hidden.to(device=lens_device, dtype=translator_dtype))
                    else:
                        score_hidden = head_hidden
                    gold_logprob = first_token_logprob_from_hidden(
                        score_hidden,
                        prompt_len,
                        gold_first_token_id,
                        output_head,
                    )
                    wrong_logprob = first_token_logprob_from_hidden(
                        score_hidden,
                        prompt_len,
                        wrong_first_token_id,
                        output_head,
                    )
                    samples.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "record_id": row.get("record_id"),
                            "sample_key": row.get("sample_key"),
                            "img_id": row.get("img_id"),
                            "question": row.get("question"),
                            "source_manifest": row.get("source_manifest"),
                            "follow_label": row.get("follow_label"),
                            "prompt_type": row.get("prompt_type"),
                            "condition": row["condition"],
                            "layer": layer_idx,
                            "head": head_idx,
                            "head_label": f"Head {layer_idx * 32 + head_idx}",
                            "gold_answer": row.get("gold_answer"),
                            "wrong_answer": row.get("wrong_answer"),
                            "gold_logprob": gold_logprob,
                            "wrong_logprob": wrong_logprob,
                            "margin": gold_logprob - wrong_logprob,
                        }
                    )
        batch_pbar.set_postfix(samples=len(samples))

    return samples


def summarize(samples: list[dict]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in samples:
        grouped[(row["head_label"], row["condition"])].append(float(row["margin"]))

    heads = sorted({row["head_label"] for row in samples}, key=lambda x: int(x.split()[1]))
    by_head = []
    for head_label in heads:
        head_rows = [row for row in samples if row["head_label"] == head_label]
        layer = int(head_rows[0]["layer"])
        head = int(head_rows[0]["head"])
        summary_row = {
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
                summary_row[f"{cond.lower()}_mean_margin"] = mu
                summary_row[f"{cond.lower()}_std_margin"] = (
                    math.sqrt(sum((x - mu) ** 2 for x in values) / max(1, len(values) - 1))
                    if len(values) > 1
                    else 0.0
                )
            else:
                summary_row[f"{cond.lower()}_mean_margin"] = None
                summary_row[f"{cond.lower()}_std_margin"] = None
        by_head.append(summary_row)
    return {"heads": by_head}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer

    manifests = args.manifest if args.manifest else DEFAULT_MANIFESTS
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

    prompt_types = {item.strip() for item in args.prompt_types.split(",") if item.strip()}
    follow_labels = {item.strip() for item in args.follow_labels.split(",") if item.strip()}
    global PROCESSOR, TOKENIZER
    PROCESSOR = processor
    TOKENIZER = tokenizer

    rows = prepare_rows(
        manifests,
        prompt_types,
        follow_labels,
        args.max_records,
        tokenizer,
        require_nc_correct=args.require_nc_correct,
        require_ic_wrong=args.require_ic_wrong,
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
    )

    samples_path = args.out_dir / "head_margin_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for row in all_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "selected_heads_path": str(args.selected_heads),
        "manifests": [str(path) for path in manifests],
        "lens_ckpt": str(args.lens_ckpt),
        "model": args.model,
        "lens_mode": args.lens_mode,
        "position": "before_question",
        "margin_definition": "next_token_logprob(gold)-next_token_logprob(wrong)",
        "n_selected_heads": len(selected_pairs),
        "n_records": len(rows),
        "prompt_types": sorted(prompt_types),
        "follow_labels": sorted(follow_labels),
        "require_nc_correct": args.require_nc_correct,
        "require_ic_wrong": args.require_ic_wrong,
        "head_summaries": summarize(all_samples)["heads"],
    }
    summary_path = args.out_dir / "head_margin_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"samples": str(samples_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


PROCESSOR = None
TOKENIZER = None


if __name__ == "__main__":
    COMMON.ensure_video_import_compat()
    main()
