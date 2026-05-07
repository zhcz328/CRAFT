import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import (
    build_condition_pack,
    collate_processor_outputs,
    ensure_video_import_compat,
    find_self_attn_modules,
    first_answer_token_id,
    get_final_norm,
    get_output_head,
    load_mm_model,
    move_to_device,
    read_jsonl,
    resolve_dtype,
    resolve_image_path,
    resize_image_if_needed,
    write_json,
)


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_and_resize_image(project_root, image_path, image_size):
    with Image.open(resolve_image_path(project_root, image_path)) as image_file:
        return resize_image_if_needed(image_file.convert("RGB"), image_size)


def build_pair_batch(processor, batch_rows, project_root, image_size, model):
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    nc_outputs = []
    ic_outputs = []
    metadata = []
    for row in batch_rows:
        image = load_and_resize_image(project_root, row["image_path"], image_size)
        nc_pack = build_condition_pack(
            processor=processor,
            prompt=row["nc_prompt_text"],
            image=image,
            model=model,
            question=row["question"],
            position=row["position"],
            evidence_answer=None,
        )
        ic_pack = build_condition_pack(
            processor=processor,
            prompt=row["ic_prompt_text"],
            image=image,
            model=model,
            question=row["question"],
            position=row["position"],
            evidence_answer=row["conflict_answer"],
        )
        nc_outputs.append(nc_pack["inputs"])
        ic_outputs.append(ic_pack["inputs"])
        metadata.append(
            {
                "sample_id": row["sample_id"],
                "sample_key": row["sample_key"],
                "split": row["split"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "img_id": row["img_id"],
                "image_path": row["image_path"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "conflict_answer": row["conflict_answer"],
                "gold_token_id": int(row["gold_token_id"]),
                "conflict_token_id": int(row["conflict_token_id"]),
                "nc_spans": {
                    "image_span": [int(nc_pack["image_span"][0]), int(nc_pack["image_span"][1])],
                    "question_span": [int(nc_pack["question_span"][0]), int(nc_pack["question_span"][1])],
                    "evidence_span": [0, 0],
                    "image_locate_method": nc_pack["image_locate_method"],
                },
                "ic_spans": {
                    "image_span": [int(ic_pack["image_span"][0]), int(ic_pack["image_span"][1])],
                    "question_span": [int(ic_pack["question_span"][0]), int(ic_pack["question_span"][1])],
                    "evidence_span": [int(ic_pack["evidence_span"][0]), int(ic_pack["evidence_span"][1])],
                    "image_locate_method": ic_pack["image_locate_method"],
                },
                "nc_regions": {
                    "image_positions": [int(x) for x in nc_pack["image_positions"]],
                    "question_span": [int(nc_pack["question_span"][0]), int(nc_pack["question_span"][1])],
                },
                "ic_regions": {
                    "image_positions": [int(x) for x in ic_pack["image_positions"]],
                    "question_span": [int(ic_pack["question_span"][0]), int(ic_pack["question_span"][1])],
                    "evidence_span": [int(ic_pack["evidence_span"][0]), int(ic_pack["evidence_span"][1])],
                },
            }
        )

    return (
        collate_processor_outputs(nc_outputs, pad_token_id),
        collate_processor_outputs(ic_outputs, pad_token_id),
        metadata,
    )


def region_mass(vector, positions):
    if not positions:
        return 0.0
    return float(vector[positions].sum().item())


def span_positions(span, upper_bound):
    start = max(0, int(span[0]))
    end = min(int(span[1]), int(upper_bound))
    if end <= start:
        return []
    return list(range(start, end))


def extract_condition_atomic(bundle_inputs, batch_meta, model, output_head, final_norm, save_dtype):
    head_param = next(output_head.parameters())
    norm_param = None
    if final_norm is not None:
        try:
            norm_param = next(final_norm.parameters())
        except StopIteration:
            norm_param = None
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bundle_inputs = dict(bundle_inputs)
    bundle_inputs.pop("token_type_ids", None)
    attention_mask_cpu = bundle_inputs["attention_mask"].clone()
    bundle_inputs = move_to_device(bundle_inputs, device, model_dtype)
    positions = bundle_inputs["attention_mask"].sum(dim=1) - 1

    with torch.no_grad():
        out = model(
            **bundle_inputs,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if not getattr(out, "attentions", None):
        raise RuntimeError("Model did not return attentions; try eager attention if available.")

    hidden_states = out.hidden_states[1:]
    attentions = out.attentions
    batch_size = len(batch_meta)
    n_layers = len(hidden_states)
    hidden_dim = hidden_states[0].shape[-1]

    hidden_tensor = torch.zeros((batch_size, n_layers, hidden_dim), dtype=resolve_dtype(save_dtype))
    attn_img = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    attn_ctx = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    attn_q = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    score_gold = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    score_conflict = torch.zeros((batch_size, n_layers), dtype=torch.float32)

    batch_index = torch.arange(batch_size, device=device)
    for layer_idx, (hs, attn) in enumerate(zip(hidden_states, attentions)):
        layer_hidden = hs[batch_index, positions]
        hidden_tensor[:, layer_idx, :] = layer_hidden.detach().to(dtype=resolve_dtype(save_dtype)).cpu()

        if final_norm is not None:
            norm_device = norm_param.device if norm_param is not None else head_param.device
            norm_dtype = norm_param.dtype if norm_param is not None else head_param.dtype
            normed_hidden = final_norm(layer_hidden.to(device=norm_device, dtype=norm_dtype))
        else:
            normed_hidden = layer_hidden.to(device=head_param.device, dtype=head_param.dtype)
        layer_logits = output_head(normed_hidden.to(device=head_param.device, dtype=head_param.dtype)).float().cpu()

        for item_idx, meta in enumerate(batch_meta):
            prompt_len = int(attention_mask_cpu[item_idx].sum().item())
            gold_id = int(meta["gold_token_id"])
            conflict_id = int(meta["conflict_token_id"])
            score_gold[item_idx, layer_idx] = float(layer_logits[item_idx, gold_id].item())
            score_conflict[item_idx, layer_idx] = float(layer_logits[item_idx, conflict_id].item())

            query_pos = int(positions[item_idx].item())
            mean_attn = attn[item_idx, :, query_pos, :prompt_len].detach().float().mean(dim=0).cpu()
            regions = meta["regions"]
            attn_img[item_idx, layer_idx] = region_mass(mean_attn, regions["image_positions"])
            attn_ctx[item_idx, layer_idx] = region_mass(mean_attn, regions["ctx_positions"])
            attn_q[item_idx, layer_idx] = region_mass(mean_attn, regions["q_positions"])

    return {
        "hidden_states": hidden_tensor,
        "attn_img": attn_img,
        "attn_ctx": attn_ctx,
        "attn_q": attn_q,
        "score_gold": score_gold,
        "score_conflict": score_conflict,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }


def _find_attention_tensor(attn_output):
    if torch.is_tensor(attn_output) and attn_output.dim() == 4:
        return attn_output
    if isinstance(attn_output, (tuple, list)):
        for item in attn_output:
            found = _find_attention_tensor(item)
            if found is not None:
                return found
    if hasattr(attn_output, "attn_weights"):
        found = _find_attention_tensor(getattr(attn_output, "attn_weights"))
        if found is not None:
            return found
    return None


def extract_condition_atomic_compact(bundle_inputs, batch_meta, model, output_head, final_norm, save_dtype):
    head_param = next(output_head.parameters())
    norm_param = None
    if final_norm is not None:
        try:
            norm_param = next(final_norm.parameters())
        except StopIteration:
            norm_param = None

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    bundle_inputs = dict(bundle_inputs)
    bundle_inputs.pop("token_type_ids", None)
    attention_mask_cpu = bundle_inputs["attention_mask"].clone()
    bundle_inputs = move_to_device(bundle_inputs, device, model_dtype)
    positions = bundle_inputs["attention_mask"].sum(dim=1) - 1

    batch_size = len(batch_meta)
    attn_modules = find_self_attn_modules(model)
    n_layers = len(attn_modules)
    attn_img = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    attn_ctx = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    attn_q = torch.zeros((batch_size, n_layers), dtype=torch.float32)

    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            attn = _find_attention_tensor(output)
            if attn is None:
                return
            attn = attn.detach().float()
            for item_idx, meta in enumerate(batch_meta):
                prompt_len = int(attention_mask_cpu[item_idx].sum().item())
                query_pos = int(positions[item_idx].item())
                mean_attn = attn[item_idx, :, query_pos, :prompt_len].mean(dim=0).cpu()
                regions = meta["regions"]
                attn_img[item_idx, layer_idx] = region_mass(mean_attn, regions["image_positions"])
                attn_ctx[item_idx, layer_idx] = region_mass(mean_attn, regions["ctx_positions"])
                attn_q[item_idx, layer_idx] = region_mass(mean_attn, regions["q_positions"])
        return hook

    for layer_idx, module in attn_modules.items():
        handles.append(module.register_forward_hook(make_hook(int(layer_idx))))

    try:
        with torch.no_grad():
            out = model(
                **bundle_inputs,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    hidden_states = out.hidden_states[1:]
    hidden_dim = hidden_states[0].shape[-1]
    hidden_tensor = torch.zeros((batch_size, n_layers, hidden_dim), dtype=resolve_dtype(save_dtype))
    score_gold = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    score_conflict = torch.zeros((batch_size, n_layers), dtype=torch.float32)
    batch_index = torch.arange(batch_size, device=device)

    for layer_idx, hs in enumerate(hidden_states):
        layer_hidden = hs[batch_index, positions]
        hidden_tensor[:, layer_idx, :] = layer_hidden.detach().to(dtype=resolve_dtype(save_dtype)).cpu()
        if final_norm is not None:
            norm_device = norm_param.device if norm_param is not None else head_param.device
            norm_dtype = norm_param.dtype if norm_param is not None else head_param.dtype
            normed_hidden = final_norm(layer_hidden.to(device=norm_device, dtype=norm_dtype))
        else:
            normed_hidden = layer_hidden.to(device=head_param.device, dtype=head_param.dtype)
        layer_logits = output_head(normed_hidden.to(device=head_param.device, dtype=head_param.dtype)).float().cpu()

        for item_idx, meta in enumerate(batch_meta):
            score_gold[item_idx, layer_idx] = float(layer_logits[item_idx, int(meta["gold_token_id"])].item())
            score_conflict[item_idx, layer_idx] = float(layer_logits[item_idx, int(meta["conflict_token_id"])].item())

    if float(attn_img.abs().sum().item()) == 0.0 and float(attn_ctx.abs().sum().item()) == 0.0 and float(attn_q.abs().sum().item()) == 0.0:
        raise RuntimeError("Compact attention hooks did not capture any attention mass. Try --attention_mode full.")

    return {
        "hidden_states": hidden_tensor,
        "attn_img": attn_img,
        "attn_ctx": attn_ctx,
        "attn_q": attn_q,
        "score_gold": score_gold,
        "score_conflict": score_conflict,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }


def augment_rows_with_token_ids(rows, tokenizer):
    kept = []
    skipped = []
    for row in rows:
        gold_id = first_answer_token_id(tokenizer, row["gold_answer"])
        conflict_id = first_answer_token_id(tokenizer, row["conflict_answer"])
        if gold_id is None or conflict_id is None:
            skipped.append(
                {
                    "sample_id": row["sample_id"],
                    "sample_key": row["sample_key"],
                    "gold_answer": row["gold_answer"],
                    "conflict_answer": row["conflict_answer"],
                    "reason": "tokenization_failed",
                }
            )
            continue
        item = dict(row)
        item["gold_token_id"] = int(gold_id)
        item["conflict_token_id"] = int(conflict_id)
        kept.append(item)
    return kept, skipped


def main():
    ap = argparse.ArgumentParser(description="Dump atomic CHDP features for paired NC/IC samples.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_implementation", default="eager")
    ap.add_argument("--attention_mode", default="compact", choices=["compact", "full"])
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    ensure_video_import_compat()
    project_root = Path(__file__).resolve().parents[1]
    rows = read_jsonl(args.manifest)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows, skipped_rows = augment_rows_with_token_ids(rows, processor.tokenizer)

    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    output_head = get_output_head(model)
    final_norm = get_final_norm(model)

    collected_meta = []
    nc_hidden_batches = []
    ic_hidden_batches = []
    nc_attn_img_batches = []
    nc_attn_ctx_batches = []
    nc_attn_q_batches = []
    ic_attn_img_batches = []
    ic_attn_ctx_batches = []
    ic_attn_q_batches = []
    nc_score_gold_batches = []
    nc_score_conflict_batches = []
    ic_score_gold_batches = []
    ic_score_conflict_batches = []

    batches = list(chunked(rows, args.batch_size))
    progress = tqdm(batches, total=len(batches), desc="dump atomic features", unit="batch")
    n_layers = None
    hidden_dim = None
    for batch_rows in progress:
        nc_inputs, ic_inputs, batch_meta = build_pair_batch(
            processor=processor,
            batch_rows=batch_rows,
            project_root=project_root,
            image_size=args.image_size,
            model=model,
        )

        nc_meta = []
        ic_meta = []
        for meta in batch_meta:
            nc_meta.append(
                {
                    "gold_token_id": meta["gold_token_id"],
                    "conflict_token_id": meta["conflict_token_id"],
                    "regions": {
                        "image_positions": meta["nc_regions"]["image_positions"],
                        "ctx_positions": [],
                        "q_positions": span_positions(meta["nc_regions"]["question_span"], len(meta["nc_regions"]["image_positions"]) + 100000),
                    },
                }
            )
            ic_meta.append(
                {
                    "gold_token_id": meta["gold_token_id"],
                    "conflict_token_id": meta["conflict_token_id"],
                    "regions": {
                        "image_positions": meta["ic_regions"]["image_positions"],
                        "ctx_positions": span_positions(meta["ic_regions"]["evidence_span"], 100000),
                        "q_positions": span_positions(meta["ic_regions"]["question_span"], 100000),
                    },
                }
            )

        extractor = extract_condition_atomic_compact if args.attention_mode == "compact" else extract_condition_atomic
        nc_atomic = extractor(
            bundle_inputs=nc_inputs,
            batch_meta=nc_meta,
            model=model,
            output_head=output_head,
            final_norm=final_norm,
            save_dtype=args.save_dtype,
        )
        ic_atomic = extractor(
            bundle_inputs=ic_inputs,
            batch_meta=ic_meta,
            model=model,
            output_head=output_head,
            final_norm=final_norm,
            save_dtype=args.save_dtype,
        )

        if n_layers is None:
            n_layers = nc_atomic["n_layers"]
            hidden_dim = nc_atomic["hidden_dim"]

        nc_hidden_batches.append(nc_atomic["hidden_states"])
        ic_hidden_batches.append(ic_atomic["hidden_states"])
        nc_attn_img_batches.append(nc_atomic["attn_img"])
        nc_attn_ctx_batches.append(nc_atomic["attn_ctx"])
        nc_attn_q_batches.append(nc_atomic["attn_q"])
        ic_attn_img_batches.append(ic_atomic["attn_img"])
        ic_attn_ctx_batches.append(ic_atomic["attn_ctx"])
        ic_attn_q_batches.append(ic_atomic["attn_q"])
        nc_score_gold_batches.append(nc_atomic["score_gold"])
        nc_score_conflict_batches.append(nc_atomic["score_conflict"])
        ic_score_gold_batches.append(ic_atomic["score_gold"])
        ic_score_conflict_batches.append(ic_atomic["score_conflict"])
        collected_meta.extend(batch_meta)
        progress.set_postfix(samples=len(collected_meta))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "manifest_path": str(args.manifest),
        "model_name": args.model,
        "attention_mode": args.attention_mode,
        "target_position": "question_end_or_assistant_start",
        "feature_note": "scores use first answer token logit after final norm + lm_head",
        "sample_ids": [item["sample_id"] for item in collected_meta],
        "sample_keys": [item["sample_key"] for item in collected_meta],
        "labels": torch.tensor([int(item["label"]) for item in collected_meta], dtype=torch.long),
        "label_names": [item["label_name"] for item in collected_meta],
        "splits": [item["split"] for item in collected_meta],
        "img_ids": [item["img_id"] for item in collected_meta],
        "image_paths": [item["image_path"] for item in collected_meta],
        "questions": [item["question"] for item in collected_meta],
        "gold_answers": [item["gold_answer"] for item in collected_meta],
        "conflict_answers": [item["conflict_answer"] for item in collected_meta],
        "gold_token_ids": torch.tensor([int(item["gold_token_id"]) for item in collected_meta], dtype=torch.long),
        "conflict_token_ids": torch.tensor([int(item["conflict_token_id"]) for item in collected_meta], dtype=torch.long),
        "spans": {
            "nc": [item["nc_spans"] for item in collected_meta],
            "ic": [item["ic_spans"] for item in collected_meta],
        },
        "nc_hidden_states": torch.cat(nc_hidden_batches, dim=0) if nc_hidden_batches else torch.zeros((0, 0, 0)),
        "ic_hidden_states": torch.cat(ic_hidden_batches, dim=0) if ic_hidden_batches else torch.zeros((0, 0, 0)),
        "nc_attn_img": torch.cat(nc_attn_img_batches, dim=0) if nc_attn_img_batches else torch.zeros((0, 0)),
        "nc_attn_ctx": torch.cat(nc_attn_ctx_batches, dim=0) if nc_attn_ctx_batches else torch.zeros((0, 0)),
        "nc_attn_q": torch.cat(nc_attn_q_batches, dim=0) if nc_attn_q_batches else torch.zeros((0, 0)),
        "ic_attn_img": torch.cat(ic_attn_img_batches, dim=0) if ic_attn_img_batches else torch.zeros((0, 0)),
        "ic_attn_ctx": torch.cat(ic_attn_ctx_batches, dim=0) if ic_attn_ctx_batches else torch.zeros((0, 0)),
        "ic_attn_q": torch.cat(ic_attn_q_batches, dim=0) if ic_attn_q_batches else torch.zeros((0, 0)),
        "nc_score_gold": torch.cat(nc_score_gold_batches, dim=0) if nc_score_gold_batches else torch.zeros((0, 0)),
        "nc_score_conflict": torch.cat(nc_score_conflict_batches, dim=0) if nc_score_conflict_batches else torch.zeros((0, 0)),
        "ic_score_gold": torch.cat(ic_score_gold_batches, dim=0) if ic_score_gold_batches else torch.zeros((0, 0)),
        "ic_score_conflict": torch.cat(ic_score_conflict_batches, dim=0) if ic_score_conflict_batches else torch.zeros((0, 0)),
        "n_layers": int(n_layers or 0),
        "hidden_dim": int(hidden_dim or 0),
    }
    torch.save(bundle, out_path)

    summary_path = out_path.with_suffix(".summary.json")
    label_counts = {"resist": 0, "hijack": 0}
    for name in bundle["label_names"]:
        label_counts[str(name)] += 1
    write_json(
        summary_path,
        {
            "manifest_path": str(args.manifest),
            "model_name": args.model,
            "attention_mode": args.attention_mode,
            "out_path": str(out_path),
            "n_samples": len(collected_meta),
            "n_layers": int(n_layers or 0),
            "hidden_dim": int(hidden_dim or 0),
            "label_counts": label_counts,
            "skipped_tokenization_rows": skipped_rows,
        },
    )

    print(f"saved_bundle={out_path}")
    print(f"saved_summary={summary_path}")
    print(f"n_samples={len(collected_meta)}")
    print(f"n_layers={int(n_layers or 0)}")
    print(f"hidden_dim={int(hidden_dim or 0)}")


if __name__ == "__main__":
    main()
