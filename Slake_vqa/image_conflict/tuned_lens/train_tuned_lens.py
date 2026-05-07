import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from common import (
    answer_token_ids,
    build_batch_inputs,
    ensure_video_import_compat,
    get_output_head,
    infer_hidden_shape,
    load_mm_model,
    move_to_device,
    read_jsonl,
)
from modeling import build_translators, parse_layer_spec, resolve_dtype
from model_utils import load_processor_with_compat, prefix_model_relative_path, resolve_model_selection
from resume_utils import ResumeTracker, build_resume_dir, build_resume_scope


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pick_device(name, fallback):
    if name == "auto":
        return fallback
    return torch.device(name)


def filter_rows(rows, prompt_types):
    wanted = {item.strip() for item in prompt_types.split(",") if item.strip()}
    return [row for row in rows if row["prompt_type"] in wanted]


def infer_position_from_manifest(path_str: str) -> str:
    stem = Path(path_str).stem
    for candidate in ("image_conflict", "before_question", "before_answer", "prefix"):
        if stem.endswith(candidate):
            return candidate
    return ""


def append_answer_tokens(batch_inputs, answer_token_lists, pad_token_id):
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    prompt_lens = attention_mask.sum(dim=1).tolist()

    extended_input_ids = []
    extended_attention_masks = []
    max_len = 0
    for idx in range(batch_size):
        prompt_len = int(prompt_lens[idx])
        answer_ids = list(answer_token_lists[idx])
        prompt_ids = input_ids[idx, :prompt_len]
        answer_tensor = torch.tensor(answer_ids, dtype=input_ids.dtype)
        extended_ids = torch.cat([prompt_ids, answer_tensor], dim=0)
        extended_mask = torch.ones(extended_ids.shape[0], dtype=attention_mask.dtype)
        extended_input_ids.append(extended_ids)
        extended_attention_masks.append(extended_mask)
        max_len = max(max_len, int(extended_ids.shape[0]))

    padded_input_ids = []
    padded_attention_masks = []
    for ids, mask in zip(extended_input_ids, extended_attention_masks):
        pad_len = max_len - int(ids.shape[0])
        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)], dim=0)
        padded_input_ids.append(ids)
        padded_attention_masks.append(mask)

    extended_batch = {}
    for key, value in batch_inputs.items():
        if key == "input_ids":
            extended_batch[key] = torch.stack(padded_input_ids, dim=0)
        elif key == "attention_mask":
            extended_batch[key] = torch.stack(padded_attention_masks, dim=0)
        else:
            extended_batch[key] = value
    return extended_batch


def compute_batch_loss(model, processor, translators, rows, layer_indices, output_head, args, first_device, lens_device, project_root):
    batch_inputs = build_batch_inputs(processor, rows, project_root, image_size=args.image_size, mask_scale=args.mask_scale)
    batch_inputs.pop("token_type_ids", None)
    prompt_lengths = batch_inputs["attention_mask"].sum(dim=1).tolist()
    answer_token_lists = [row["answer_token_ids"] for row in rows]
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0
    batch_inputs = append_answer_tokens(batch_inputs, answer_token_lists, pad_token_id)
    model_dtype = next(model.parameters()).dtype
    batch_inputs = move_to_device(batch_inputs, first_device, model_dtype)

    with torch.no_grad():
        out = model(**batch_inputs, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states[1:]
    logits = out.logits
    lm_device = output_head.weight.device
    target_probs_per_answer = []
    for item_idx, answer_ids in enumerate(answer_token_lists):
        pred_positions = torch.arange(
            prompt_lengths[item_idx] - 1,
            prompt_lengths[item_idx] - 1 + len(answer_ids),
            device=logits.device,
        )
        final_logits = logits[item_idx, pred_positions].detach()
        target_probs_per_answer.append(
            torch.softmax(
                final_logits.to(device=lm_device, dtype=output_head.weight.dtype) / args.kl_temperature,
                dim=-1,
            )
        )
    target_probs = torch.cat(target_probs_per_answer, dim=0)

    losses = []
    for layer_idx in layer_indices:
        hs_tensor = hidden_states[layer_idx]
        hs_per_answer = []
        for item_idx, answer_ids in enumerate(answer_token_lists):
            pred_positions = torch.arange(
                prompt_lengths[item_idx] - 1,
                prompt_lengths[item_idx] - 1 + len(answer_ids),
                device=hs_tensor.device,
            )
            hs_per_answer.append(hs_tensor[item_idx, pred_positions])
        hs = torch.cat(hs_per_answer, dim=0)
        translator = translators[str(layer_idx)]
        translator_dtype = next(translator.parameters()).dtype
        hs_lens = hs.to(device=lens_device, dtype=translator_dtype)
        pred_hidden = translator(hs_lens)
        pred_logits = output_head(pred_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
        pred_log_probs = torch.log_softmax(pred_logits / args.kl_temperature, dim=-1)
        kl_loss = F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
        identity_loss = F.mse_loss(pred_hidden, hs_lens)
        losses.append(kl_loss + args.identity_reg_weight * identity_loss)
    return torch.stack(losses).mean()


def evaluate(model, processor, translators, rows, layer_indices, output_head, args, first_device, lens_device, project_root):
    if not rows:
        return None
    total = 0.0
    count = 0
    batches = list(chunked(rows, args.batch_size))
    pbar = tqdm(
        batches,
        total=len(batches),
        desc="eval",
        unit="batch",
        leave=False,
    )
    for batch_rows in pbar:
        loss = compute_batch_loss(
            model,
            processor,
            translators,
            batch_rows,
            layer_indices,
            output_head,
            args,
            first_device,
            lens_device,
            project_root,
        )
        total += loss.item() * len(batch_rows)
        count += len(batch_rows)
        pbar.set_postfix(running=f"{total / max(1, count):.4f}")
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description="Train multimodal tuned-lens translators on next-token distributions.")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="nc,ic")
    ap.add_argument("--layer_spec", default="")
    ap.add_argument("--translator_rank", type=int, default=0, help="0 means full affine translators.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--translator_dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=672, help="Resize each image so its longest side is at most image_size before processing. Set <=0 to disable.")
    ap.add_argument("--mask_scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--kl_temperature", type=float, default=1.0)
    ap.add_argument("--identity_reg_weight", type=float, default=1e-3)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.model, _model_spec = resolve_model_selection(model_name=args.model_name, model=args.model)
    args.out_dir = prefix_model_relative_path(args.out_dir, model_name=args.model_name, model=args.model)
    position = infer_position_from_manifest(args.train_manifest or args.val_manifest)
    resume_scope = build_resume_scope(args.train_manifest, args.val_manifest)
    tracker = ResumeTracker(
        resume_dir=build_resume_dir(
            project_root=PARENT_DIR,
            task_name="train_tuned_lens",
            model_name=args.model_name,
            model=args.model,
            position=position,
            scope=resume_scope,
        ),
        enabled=args.resume,
    )
    tracker.start(
        task="train_tuned_lens",
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        out_dir=args.out_dir,
        position=position,
    )

    ensure_video_import_compat()
    project_root = Path(__file__).resolve().parents[1]
    train_rows = filter_rows(read_jsonl(args.train_manifest), args.prompt_types)
    val_rows = filter_rows(read_jsonl(args.val_manifest), args.prompt_types) if args.val_manifest else []
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]
    if not train_rows:
        raise RuntimeError("No train rows after filtering prompt types.")

    processor = load_processor_with_compat(args.model)
    tokenizer = processor.tokenizer
    skipped_unencodable_train = 0
    skipped_unencodable_val = 0
    filtered_train_rows = []
    for row in train_rows:
        answer_ids = answer_token_ids(tokenizer, row["gold_answer"])
        if not answer_ids:
            skipped_unencodable_train += 1
            continue
        row = dict(row)
        row["answer_token_ids"] = answer_ids
        filtered_train_rows.append(row)
    filtered_val_rows = []
    for row in val_rows:
        answer_ids = answer_token_ids(tokenizer, row["gold_answer"])
        if not answer_ids:
            skipped_unencodable_val += 1
            continue
        row = dict(row)
        row["answer_token_ids"] = answer_ids
        filtered_val_rows.append(row)
    train_rows = filtered_train_rows
    val_rows = filtered_val_rows
    if not train_rows:
        raise RuntimeError("No train rows left after answer tokenization.")
    model = load_mm_model(
        args.model,
        device_map=args.device_map,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_head = get_output_head(model)
    first_device = next(model.parameters()).device
    lens_device = pick_device(args.lens_device, first_device)

    n_layers, hidden_dim = infer_hidden_shape(
        model,
        processor,
        train_rows[0],
        project_root,
        first_device,
        image_size=args.image_size,
        mask_scale=float(args.mask_scale),
    )
    layer_indices = parse_layer_spec(args.layer_spec, n_layers)
    translators = build_translators(
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        dtype=resolve_dtype(args.translator_dtype),
        rank=args.translator_rank,
    ).to(lens_device)

    optimizer = torch.optim.AdamW(translators.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_ckpt_path = tracker.resume_dir / "train_resume.pt"
    history = []
    best_state = None
    best_val = None
    start_epoch = 0
    resume_step = 1
    resume_total = 0.0
    resume_count = 0
    if args.resume and resume_ckpt_path.exists():
        resume_payload = torch.load(resume_ckpt_path, map_location="cpu")
        translators.load_state_dict(resume_payload["translators_state"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        history = list(resume_payload.get("history", []))
        best_state = resume_payload.get("best_state")
        best_val = resume_payload.get("best_val")
        start_epoch = int(resume_payload.get("epoch", 0))
        resume_step = int(resume_payload.get("step", 1))
        resume_total = float(resume_payload.get("running_total", 0.0))
        resume_count = int(resume_payload.get("running_count", 0))

    print(
        json.dumps(
            {
                "n_train_rows": len(train_rows),
                "n_val_rows": len(val_rows),
                "layer_indices": layer_indices,
                "hidden_dim": hidden_dim,
                "translator_rank": args.translator_rank,
                "translator_dtype": args.translator_dtype,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "mask_scale": float(args.mask_scale),
                "epochs": args.epochs,
                "lr": args.lr,
                "identity_reg_weight": args.identity_reg_weight,
                "answer_supervision": "gold_full_sequence",
                "skipped_unencodable_train": skipped_unencodable_train,
                "skipped_unencodable_val": skipped_unencodable_val,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(start_epoch, args.epochs):
        total = resume_total if epoch == start_epoch else 0.0
        count = resume_count if epoch == start_epoch else 0
        batches = list(chunked(train_rows, args.batch_size))
        pbar = tqdm(
            enumerate(batches, start=1),
            total=len(batches),
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for step_idx, batch_rows in pbar:
            if epoch == start_epoch and step_idx < resume_step:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = compute_batch_loss(
                model,
                processor,
                translators,
                batch_rows,
                layer_indices,
                output_head,
                args,
                first_device,
                lens_device,
                project_root,
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_rows)
            count += len(batch_rows)
            running_loss = total / max(1, count)
            if step_idx == 1 or step_idx % args.log_every == 0 or step_idx == len(batches):
                pbar.set_postfix(loss=f"{loss.item():.4f}", running=f"{running_loss:.4f}")
                if args.resume:
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step_idx + 1,
                            "running_total": total,
                            "running_count": count,
                            "history": history,
                            "best_state": best_state,
                            "best_val": best_val,
                            "translators_state": {k: v.detach().cpu() for k, v in translators.state_dict().items()},
                            "optimizer_state": optimizer.state_dict(),
                        },
                        resume_ckpt_path,
                    )
                    tracker.update(epoch=epoch, step=step_idx, running_total=total, running_count=count)

        train_loss = total / max(1, count)
        val_loss = evaluate(
            model,
            processor,
            translators,
            val_rows,
            layer_indices,
            output_head,
            args,
            first_device,
            lens_device,
            project_root,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, ensure_ascii=False))

        if val_loss is None:
            best_state = {k: v.detach().cpu() for k, v in translators.state_dict().items()}
        elif best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in translators.state_dict().items()}
        if args.resume:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": 1,
                    "running_total": 0.0,
                    "running_count": 0,
                    "history": history,
                    "best_state": best_state,
                    "best_val": best_val,
                    "translators_state": {k: v.detach().cpu() for k, v in translators.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                },
                resume_ckpt_path,
            )
            tracker.update(epoch=epoch + 1, step=1, best_val=best_val, history_len=len(history))

    if best_state is not None:
        translators.load_state_dict(best_state, strict=True)

    ckpt = {
        "layer_indices": layer_indices,
        "translator_rank": args.translator_rank,
        "translator_dtype": args.translator_dtype,
        "identity_reg_weight": args.identity_reg_weight,
        "answer_supervision": "gold_full_sequence",
        "hidden_dim": hidden_dim,
        "state_dict": {k: v.detach().cpu() for k, v in translators.state_dict().items()},
        "history": history,
        "prompt_types": args.prompt_types,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "model": args.model,
    }
    ckpt_path = out_dir / "tuned_lens.pt"
    torch.save(ckpt, ckpt_path)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "layer_indices": layer_indices,
                "translator_rank": args.translator_rank,
                "history": history,
                "image_size": args.image_size,
                "mask_scale": float(args.mask_scale),
                "answer_supervision": "gold_full_sequence",
                "skipped_unencodable_train": skipped_unencodable_train,
                "skipped_unencodable_val": skipped_unencodable_val,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    tracker.finish(checkpoint=str(ckpt_path), summary_json=str(out_dir / "summary.json"), epochs=args.epochs)

    print(f"saved_checkpoint={ckpt_path}")
    print(f"saved_summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
