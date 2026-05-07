import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import ensure_padding_token, read_jsonl, wrap_as_chat
from modeling import build_translators, parse_layer_spec, resolve_dtype


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


def compute_batch_loss(model, tokenizer, translators, rows, layer_indices, output_head, args, first_device, lens_device):
    prompts = [
        wrap_as_chat(tokenizer, row["prompt_text"], enable_thinking=args.enable_thinking)
        for row in rows
    ]
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(first_device) for k, v in enc.items()}
    positions = enc["attention_mask"].sum(dim=1) - 1

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states[1:]
    logits = out.logits
    logits_batch_index = torch.arange(len(rows), device=logits.device)
    logits_positions = positions.to(logits.device)
    final_logits = logits[logits_batch_index, logits_positions].detach()
    lm_device = output_head.weight.device
    target_probs = torch.softmax(final_logits.to(lm_device) / args.kl_temperature, dim=-1)

    losses = []
    for layer_idx in layer_indices:
        hs_tensor = hidden_states[layer_idx]
        batch_index = torch.arange(len(rows), device=hs_tensor.device)
        local_positions = positions.to(hs_tensor.device)
        translator = translators[str(layer_idx)]
        translator_dtype = next(translator.parameters()).dtype
        hs = hs_tensor[batch_index, local_positions].to(device=lens_device, dtype=translator_dtype)
        pred_hidden = translator(hs)
        pred_logits = output_head(pred_hidden.to(device=lm_device, dtype=output_head.weight.dtype))
        pred_log_probs = torch.log_softmax(pred_logits / args.kl_temperature, dim=-1)
        losses.append(F.kl_div(pred_log_probs, target_probs, reduction="batchmean"))
    return torch.stack(losses).mean()


def evaluate(model, tokenizer, translators, rows, layer_indices, output_head, args, first_device, lens_device):
    if not rows:
        return None
    total = 0.0
    count = 0
    for batch_rows in chunked(rows, args.batch_size):
        loss = compute_batch_loss(
            model,
            tokenizer,
            translators,
            batch_rows,
            layer_indices,
            output_head,
            args,
            first_device,
            lens_device,
        )
        total += loss.item() * len(batch_rows)
        count += len(batch_rows)
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description="Train tuned-lens translators on next-token distributions.")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_types", default="base,support,conflict")
    ap.add_argument("--layer_spec", default="")
    ap.add_argument("--translator_rank", type=int, default=0, help="0 means full affine translators.")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--translator_dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--lens_device", default="auto")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--kl_temperature", type=float, default=1.0)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    train_rows = filter_rows(read_jsonl(args.train_manifest), args.prompt_types)
    val_rows = filter_rows(read_jsonl(args.val_manifest), args.prompt_types) if args.val_manifest else []
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]

    tok = ensure_padding_token(AutoTokenizer.from_pretrained(args.model, trust_remote_code=True))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    output_head = model.get_output_embeddings()
    if output_head is None:
        raise RuntimeError("Model does not expose output embeddings.")

    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        raise RuntimeError("Could not infer num_hidden_layers from model config.")
    hidden_dim = getattr(model.config, "hidden_size", None)
    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden_size from model config.")

    layer_indices = parse_layer_spec(args.layer_spec, n_layers)
    first_device = next(model.parameters()).device
    lens_device = pick_device(args.lens_device, first_device)
    translators = build_translators(
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        dtype=resolve_dtype(args.translator_dtype),
        rank=args.translator_rank,
    ).to(lens_device)

    optimizer = torch.optim.AdamW(translators.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_state = None
    best_val = None

    print(
        json.dumps(
            {
                "n_train_rows": len(train_rows),
                "n_val_rows": len(val_rows),
                "layer_indices": layer_indices,
                "translator_rank": args.translator_rank,
                "translator_dtype": args.translator_dtype,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(args.epochs):
        total = 0.0
        count = 0
        batches = list(chunked(train_rows, args.batch_size))
        pbar = tqdm(
            enumerate(batches, start=1),
            total=len(batches),
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        for step_idx, batch_rows in pbar:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_batch_loss(
                model,
                tok,
                translators,
                batch_rows,
                layer_indices,
                output_head,
                args,
                first_device,
                lens_device,
            )
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_rows)
            count += len(batch_rows)
            running_loss = total / max(1, count)
            if step_idx == 1 or step_idx % args.log_every == 0 or step_idx == len(batches):
                pbar.set_postfix(loss=f"{loss.item():.4f}", running=f"{running_loss:.4f}")

        train_loss = total / max(1, count)
        val_loss = evaluate(
            model,
            tok,
            translators,
            val_rows,
            layer_indices,
            output_head,
            args,
            first_device,
            lens_device,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ensure_ascii=False,
            )
        )

        if val_loss is None:
            best_state = translators.state_dict()
        elif best_val is None or val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in translators.state_dict().items()}

    if best_state is not None:
        translators.load_state_dict(best_state, strict=True)

    ckpt = {
        "layer_indices": layer_indices,
        "translator_rank": args.translator_rank,
        "translator_dtype": args.translator_dtype,
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
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_checkpoint={ckpt_path}")
    print(f"saved_summary={out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
