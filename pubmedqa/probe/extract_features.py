import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from common import read_jsonl, wrap_as_chat


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_dtype(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def main():
    ap = argparse.ArgumentParser(description="Extract last-token hidden states for probe training.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.eval()

    first_device = next(model.parameters()).device

    hidden_batches = []
    sample_ids = []
    pair_ids = []
    sides = []
    prompt_types = []
    gold_answers = []
    conflict_answers = []
    conflict_targets = []
    follow_targets = []
    splits = []

    for batch_rows in chunked(rows, args.batch_size):
        prompts = [
            wrap_as_chat(tok, row["prompt_text"], enable_thinking=args.enable_thinking)
            for row in batch_rows
        ]
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(first_device) for k, v in enc.items()}
        positions = enc["attention_mask"].sum(dim=1) - 1

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        hidden_states = list(out.hidden_states)
        if not args.include_embeddings:
            hidden_states = hidden_states[1:]

        layer_states = []
        for hs in hidden_states:
            batch_index = torch.arange(len(batch_rows), device=hs.device)
            local_positions = positions.to(hs.device)
            last_token = hs[batch_index, local_positions]
            layer_states.append(last_token.detach().to(resolve_dtype(args.save_dtype)).cpu())
        hidden_batches.append(torch.stack(layer_states, dim=1))

        for row in batch_rows:
            sample_ids.append(row["record_id"])
            pair_ids.append(row["pair_id"])
            sides.append(row["side"])
            prompt_types.append(row["prompt_type"])
            gold_answers.append(row["gold_answer"])
            conflict_answers.append(row["conflict_answer"])
            conflict_targets.append(row["conflict_target"])
            follow_targets.append(row["follow_target"])
            splits.append(row["split"])

    hidden_tensor = torch.cat(hidden_batches, dim=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "manifest_path": str(manifest_path),
            "hidden_states": hidden_tensor,
            "sample_ids": sample_ids,
            "pair_ids": pair_ids,
            "sides": sides,
            "prompt_types": prompt_types,
            "gold_answers": gold_answers,
            "conflict_answers": conflict_answers,
            "conflict_targets": torch.tensor(conflict_targets, dtype=torch.long),
            "follow_targets": torch.tensor(follow_targets, dtype=torch.long),
            "splits": splits,
            "include_embeddings": args.include_embeddings,
            "n_layers": hidden_tensor.shape[1],
            "hidden_dim": hidden_tensor.shape[2],
        },
        out_path,
    )

    print(f"saved={out_path}")
    print(f"n_samples={hidden_tensor.shape[0]}")
    print(f"n_layers={hidden_tensor.shape[1]}")
    print(f"hidden_dim={hidden_tensor.shape[2]}")


if __name__ == "__main__":
    main()
