import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


REPO_ROOT = Path("/root/logit_lens")
ABALTE_HEAD_PATH = REPO_ROOT / "VQA_RAD/Hulu-med/ablate_head.py"
DEFAULT_MODEL = "/root/autodl-tmp/Hulu-Med-4B"
DEFAULT_DATA_CSV = str(REPO_ROOT / "VQA_RAD/Hulu-med/data/nc_cc_both_correct_rerun_tmp_val.csv")
DEFAULT_SELECTED_HEADS = str(
    REPO_ROOT
    / "VQA_RAD/Hulu-med/result_train_hulumed4b_before_question/"
      "headscan_vqarad_mm_hulumed4b_before_question/selected_heads_stable_hulumed4b.json"
)
DEFAULT_OUT_DIR = Path("/root/logit_lens/exp/Computational_cost")


def load_ablate_module(ablate_head_path: str):
    ablate_head_path = Path(ablate_head_path)
    module_dir = str(ablate_head_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("benchmark_ablate_head", ablate_head_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model(module, model_name: str, device: str, dtype):
    if hasattr(module, "load_mm_model"):
        model = module.load_mm_model(
            model_name,
            device_map=None,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()
        return model

    if hasattr(module, "shared_load_mm_model"):
        model = module.shared_load_mm_model(
            model_name,
            device_map=None,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()
        return model

    model = None
    load_errs = []
    for cls in [module.AutoModelForVision2Seq, module.AutoModelForCausalLM]:
        try:
            model = cls.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            break
        except Exception as exc:
            load_errs.append(f"{cls.__name__}: {repr(exc)}")
    if model is None:
        raise RuntimeError("Failed to load model. " + " | ".join(load_errs))
    model = model.to(device)
    model.eval()
    return model


def select_sample(module, csv_path: str, image_root: str, sample_index: int):
    df = pd.read_csv(csv_path)
    row = df.iloc[sample_index]

    image_col = module.infer_col(df, ["image", "image_path", "img_path", "img", "path"])
    q_col = module.infer_col(df, ["question", "query"])
    gold_col = module.infer_col(df, ["answer", "gold", "gt_answer", "label"])
    wrong_col = module.infer_col(df, ["wrong", "conflict", "wrong_answer"], required=False)

    question = module.normalize_text(row[q_col])
    gold = module.normalize_text(row[gold_col]).lower()
    if wrong_col is not None and module.normalize_text(row[wrong_col]):
        wrong = module.normalize_text(row[wrong_col]).lower()
    else:
        wrong = module.invert_yesno(gold)

    img_path = module.resolve_image_path(image_root, module.normalize_text(row[image_col]))
    return {
        "row_idx": int(row.get("id", sample_index)),
        "question": question,
        "gold": gold,
        "wrong": wrong,
        "img_path": img_path,
    }


def build_sample_inputs(module, processor, sample: dict, position: str, device: str, max_image_side: int):
    nc_prompt, ctx_prompt = module.make_prompts(
        question=sample["question"],
        gold=sample["gold"],
        wrong=sample["wrong"],
        position=position,
        trace_mode="conflict",
    )
    image = module.resize_image_max_side(Image.open(sample["img_path"]), max_side=max_image_side)
    nc_inputs = module.build_inputs_mm(processor, image, nc_prompt, "cpu")
    ctx_inputs = module.build_inputs_mm(processor, image, ctx_prompt, "cpu")
    return {
        "image_size": list(image.size),
        "nc_prompt": nc_prompt,
        "ctx_prompt": ctx_prompt,
        "nc_inputs": nc_inputs,
        "ctx_inputs": ctx_inputs,
        "device": device,
    }


def sync_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_if_needed(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def run_ctx_pipeline(module, model, tokenizer, ctx_inputs, gold: str, wrong: str, device: str):
    prompt_cache = module.build_prompt_cache_from_inputs(model, ctx_inputs)
    try:
        scores = module.score_answer_candidates_with_cache(
            model=model,
            tokenizer=tokenizer,
            prompt_cache=prompt_cache,
            gold=gold,
            wrong=wrong,
            trace_mode="conflict",
            device=device,
        )
        metrics = module.compute_scalar_metrics(scores, "conflict")
        pred = module.pred_label(scores)
        return {
            "scores": scores,
            "metrics": metrics,
            "pred": pred,
        }
    finally:
        del prompt_cache


def benchmark_scenario(
    module,
    model,
    tokenizer,
    ctx_inputs,
    gold: str,
    wrong: str,
    device: str,
    warmup: int,
    repeats: int,
):
    for _ in range(warmup):
        _ = run_ctx_pipeline(module, model, tokenizer, ctx_inputs, gold, wrong, device)
        sync_if_needed(device)

    elapsed_ms = []
    peak_memories = []
    last_result = None

    for _ in range(repeats):
        reset_peak_memory_if_needed(device)
        sync_if_needed(device)
        start = time.perf_counter()
        last_result = run_ctx_pipeline(module, model, tokenizer, ctx_inputs, gold, wrong, device)
        sync_if_needed(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
        peak_memories.append(peak_memory_mb(device))

    assert last_result is not None
    return {
        "warmup": warmup,
        "repeats": repeats,
        "elapsed_ms": elapsed_ms,
        "elapsed_ms_mean": sum(elapsed_ms) / len(elapsed_ms),
        "elapsed_ms_min": min(elapsed_ms),
        "elapsed_ms_max": max(elapsed_ms),
        "peak_memory_mb": peak_memories,
        "peak_memory_mb_mean": sum(peak_memories) / len(peak_memories),
        "result": last_result,
    }


def profile_scenario(
    module,
    model,
    tokenizer,
    ctx_inputs,
    gold: str,
    wrong: str,
    device: str,
):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        with_flops=True,
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        _ = run_ctx_pipeline(module, model, tokenizer, ctx_inputs, gold, wrong, device)
        sync_if_needed(device)

    total_flops = 0
    cpu_time_us = 0
    cuda_time_us = 0
    cpu_mem = 0
    cuda_mem = 0
    event_count = 0
    top_ops = []

    for evt in prof.key_averages():
        event_count += 1
        evt_flops = getattr(evt, "flops", 0) or 0
        total_flops += int(evt_flops)
        cpu_time_us += int(getattr(evt, "self_cpu_time_total", 0) or 0)
        cuda_time_us += int(getattr(evt, "self_cuda_time_total", 0) or 0)
        cpu_mem += int(getattr(evt, "self_cpu_memory_usage", 0) or 0)
        cuda_mem += int(getattr(evt, "self_cuda_memory_usage", 0) or 0)

    for evt in sorted(prof.key_averages(), key=lambda x: getattr(x, "flops", 0) or 0, reverse=True)[:10]:
        top_ops.append(
            {
                "key": evt.key,
                "flops": int(getattr(evt, "flops", 0) or 0),
                "self_cpu_time_total_us": int(getattr(evt, "self_cpu_time_total", 0) or 0),
                "self_cuda_time_total_us": int(getattr(evt, "self_cuda_time_total", 0) or 0),
            }
        )

    return {
        "reported_total_flops": total_flops,
        "reported_total_flops_giga": total_flops / 1e9,
        "self_cpu_time_total_ms": cpu_time_us / 1000.0,
        "self_cuda_time_total_ms": cuda_time_us / 1000.0,
        "self_cpu_memory_usage_mb": cpu_mem / (1024 ** 2),
        "self_cuda_memory_usage_mb": cuda_mem / (1024 ** 2),
        "event_count": event_count,
        "top_flop_ops": top_ops,
    }


def write_markdown(out_path: Path, payload: dict):
    normal = payload["normal_ctx"]
    ablated = payload["ablated_ctx"]
    delta_time = ablated["timing"]["elapsed_ms_mean"] - normal["timing"]["elapsed_ms_mean"]
    delta_pct = 0.0
    if normal["timing"]["elapsed_ms_mean"] != 0:
        delta_pct = delta_time / normal["timing"]["elapsed_ms_mean"] * 100.0

    lines = [
        "# Hulu-Med computational cost benchmark",
        "",
        "## Setup",
        "",
        f"- Model: {payload['config']['model']}",
        f"- Data CSV: {payload['config']['data_csv']}",
        f"- Selected heads: {payload['config']['selected_heads_json']}",
        f"- Sample row id: {payload['sample']['row_idx']}",
        f"- Question: {payload['sample']['question']}",
        f"- Gold / wrong: {payload['sample']['gold']} / {payload['sample']['wrong']}",
        f"- Image path: {payload['sample']['img_path']}",
        f"- Image size after resize: {payload['sample_inputs']['image_size']}",
        f"- Warmup / repeats: {payload['config']['warmup']} / {payload['config']['repeats']}",
        "",
        "## Timing",
        "",
        f"- Normal ctx mean latency: {normal['timing']['elapsed_ms_mean']:.3f} ms",
        f"- Ablated ctx mean latency: {ablated['timing']['elapsed_ms_mean']:.3f} ms",
        f"- Latency delta: {delta_time:.3f} ms ({delta_pct:.2f}%)",
        f"- Normal peak memory mean: {normal['timing']['peak_memory_mb_mean']:.3f} MB",
        f"- Ablated peak memory mean: {ablated['timing']['peak_memory_mb_mean']:.3f} MB",
        "",
        "## FLOPs",
        "",
        f"- Normal reported FLOPs: {normal['profile']['reported_total_flops_giga']:.6f} GFLOPs",
        f"- Ablated reported FLOPs: {ablated['profile']['reported_total_flops_giga']:.6f} GFLOPs",
        f"- FLOPs delta: {ablated['profile']['reported_total_flops_giga'] - normal['profile']['reported_total_flops_giga']:.6f} GFLOPs",
        "- Note: this ablation is implemented with attention-mask hooks, so theoretical matmul FLOPs are expected to stay essentially unchanged; any runtime gap mainly reflects hook and mask overhead.",
        "",
        "## Outputs",
        "",
        f"- Normal pred: {normal['timing']['result']['pred']}",
        f"- Ablated pred: {ablated['timing']['result']['pred']}",
        f"- Normal follow_conflict: {normal['timing']['result']['metrics']['follow_conflict']:.6f}",
        f"- Ablated follow_conflict: {ablated['timing']['result']['metrics']['follow_conflict']:.6f}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hulu-Med normal vs selected-head ablated inference cost.")
    parser.add_argument("--ablate_head_path", default=str(ABALTE_HEAD_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data_csv", default=DEFAULT_DATA_CSV)
    parser.add_argument("--image_root", default=".")
    parser.add_argument("--selected_heads", default=DEFAULT_SELECTED_HEADS)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--position", default="before_question")
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_image_side", type=int, default=672)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output_stem", default="hulumed_before_question_ablation_cost")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    module = load_ablate_module(args.ablate_head_path)
    module.set_seed(0)
    dtype = module.str2dtype(args.dtype)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    processor = module.load_processor_with_compat(args.model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = load_model(module, args.model, device, dtype)

    sample = select_sample(module, args.data_csv, args.image_root, args.sample_index)
    sample_inputs = build_sample_inputs(
        module=module,
        processor=processor,
        sample=sample,
        position=args.position,
        device=device,
        max_image_side=args.max_image_side,
    )

    selected_pairs = module.parse_selected_heads(args.selected_heads)
    layer_to_heads = module.pack_layer_to_heads(selected_pairs)

    normal_timing = benchmark_scenario(
        module=module,
        model=model,
        tokenizer=tokenizer,
        ctx_inputs=sample_inputs["ctx_inputs"],
        gold=sample["gold"],
        wrong=sample["wrong"],
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    normal_profile = profile_scenario(
        module=module,
        model=model,
        tokenizer=tokenizer,
        ctx_inputs=sample_inputs["ctx_inputs"],
        gold=sample["gold"],
        wrong=sample["wrong"],
        device=device,
    )

    handles = module.install_head_mask_hooks(model, layer_to_heads, keep_mode="self")
    try:
        ablated_timing = benchmark_scenario(
            module=module,
            model=model,
            tokenizer=tokenizer,
            ctx_inputs=sample_inputs["ctx_inputs"],
            gold=sample["gold"],
            wrong=sample["wrong"],
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        ablated_profile = profile_scenario(
            module=module,
            model=model,
            tokenizer=tokenizer,
            ctx_inputs=sample_inputs["ctx_inputs"],
            gold=sample["gold"],
            wrong=sample["wrong"],
            device=device,
        )
    finally:
        module.remove_handles(handles)

    payload = {
        "config": {
            "ablate_head_path": args.ablate_head_path,
            "model": args.model,
            "data_csv": args.data_csv,
            "image_root": args.image_root,
            "selected_heads_json": args.selected_heads,
            "sample_index": args.sample_index,
            "position": args.position,
            "dtype": args.dtype,
            "device": device,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_image_side": args.max_image_side,
        },
        "sample": sample,
        "sample_inputs": {
            "image_size": sample_inputs["image_size"],
            "ctx_prompt_token_count": int(sample_inputs["ctx_inputs"]["input_ids"].shape[-1]),
            "nc_prompt_token_count": int(sample_inputs["nc_inputs"]["input_ids"].shape[-1]),
        },
        "selected_heads": [{"layer": l, "head": h} for l, h in selected_pairs],
        "normal_ctx": {
            "timing": normal_timing,
            "profile": normal_profile,
        },
        "ablated_ctx": {
            "timing": ablated_timing,
            "profile": ablated_profile,
        },
    }

    json_path = out_dir / f"{args.output_stem}.json"
    md_path = out_dir / f"{args.output_stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)

    print(json.dumps(
        {
            "json": str(json_path),
            "markdown": str(md_path),
            "normal_mean_ms": normal_timing["elapsed_ms_mean"],
            "ablated_mean_ms": ablated_timing["elapsed_ms_mean"],
            "normal_reported_gflops": normal_profile["reported_total_flops_giga"],
            "ablated_reported_gflops": ablated_profile["reported_total_flops_giga"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
