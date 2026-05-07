import argparse
from pathlib import Path

import torch


FEATURE_GROUPS = {
    "chdp_minimal": [
        "d_h",
        "d_delta",
        "A_img_ic",
        "A_ctx_ic",
        "A_ctx_minus_img_ic",
        "M_gc_nc",
        "M_gc_ic",
    ],
    "chdp_plus_gap": [
        "d_h",
        "d_delta",
        "A_img_ic",
        "A_ctx_ic",
        "A_ctx_minus_img_ic",
        "M_gc_nc",
        "M_gc_ic",
        "M_gc_gap",
    ],
    "attention_only": [
        "A_img_ic",
        "A_ctx_ic",
        "A_ctx_minus_img_ic",
    ],
    "score_only": [
        "M_gc_nc",
        "M_gc_ic",
    ],
    "divergence_only": [
        "d_h",
        "d_delta",
    ],
    "no_divergence": [
        "A_img_ic",
        "A_ctx_ic",
        "A_ctx_minus_img_ic",
        "M_gc_nc",
        "M_gc_ic",
    ],
    "no_attention": [
        "d_h",
        "d_delta",
        "M_gc_nc",
        "M_gc_ic",
    ],
    "no_score": [
        "d_h",
        "d_delta",
        "A_img_ic",
        "A_ctx_ic",
        "A_ctx_minus_img_ic",
    ],
}


def compute_d_delta(nc_hidden, ic_hidden):
    nc_delta = torch.zeros_like(nc_hidden)
    ic_delta = torch.zeros_like(ic_hidden)
    nc_delta[:, 0, :] = nc_hidden[:, 0, :]
    ic_delta[:, 0, :] = ic_hidden[:, 0, :]
    nc_delta[:, 1:, :] = nc_hidden[:, 1:, :] - nc_hidden[:, :-1, :]
    ic_delta[:, 1:, :] = ic_hidden[:, 1:, :] - ic_hidden[:, :-1, :]
    return torch.linalg.norm(nc_delta - ic_delta, dim=-1)


def main():
    ap = argparse.ArgumentParser(description="Build layer-wise CHDP feature tensors from atomic dumps.")
    ap.add_argument("--atomic_bundle", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    atomic = torch.load(args.atomic_bundle, map_location="cpu")
    nc_hidden = atomic["nc_hidden_states"].float()
    ic_hidden = atomic["ic_hidden_states"].float()

    features = {
        "d_h": torch.linalg.norm(nc_hidden - ic_hidden, dim=-1),
        "d_delta": compute_d_delta(nc_hidden, ic_hidden),
        "A_img_ic": atomic["ic_attn_img"].float(),
        "A_ctx_ic": atomic["ic_attn_ctx"].float(),
        "A_ctx_minus_img_ic": atomic["ic_attn_ctx"].float() - atomic["ic_attn_img"].float(),
        "M_gc_nc": atomic["nc_score_gold"].float() - atomic["nc_score_conflict"].float(),
        "M_gc_ic": atomic["ic_score_gold"].float() - atomic["ic_score_conflict"].float(),
    }
    features["M_gc_gap"] = features["M_gc_nc"] - features["M_gc_ic"]

    feature_names = list(features.keys())
    feature_tensor = torch.stack([features[name] for name in feature_names], dim=-1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "source_atomic_bundle": str(args.atomic_bundle),
            "feature_names": feature_names,
            "feature_groups": FEATURE_GROUPS,
            "features": feature_tensor,
            "labels": atomic["labels"].long(),
            "label_names": atomic["label_names"],
            "sample_ids": atomic["sample_ids"],
            "sample_keys": atomic["sample_keys"],
            "splits": atomic["splits"],
            "questions": atomic["questions"],
            "img_ids": atomic["img_ids"],
            "gold_answers": atomic["gold_answers"],
            "conflict_answers": atomic["conflict_answers"],
            "n_layers": int(feature_tensor.shape[1]),
            "feature_dim": int(feature_tensor.shape[2]),
        },
        out_path,
    )

    summary_path = out_path.with_suffix(".summary.json")
    label_counts = {"resist": 0, "hijack": 0}
    for name in atomic["label_names"]:
        label_counts[str(name)] += 1
    summary_path.write_text(
        __import__("json").dumps(
            {
                "source_atomic_bundle": str(args.atomic_bundle),
                "out_path": str(out_path),
                "n_samples": int(feature_tensor.shape[0]),
                "n_layers": int(feature_tensor.shape[1]),
                "feature_dim": int(feature_tensor.shape[2]),
                "feature_names": feature_names,
                "feature_groups": FEATURE_GROUPS,
                "label_counts": label_counts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved_dataset={out_path}")
    print(f"saved_summary={summary_path}")
    print(f"n_samples={int(feature_tensor.shape[0])}")
    print(f"n_layers={int(feature_tensor.shape[1])}")
    print(f"feature_dim={int(feature_tensor.shape[2])}")


if __name__ == "__main__":
    main()
