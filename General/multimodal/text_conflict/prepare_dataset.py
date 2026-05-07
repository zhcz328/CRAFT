from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from dataset_adapters import prepare_gqa_closed_with_wrong, prepare_vqav2_closed_with_wrong


DEFAULT_DATASET_ROOTS = {
    "gqa": Path("../../data/GQA"),
    "vqav2_validation": Path("../../data/VQAv2_validation"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare closed-only multimodal CSV with wrong answers.")
    ap.add_argument("--dataset_name", required=True, choices=["gqa", "vqav2_validation"])
    ap.add_argument("--dataset_root", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--image_dir_name", default="images")
    ap.add_argument("--disable_gqa_scene_graph_synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_name = args.dataset_name.lower()
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = (THIS_DIR / DEFAULT_DATASET_ROOTS[dataset_name]).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        stem = "gqa_closed_with_wrong.csv" if dataset_name == "gqa" else "vqav2_validation_closed_with_wrong.csv"
        out_csv = dataset_root / stem

    # Requirement: if GQA is already prepared in image_conflict, text_conflict can reuse it.
    if dataset_name == "gqa" and out_csv.exists() and not args.force:
        print(f"dataset={dataset_name}")
        print(f"dataset_root={dataset_root}")
        print(f"out_csv={out_csv}")
        print("reused_existing=true")
        return

    if dataset_name == "gqa":
        summary = prepare_gqa_closed_with_wrong(
            dataset_root=dataset_root,
            out_csv=out_csv,
            image_dir_name=args.image_dir_name,
            allow_scene_graph_synthetic=not args.disable_gqa_scene_graph_synthetic,
        )
    else:
        summary = prepare_vqav2_closed_with_wrong(
            dataset_root=dataset_root,
            out_csv=out_csv,
            image_dir_name=args.image_dir_name,
        )

    print(f"dataset={summary['dataset_name']}")
    print(f"dataset_root={summary['dataset_root']}")
    print(f"out_csv={summary['out_csv']}")
    print(f"n_closed_rows={summary['n_closed_rows']}")


if __name__ == "__main__":
    main()
