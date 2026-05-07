from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from dataset_adapters import prepare_gqa_closed_with_wrong


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare GQA closed-only CSV with wrong answers for image-conflict pipeline.")
    ap.add_argument("--dataset_root", default=str((THIS_DIR / "../../data/GQA").resolve()))
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--image_dir_name", default="images")
    ap.add_argument("--disable_scene_graph_synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    out_csv = Path(args.out_csv) if args.out_csv else dataset_root / "gqa_closed_with_wrong.csv"
    if out_csv.exists() and not args.force:
        print("dataset=gqa")
        print(f"dataset_root={dataset_root}")
        print(f"out_csv={out_csv}")
        print("reused_existing=true")
        return

    summary = prepare_gqa_closed_with_wrong(
        dataset_root=dataset_root,
        out_csv=out_csv,
        image_dir_name=args.image_dir_name,
        allow_scene_graph_synthetic=not args.disable_scene_graph_synthetic,
    )

    print(f"dataset={summary['dataset_name']}")
    print(f"dataset_root={summary['dataset_root']}")
    print(f"out_csv={summary['out_csv']}")
    print(f"n_closed_rows={summary['n_closed_rows']}")


if __name__ == "__main__":
    main()
