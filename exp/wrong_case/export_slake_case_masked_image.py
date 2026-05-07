from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


REPO_ROOT = Path("/root/logit_lens")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Slake_vqa.image_conflict.mask_utils import build_masked_image, scale_binary_mask  # noqa: E402


DEFAULT_IMAGE_PATH = Path("/root/autodl-tmp/data/SLAKE/imgs/xmlab208/source.jpg")
DEFAULT_DETECTION_PATH = Path("/root/autodl-tmp/data/SLAKE/imgs/xmlab208/detection.json")
DEFAULT_MASK_PATH = Path("/root/autodl-tmp/data/SLAKE/imgs/xmlab208/mask.png")
DEFAULT_TARGET_LABELS = ["Liver"]
DEFAULT_OUTPUT_DIR = Path("/root/logit_lens/exp/wrong_case/slake_two_choice_case_assets")


def make_overlay(source: Image.Image, mask: Image.Image, color=(255, 64, 64), alpha: int = 110) -> Image.Image:
    source = source.convert("RGBA")
    mask = mask.convert("L")
    overlay = Image.new("RGBA", source.size, (*color, 0))
    alpha_mask = mask.point(lambda px: alpha if px > 0 else 0)
    overlay.putalpha(alpha_mask)
    return Image.alpha_composite(source, overlay).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export the SLAKE wrong-case masked image using the repository's mask_scale behavior."
    )
    ap.add_argument("--image_path", type=Path, default=DEFAULT_IMAGE_PATH)
    ap.add_argument("--detection_path", type=Path, default=DEFAULT_DETECTION_PATH)
    ap.add_argument("--mask_path", type=Path, default=DEFAULT_MASK_PATH)
    ap.add_argument("--target_labels", nargs="+", default=DEFAULT_TARGET_LABELS)
    ap.add_argument("--mask_scale", type=float, default=2.0)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    if args.mask_scale <= 0:
        raise SystemExit("--mask_scale must be positive")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image_path).convert("RGB")
    source.save(out_dir / "source.png")

    masked, meta = build_masked_image(
        source_path=args.image_path,
        detection_path=args.detection_path,
        target_labels=args.target_labels,
        mask_path=args.mask_path,
        mask_scale=float(args.mask_scale),
    )
    masked.save(out_dir / "masked_scale2.png")

    raw_mask = Image.open(args.mask_path).convert("L")
    if raw_mask.size != source.size:
        raw_mask = raw_mask.resize(source.size, resample=Image.Resampling.NEAREST)
    scaled_mask = scale_binary_mask(raw_mask, float(args.mask_scale))
    scaled_mask.save(out_dir / "scaled_mask_scale2.png")

    overlay = make_overlay(source, scaled_mask)
    overlay.save(out_dir / "overlay_scale2.png")

    payload = {
        "image_path": str(args.image_path),
        "detection_path": str(args.detection_path),
        "mask_path": str(args.mask_path),
        "target_labels": list(args.target_labels),
        "mask_scale": float(args.mask_scale),
        "outputs": {
            "source": str(out_dir / "source.png"),
            "masked": str(out_dir / "masked_scale2.png"),
            "scaled_mask": str(out_dir / "scaled_mask_scale2.png"),
            "overlay": str(out_dir / "overlay_scale2.png"),
        },
        "mask_meta": meta,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
