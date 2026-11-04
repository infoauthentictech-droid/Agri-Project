"""Visualize augmentation before/after examples.

Modes:
- single image: python -m ml.augmentation.visualize --config configs/field_condition.json --image <img> [--n 5 --seed 42 --outdir ...]
- example set:  python -m ml.augmentation.visualize --examples [--source-dir <dir>] [--n 5 --seed 42]

All randomness is seeded: panel k of a run uses seed = <base seed> + k, so the
exact same before/after grid can be reproduced on any machine.
Raw data is only read; outputs go to the --outdir (default
datasets/processed/augmentation_examples).
"""
from __future__ import annotations

import argparse
import json
import sys
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import AugmentationPipeline, load_config  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

CONFIGS_DIR = ROOT / "configs"
DEFAULT_EXAMPLES = ["none", "standard", "field_condition"]
DEFAULT_SOURCE = (
    Path(r"D:/Flutter/PHD PROJECT/Agri Project/Documents/PhD Dataset for learning ml models")
    / "DataSet/Secondary/Peanut-dataset-main"
)


def _crop_canvas(panels, labels, cols=3, pad=6, caption_h=18):
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    rows = ceil(len(panels) / cols)
    canvas = Image.new(
        "RGB",
        (cols * w + (cols + 1) * pad, rows * (h + caption_h) + (rows + 1) * pad),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + caption_h + pad)
        canvas.paste(panel.resize((w, h)), (x, y))
        draw.text((x + 2, y + h + 3), str(label), fill=(20, 20, 20))
    return canvas


def build_grid(config, image, n, base_seed):
    """Return (grid_pil, metadata) with original + n seeded augmentations."""
    import numpy as np

    img = np.asarray(image)
    pipeline = AugmentationPipeline(config, seed=base_seed)
    panels = [Image.fromarray(img)]
    labels = ["original"]
    seeds = [base_seed]
    for k in range(1, n + 1):
        pipeline.set_seed(base_seed + k)
        aug = pipeline.apply(img)
        panels.append(Image.fromarray(aug))
        labels.append(f"seed={base_seed + k}")
        seeds.append(base_seed + k)
    return _crop_canvas(panels, labels), {
        "config": pipeline.name,
        "base_seed": base_seed,
        "panel_seeds": seeds,
        "transforms": pipeline.describe(),
    }


def run_single(args) -> None:
    config = load_config(args.config)
    image = Image.open(args.image).convert("RGB")
    grid, meta = build_grid(config, image, args.n, args.seed)
    outdir = Path(args.outdir) / config["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    dst = outdir / f"{stem}.png"
    grid.save(dst)
    meta["image"] = str(args.image)
    meta["output"] = str(dst)
    (outdir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{config['name']}] saved {dst} ({len(meta['panel_seeds'])} panels)")
    print(f"  pipeline: {meta['transforms']}")



def run_examples(args) -> None:
    configs = [load_config(CONFIGS_DIR / f"{name}.json") for name in DEFAULT_EXAMPLES]
    src = Path(args.source_dir) if args.source_dir else DEFAULT_SOURCE / "images"
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    images = images[:3]
    if not images:
        raise SystemExit(f"no images found in {src}")
    masks_dir = DEFAULT_SOURCE / "labels" if args.source_dir is None else None

    outroot = Path(args.outdir)
    import numpy as np

    for i, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        for config in configs:
            grid, meta = build_grid(config, image, args.n, args.seed + i * 100)
            outdir = outroot / config["name"]
            outdir.mkdir(parents=True, exist_ok=True)
            dst = outdir / f"{img_path.stem}.png"
            grid.save(dst)
            meta["image"] = str(img_path)
            meta["output"] = str(dst)
            (outdir / f"{img_path.stem}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(f"[{config['name']}] {dst}")

        # Mask-synchronization demo (peanut image + RGB mask)
        if masks_dir is not None:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                mask = Image.open(mask_path).convert("RGB")
                pipeline = AugmentationPipeline(configs[2], seed=args.seed + i * 100 + 7)
                aug_img, aug_mask = pipeline.apply_with_mask(
                    np.asarray(image), np.asarray(mask)
                )
                demo = _crop_canvas(
                    [image, Image.fromarray(aug_img), mask, Image.fromarray(aug_mask)],
                    ["original", "field_condition", "original mask", "synced mask"],
                    cols=2,
                )
                ddir = outroot / "mask_sync"
                ddir.mkdir(parents=True, exist_ok=True)
                demo.save(ddir / f"{img_path.stem}.png")
                print(f"[mask_sync] {ddir / f'{img_path.stem}.png'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Groundnut AI augmentation visualization")
    ap.add_argument("--config", default=None, help="path to one config JSON")
    ap.add_argument("--image", default=None, help="path to a single source image")
    ap.add_argument("--examples", action="store_true",
                    help="generate before/after examples for all configs")
    ap.add_argument("--source-dir", default=None, help="override example source folder")
    ap.add_argument("--n", type=int, default=5, help="number of augmented panels")
    ap.add_argument("--seed", type=int, default=42, help="base random seed")
    ap.add_argument("--outdir", default="datasets/processed/augmentation_examples")
    args = ap.parse_args(argv)

    if args.examples:
        run_examples(args)
    elif args.config and args.image:
        run_single(args)
    else:
        ap.error("provide --config + --image, or --examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())