"""Build the 2-class groundnut-vs-weed YOLO detection dataset.

Data semantics (confirmed by the researcher on 2026-09-17):
  - ``agri_data``            : BOTH YOLO classes (class_0, class_1) are WEEDS
                               (other crops / weeds / non-groundnut plants).
                               Every box is mapped to class ``weed``.
  - ``Peanut-dataset-main``  : RGB segmentation masks, green = crop (groundnut),
                               red = weed. Crop components -> class ``groundnut``,
                               weed components -> class ``weed``.
  - ``Arachis hypogaea - Optimized Images`` : ALL images are groundnut crop/plant.
                               No object-level annotation exists, so each image
                               receives ONE whole-frame box (class ``groundnut``,
                               inset 4% from every edge) as documented weak
                               supervision.

Class ids in the output dataset:
    0: groundnut
    1: weed

Output layout (Ultralytics/YOLO):
    datasets/processed/groundnut_weed_detection/
        images/{train,val,test}/<stem>.jpg
        labels/{train,val,test}/<stem>.txt      (class cx cy w h, normalized)
        data.yaml                                 (names 0: groundnut, 1: weed)

Usage (from the project root):
    python ml/data/build_detection_dataset.py [--seed 42] [--force] [--target-size WxH]
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "ml" / "data"))

from src.image_utils import copy_image, parse_size  # noqa: E402

RAW = (
    PROJECT
    / "Documents/PhD Dataset for learning ml models/DataSet/Secondary"
)
AGRI_IMAGES = RAW / "agri_data/data"
PEANUT_IMAGES = RAW / "Peanut-dataset-main/images"
PEANUT_MASKS = RAW / "Peanut-dataset-main/labels"
ARACHIS_IMAGES = RAW / "Arachis hypogaea - Optimized Images"

OUT = PROJECT / "datasets/processed/groundnut_weed_detection"

GROUNDNUT_ID = 0
WEED_ID = 1

# Whole-frame box inset for the weakly-labelled Arachis (groundnut) images.
ARACHIS_BOX = (0.5, 0.5, 0.92, 0.92)  # cx, cy, w, h (normalized)

SPLIT_NAMES = ("train", "val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)

MIN_COMPONENT_AREA = 400  # px^2 — drop specks/noise from mask components
MIN_BOX_SIDE = 12  # px — drop tiny boxes


def _classify_mask(mask_rgb: np.ndarray):
    """Return boolean masks for crop (groundnut) and weed from an RGB mask."""
    r = mask_rgb[..., 0].astype(int)
    g = mask_rgb[..., 1].astype(int)
    b = mask_rgb[..., 2].astype(int)
    crop = (g > 90) & (g > r + 30) & (g > b + 30)
    weed = (r > 90) & (r > g + 30) & (r > b + 30)
    return crop, weed


def _components_to_boxes(mask: np.ndarray, img_w: int, img_h: int):
    """Convert a boolean mask to normalized YOLO boxes (one per component)."""
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):  # 0 is background
        x, y, w, h, area = stats[i]
        if area < MIN_COMPONENT_AREA or w < MIN_BOX_SIDE or h < MIN_BOX_SIDE:
            continue
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        nw = w / img_w
        nh = h / img_h
        boxes.append((cx, cy, nw, nh))
    return boxes


def _rewrite_agri_label(src: Path, dst: Path) -> int:
    """Rewrite an agri_data YOLO label so every box becomes class 'weed'."""
    lines = []
    for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            x, y, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        lines.append(f"{WEED_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    ap.add_argument("--target-size", default=None,
                    help="resize processed copies to WxH (e.g. 640x640)")
    args = ap.parse_args()

    target = parse_size(args.target_size) if args.target_size else None

    if OUT.exists() and not args.force and not target:
        print(f"{OUT} already exists (use --force to rebuild).")
        return 0

    # Clean rebuild: wipe the output tree BEFORE staging temporary labels so
    # no stale file from a previous build survives.
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "_tmp").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # ---- Stage 1: collect images -------------------------------------------
    # stem -> (img_src, kind) where kind is "agri", "peanut" or "arachis"
    images: dict = {}

    # 1a) agri_data -> all weeds
    n_agri = 0
    n_agri_boxes = 0
    for lbl in sorted(AGRI_IMAGES.glob("*.txt")):
        img = AGRI_IMAGES / f"{lbl.stem}.jpeg"
        if not img.exists():
            continue
        images[lbl.stem] = (img, "agri")
        n_agri_boxes += _rewrite_agri_label(lbl, OUT / "_tmp" / f"{lbl.stem}.txt")
        n_agri += 1

    # 1b) Peanut -> groundnut + weed boxes from segmentation masks
    n_peanut = 0
    for img_path in sorted(PEANUT_IMAGES.glob("*.jpg")):
        msk_path = PEANUT_MASKS / f"{img_path.stem}.png"
        if not msk_path.exists():
            continue
        images[img_path.stem] = (img_path, "peanut")
        n_peanut += 1

    # 1c) Arachis hypogaea -> ALL groundnut, one whole-frame weak box each
    n_arachis = 0
    for img_path in sorted(ARACHIS_IMAGES.glob("*.jpg")):
        images[img_path.stem] = (img_path, "arachis")
        n_arachis += 1

    print(f"agri_data images: {n_agri} (boxes {n_agri_boxes}), "
          f"peanut images: {n_peanut}, arachis images: {n_arachis}")

    # ---- Stage 2: split by stem (random, seed) ------------------------------
    stems = sorted(images.keys())
    rng.shuffle(stems)
    n = len(stems)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    split = {}
    split["train"] = set(stems[:n_train])
    split["val"] = set(stems[n_train:n_train + n_val])
    split["test"] = set(stems[n_train + n_val:])
    print("split sizes:", {k: len(v) for k, v in split.items()})

    # ---- Stage 3: materialize ----------------------------------------------
    tmp_label = OUT / "_tmp"
    for sname in SPLIT_NAMES:
        (OUT / "images" / sname).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / sname).mkdir(parents=True, exist_ok=True)

    stats = {"groundnut": 0, "weed": 0, "images_with_gnd": 0,
             "arachis_images": 0}
    for stem in stems:
        img_src, kind = images[stem]
        sname = next(s for s in SPLIT_NAMES if stem in split[s])
        ext = img_src.suffix
        dst_img = OUT / "images" / sname / f"{stem}{ext}"
        copy_image(img_src, dst_img, target)

        if kind == "agri":
            src_lbl = tmp_label / f"{stem}.txt"
            dst_lbl = OUT / "labels" / sname / f"{stem}.txt"
            shutil.copy2(src_lbl, dst_lbl)
            nb = len(src_lbl.read_text(encoding="utf-8").splitlines())
            stats["weed"] += nb
        elif kind == "arachis":
            cx, cy, nw, nh = ARACHIS_BOX
            dst_lbl = OUT / "labels" / sname / f"{stem}.txt"
            dst_lbl.write_text(
                f"{GROUNDNUT_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n",
                encoding="utf-8")
            stats["groundnut"] += 1
            stats["images_with_gnd"] += 1
            stats["arachis_images"] += 1
        else:
            img = np.asarray(Image.open(img_src).convert("RGB"))
            h, w = img.shape[:2]
            mask = np.asarray(Image.open(PEANUT_MASKS / f"{stem}.png").convert("RGB"))
            crop, weed = _classify_mask(mask)
            lines = []
            for cx, cy, nw, nh in _components_to_boxes(crop, w, h):
                lines.append(f"{GROUNDNUT_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats["groundnut"] += 1
            for cx, cy, nw, nh in _components_to_boxes(weed, w, h):
                lines.append(f"{WEED_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                stats["weed"] += 1
            if any(l.startswith(f"{GROUNDNUT_ID} ") for l in lines):
                stats["images_with_gnd"] += 1
            dst_lbl = OUT / "labels" / sname / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""),
                               encoding="utf-8")

    # ---- Stage 4: data.yaml -------------------------------------------------
    yaml_lines = [
        "# Auto-generated by ml/data/build_detection_dataset.py",
        "# Detector semantics CONFIRMED on 2026-09-17 by the researcher.",
        "#   0 = groundnut (Arachis hypogaea whole-frame weak boxes + Peanut crop masks)",
        "#   1 = weed      (agri_data boxes + Peanut weed masks)",
        "path: datasets/processed/groundnut_weed_detection",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: groundnut",
        "  1: weed",
    ]
    (OUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    shutil.rmtree(tmp_label, ignore_errors=True)

    print("class box counts:", stats)
    print(f"dataset written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())