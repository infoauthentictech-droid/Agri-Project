"""Build the primary / secondary / hybrid YOLO detection and segmentation datasets.

Reads the raw folders under ``datasets/`` (existing files are never changed; the
only addition is an empty ``labels`` folder beside the mixed images) and writes:

    datasets/{primary,secondary,hybrid}/{detection,segmentation}/
        images/{train,val,test}/<source>_<stem>.<ext>   (hard links to the originals)
        labels/{train,val,test}/<source>_<stem>.txt
        data.yaml
        metadata.json

Sources and how trustworthy each label is (recorded per image in the manifest):
  - agri      : public 1,300-image set. Both of its classes map to ``weed``
                (researcher decision, configs/detector_labels.yaml). Boxes: as published.
  - peanut    : public 400-image set. Polygons/boxes converted from the published RGB
                masks only; nearby parts of one class count as one plant.
  - closeup   : own groundnut close-ups. Boxes and polygons: pretrained SAM 2.1 prompted
                by the ExG vegetation mask, keeping the ExG mask when the two disagree.
  - weed      : own weed close-ups (DSC* files). Same method.
  - mixed     : own mixed field images labelled with ml/data/label_tool.py. Always the
                test split.
  - field     : own wide field images labelled with the same tool.
  - field_machine : other own field images labelled by ml/data/pseudo_label.py from the
                researcher's examples. Train split only.

Splits are made per source-photo group so near-duplicates never cross a split:
  - agri, peanut    : ORB + RANSAC matching, including flipped copies (agri is
                      augmented from about 500 photos).
  - closeup, weed   : blocks of consecutive DSC numbers (burst shots).

Usage (from the project root):
    python ml/data/build_yolo_datasets.py [--seed 42] [--weed-cap 1500] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT =Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "ml" / "data"))

from gpu_loader import decode_file  # noqa: E402
from build_detection_dataset import (  # noqa: E402
    GROUNDNUT_ID,
    MIN_BOX_SIDE,
    MIN_COMPONENT_AREA,
    SPLIT_NAMES,
    SPLIT_RATIOS,
    WEED_ID,
    _classify_mask,
    _rewrite_agri_label,
)

DATASETS = PROJECT / "datasets"
OWN = DATASETS / "my_own_dataset"
PUBLIC = DATASETS / "public_dataset"

AGRI_DIR = PUBLIC / "crop_weed_mixed_public_dataset/data"
PEANUT_IMAGES = PUBLIC / "crop_weed_labled_or_masked_images/images"
PEANUT_MASKS = PUBLIC / "crop_weed_labled_or_masked_images/labels"
CLOSEUP_IMAGES = OWN / "crop/groundnut_closeup_images"
STAGE_FOLDERS = {"stage_vegetative": "crop 1", "stage_seedling": "crop 2", "stage_flowering": "crop 3"}
WEED_DIR = OWN / "weed"
MIXED_DIR = OWN / "Mixed dataset (crop and weed)"

MARKER = ".built_by_build_yolo_datasets"
MANIFEST = DATASETS / "yolo_build_manifest.csv"
REPORT = DATASETS / "yolo_build_report.json"

DETECTION_NAMES = {GROUNDNUT_ID: "groundnut", WEED_ID: "weed"}  # configs/detector_labels.yaml
SEGMENTATION_NAMES = {0: "crop", 1: "weed"}  # CANONICAL_PIPELINES.md

FIELD_DIR = OWN / "crop" / "groundnut_field_images"
SAM_WEIGHTS = PROJECT / "weights" / "sam2.1_b.pt"
SAM_BOX_PAD = 0.05  # of the frame, added around the colour box so pale leaf tips fit inside
SAM_MIN_COVER = 0.80  # SAM must contain this share of the colour mask...
SAM_MIN_AREA = 0.60  # ...and be between these multiples of its area, or the colour mask is kept
SAM_MAX_AREA = 1.60
PLANT_JOIN_KERNEL = 15  # px; parts of one class closer than this count as one plant
MIN_POLYGON_AREA = 100  # px^2 at the mask's own resolution
POLYGON_EPSILON = 1.0  # px, approxPolyDP tolerance
DSC_BLOCK = 25  # consecutive DSC numbers treated as one burst
ORB_CANDIDATES = 12
ORB_MIN_INLIERS = 25
ORB_BATCH = 128  # image pairs per GPU matrix multiply; about 1 GB of video memory
PLANT_CENTER_FRAC = 0.8  # the plant's main region must be centred in this part of the frame
PLANT_PARTS_FRAC = 0.9  # other parts of the plant must be centred in this part of the frame
PLANT_PART_MIN_FRACTION = 0.02  # of the main region's area
WEAK_MASK_MAX_COVER = 0.97  # a mask filling its box this fully means Otsu failed


@dataclass
class Item:
    name: str  # output stem, unique across all sources
    image: Path
    source: str
    group: str
    stratum: str
    quality: str
    detection: Optional[List[str]] = None
    segmentation: Optional[List[str]] = None
    split: str = ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    return torch.nn.functional.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)


def _erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    # Pixels beyond the frame count as foreground, so plants touching an edge keep their edge.
    return 1 - _dilate(1 - mask, kernel)


def close_mask(mask: np.ndarray, kernel: int, iterations: int) -> np.ndarray:
    """Morphological closing of a boolean mask on the GPU; returns uint8 0/1."""
    x = torch.from_numpy(mask.astype(np.float32)).to("cuda")[None, None]
    for _ in range(iterations):
        x = _dilate(x, kernel)
    for _ in range(iterations):
        x = _erode(x, kernel)
    return x[0, 0].to(torch.uint8).cpu().numpy()


def read_bgr_gpu(path: Path) -> torch.Tensor:
    """Decode an image on the GPU as HxWx3 BGR, the channel order the label steps use."""
    return decode_file(path)[0].permute(1, 2, 0).flip(2)


def vegetation_mask(img_bgr, close_kernel: int = 15, close_iterations: int = 2) -> np.ndarray:
    """Excess Green + Otsu vegetation mask on the GPU, cleaned so a plant's leaflets merge."""
    x = torch.as_tensor(img_bgr, device="cuda").float()
    exg = 2 * x[..., 1] - x[..., 2] - x[..., 0]
    low, high = exg.min(), exg.max()
    gray = ((exg - low) * 255 / (high - low).clamp(min=1e-6)).to(torch.uint8)
    hist = torch.bincount(gray.ravel().long(), minlength=256).double()
    weight = hist.cumsum(0)
    mean = (hist * torch.arange(256, device=hist.device)).cumsum(0)
    between = (mean[-1] * weight / weight[-1] - mean) ** 2 / (weight * (weight[-1] - weight)).clamp(min=1)
    veg = (gray > between.argmax()).float()[None, None]
    for _ in range(close_iterations):
        veg = _dilate(veg, close_kernel)
    for _ in range(close_iterations):
        veg = _erode(veg, close_kernel)
    veg = _dilate(_erode(veg, 5), 5)
    return (veg[0, 0] * 255).to(torch.uint8).cpu().numpy()


def to_yolo_line(bbox: Tuple[int, int, int, int], img_w: int, img_h: int, class_id: int) -> str:
    x0, y0, x1, y1 = bbox
    return (f"{class_id} {(x0 + x1) / 2 / img_w:.6f} {(y0 + y1) / 2 / img_h:.6f} "
            f"{(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}")


def drop_exact_duplicates(paths: Iterable[Path], seen: set) -> Tuple[List[Path], List[Path]]:
    kept, dropped = [], []
    for p in sorted(paths):
        digest = sha256(p)
        if digest in seen:
            dropped.append(p)
        else:
            seen.add(digest)
            kept.append(p)
    return kept, dropped


def mask_to_polygons(mask: np.ndarray, min_area: float = MIN_POLYGON_AREA,
                     epsilon: float = POLYGON_EPSILON) -> List[np.ndarray]:
    """Return one simplified outer contour per sufficiently large region."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polygons.append(approx)
    return polygons


def polygon_line(class_id: int, polygon: np.ndarray, width: int, height: int) -> str:
    coords = []
    for x, y in polygon:
        coords.append(f"{min(max(x / width, 0.0), 1.0):.6f}")
        coords.append(f"{min(max(y / height, 0.0), 1.0):.6f}")
    return f"{class_id} " + " ".join(coords)


def rasterize(lines: Iterable[str], class_id: int, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        parts = line.split()
        if int(parts[0]) != class_id:
            continue
        pts = np.array(parts[1:], dtype=np.float64).reshape(-1, 2) * (width, height)
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas.astype(bool)


def assign_groups(items: Sequence[Item], seed: int) -> None:
    """Split whole groups 70/15/15 inside each stratum.

    A group spanning two strata (a burst across two stage folders) stays whole,
    under the stratum of its first image.
    """
    home: Dict[str, str] = {}
    strata: Dict[str, Dict[str, List[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        strata[home.setdefault(item.group, item.stratum)][item.group].append(item)
    rng = random.Random(seed)
    for _, groups in sorted(strata.items()):
        keys = sorted(groups)
        rng.shuffle(keys)
        total = sum(len(v) for v in groups.values())
        counts = Counter()
        for key in keys:
            # Largest shortfall first, so big burst groups cannot starve the last split.
            split = max(zip(SPLIT_NAMES, SPLIT_RATIOS), key=lambda s: s[1] * total - counts[s[0]])[0]
            counts[split] += len(groups[key])
            for item in groups[key]:
                item.split = split


def ratio_matches(queries: Sequence[np.ndarray], trains: Sequence[np.ndarray], device: str) -> List[np.ndarray]:
    """Batched Hamming two-nearest search with Lowe's ratio test, one result per pair.

    Bits become +1/-1, so a dot product gives the Hamming distance and the whole
    batch runs as one matrix multiply. Returns (query_index, train_index) rows.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32

    def signed(batch: Sequence[np.ndarray], size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        packed = np.zeros((len(batch), size, 32), dtype=np.uint8)
        valid = np.zeros((len(batch), size), dtype=bool)
        for i, des in enumerate(batch):
            packed[i, :len(des)] = des
            valid[i, :len(des)] = True
        bits = torch.from_numpy(np.unpackbits(packed, axis=2)).to(device, dtype=dtype) * 2 - 1
        return bits, torch.from_numpy(valid).to(device)

    with torch.no_grad():
        a, valid_a = signed(queries, max(len(d) for d in queries))
        b, valid_b = signed(trains, max(len(d) for d in trains))
        distance = (256 - a @ b.transpose(1, 2)) / 2
        distance.masked_fill_(~valid_b[:, None, :], 1024)  # padding can never be a neighbour
        nearest, index = distance.topk(2, dim=2, largest=False)
        good = valid_a & (nearest[..., 0] < 0.75 * nearest[..., 1])
    good, index = good.cpu().numpy(), index[..., 0].cpu().numpy()
    return [np.stack((np.nonzero(g)[0], i[g]), axis=1) for g, i in zip(good, index)]


def source_photo_clusters(paths: Sequence[Path]) -> List[int]:
    """Cluster images sharing a source photo, allowing for flips and small shifts."""
    gray, hists, descriptors = [], [], []
    orb = cv2.ORB_create(1500)
    for path in paths:
        image = cv2.imread(str(path))
        scale = 512 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale)
        hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16], [0, 180, 0, 256]).ravel()
        hists.append(hist / np.linalg.norm(hist))
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray.append(g)
        descriptors.append(orb.detectAndCompute(g, None))
    similarity = np.array(hists) @ np.array(hists).T
    np.fill_diagonal(similarity, -1)
    pairs = {tuple(sorted((i, int(j)))) for i in range(len(paths)) for j in np.argsort(-similarity[i])[:ORB_CANDIDATES]}

    parent = list(range(len(paths)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    flipped: Dict[Tuple[int, int], tuple] = {}

    def features(index: int, flip: Optional[int]) -> tuple:
        if flip is None:
            return descriptors[index]
        key = (index, flip)
        if key not in flipped:
            flipped[key] = orb.detectAndCompute(cv2.flip(gray[index], flip), None)
        return flipped[key]

    def usable(des: Optional[np.ndarray]) -> bool:
        return des is not None and len(des) >= 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  matching {len(pairs)} candidate pairs on {device}", flush=True)
    for flip in (None, 1, 0, -1):
        remaining = [(a, b) for a, b in sorted(pairs) if find(a) != find(b)]
        for start in range(0, len(remaining), ORB_BATCH):
            chunk = [(a, b) for a, b in remaining[start:start + ORB_BATCH]
                     if usable(descriptors[a][1]) and usable(features(b, flip)[1])]
            if not chunk:
                continue
            matches = ratio_matches([descriptors[a][1] for a, _ in chunk],
                                    [features(b, flip)[1] for _, b in chunk], device)
            for (a, b), good in zip(chunk, matches):
                if len(good) < ORB_MIN_INLIERS:
                    continue
                kp_a, kp_b = descriptors[a][0], features(b, flip)[0]
                pts_a = np.float32([kp_a[q].pt for q, _ in good])
                pts_b = np.float32([kp_b[t].pt for _, t in good])
                _, inliers = cv2.estimateAffinePartial2D(pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=4)
                if inliers is not None and int(inliers.sum()) >= ORB_MIN_INLIERS:
                    parent[find(a)] = find(b)
        print(f"  flip {flip}: {len(remaining)} pairs checked, "
              f"{len({find(i) for i in range(len(paths))})} groups so far", flush=True)
    return [find(i) for i in range(len(paths))]


def dsc_group(stem: str) -> str:
    match = re.match(r"DSC(\d{5})", stem)
    return f"dsc{int(match.group(1)) // DSC_BLOCK}" if match else stem


def plant_mask(image: np.ndarray) -> Optional[np.ndarray]:
    """Vegetation mask of the centred plant in a single-plant close-up, or None.

    Starts from the largest central ExG region and adds the plant's other parts,
    which thin stems often disconnect. Used for both classes so box size cannot
    become a cue for the class.
    """
    height, width = image.shape[:2]
    veg = (vegetation_mask(image) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(veg, connectivity=8)

    def inside(index: int, frac: float) -> bool:
        cx, cy = centroids[index]
        margin_x, margin_y = width * (1 - frac) / 2, height * (1 - frac) / 2
        return margin_x <= cx <= width - margin_x and margin_y <= cy <= height - margin_y

    central = [i for i in range(1, count) if inside(i, PLANT_CENTER_FRAC)
               and stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA]
    if not central:
        return None
    main_area = max(stats[i, cv2.CC_STAT_AREA] for i in central)
    keep = [i for i in range(1, count) if inside(i, PLANT_PARTS_FRAC)
            and stats[i, cv2.CC_STAT_AREA] >= max(MIN_COMPONENT_AREA, PLANT_PART_MIN_FRACTION * main_area)]
    mask = np.isin(labels, keep)
    ys, xs = np.nonzero(mask)
    box_area = (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min())
    return None if mask.sum() / box_area > WEAK_MASK_MAX_COVER else mask


def sam_agrees(sam: np.ndarray, colour: np.ndarray) -> bool:
    """SAM sometimes outlines soil or the whole frame; colour catches that without being perfect."""
    area = max(1, int(colour.sum()))
    cover = (sam & colour).sum() / area
    return bool(cover >= SAM_MIN_COVER and SAM_MIN_AREA <= sam.sum() / area <= SAM_MAX_AREA)


class SamRefiner:
    """Redraw a plant mask with pretrained SAM 2.1, prompted by the colour mask's box."""

    def __init__(self, weights: Path = SAM_WEIGHTS) -> None:
        if not weights.exists():
            raise SystemExit(f"{weights} not found: it holds the pretrained SAM 2.1 model.")
        from ultralytics import SAM
        self.model = SAM(str(weights))
        self.used = False
        self.accepted = 0
        self.rejected = 0

    def refine(self, image: torch.Tensor, colour: np.ndarray) -> np.ndarray:
        height, width = colour.shape
        ys, xs = np.nonzero(colour)
        pad_x, pad_y = int(SAM_BOX_PAD * width), int(SAM_BOX_PAD * height)
        box = [max(0, int(xs.min()) - pad_x), max(0, int(ys.min()) - pad_y),
               min(width, int(xs.max()) + pad_x), min(height, int(ys.max()) + pad_y)]
        bgr = image.cpu().numpy() if isinstance(image, torch.Tensor) else image
        result = self.model(bgr, bboxes=[box], device=0, verbose=False)[0]
        sam = result.masks.data[0].cpu().numpy().astype(np.uint8) if result.masks is not None else None
        self.used = False
        if sam is not None:
            if sam.shape != colour.shape:
                sam = cv2.resize(sam, (width, height), interpolation=cv2.INTER_NEAREST)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(sam, connectivity=8)
            if count > 1:
                largest = stats[1:, cv2.CC_STAT_AREA].max()
                keep = [i for i in range(1, count)
                        if stats[i, cv2.CC_STAT_AREA] >= max(MIN_COMPONENT_AREA, PLANT_PART_MIN_FRACTION * largest)]
                sam = np.isin(labels, keep)
                self.used = bool(keep) and sam_agrees(sam, colour)
        self.accepted += self.used
        self.rejected += not self.used
        return sam if self.used else colour

    @property
    def quality(self) -> str:
        return "auto_sam" if self.used else "auto_exg"


def weak_labels(image: np.ndarray, det_id: int, seg_id: int,
                refiner: Optional[SamRefiner] = None) -> Optional[Tuple[List[str], List[str]]]:
    """Box and polygon lines for a single-plant close-up, or None if no usable mask."""
    mask = plant_mask(image)
    if mask is None:
        return None
    if refiner is not None:
        mask = refiner.refine(image, mask)
    height, width = image.shape[:2]
    polygons = mask_to_polygons(mask, min_area=MIN_COMPONENT_AREA, epsilon=2.0)
    if not polygons:
        return None
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return [to_yolo_line(box, width, height, det_id)], [polygon_line(seg_id, p, width, height) for p in polygons]


def collect_agri(seen: set, tmp: Path) -> Tuple[List[Item], dict]:
    images = [p for p in AGRI_DIR.glob("*.jpeg") if p.with_suffix(".txt").exists()]
    kept, dropped = drop_exact_duplicates(images, seen)
    print(f"agri: clustering {len(kept)} images by source photo...")
    clusters = source_photo_clusters(kept)
    items = []
    for path, cluster in zip(kept, clusters):
        dst = tmp / f"agri_{path.stem}.txt"
        _rewrite_agri_label(path.with_suffix(".txt"), dst)
        items.append(Item(f"agri_{path.stem}", path, "agri", f"agri{cluster}", "agri", "published_boxes_remapped_to_weed",
                          detection=dst.read_text(encoding="utf-8").splitlines()))
    sizes = Counter(clusters).values()
    return items, {"images": len(items), "exact_duplicates_dropped": [p.name for p in dropped],
                   "source_photo_groups": len(sizes), "largest_group": max(sizes)}


def peanut_labels(stem: str, width: int, height: int) -> Tuple[List[str], List[str], Dict[str, Tuple[int, int]]]:
    """Convert one mask to box lines and polygon lines, one object per plant."""
    mask = np.asarray(Image.open(PEANUT_MASKS / f"{stem}.png").convert("RGB"))
    crop, weed = _classify_mask(mask)
    boxes, polygons = [], []
    for sem, det_id, seg_id in ((crop, GROUNDNUT_ID, 0), (weed, WEED_ID, 1)):
        # Thin stems leave one plant's leaves disconnected, so nearby parts count as one plant.
        count, groups = cv2.connectedComponents(close_mask(sem, PLANT_JOIN_KERNEL, 1), connectivity=8)
        regions = [(groups == i) & sem for i in range(1, count)]
        for region in regions:
            ys, xs = np.nonzero(region)
            if xs.size == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if xs.size >= MIN_COMPONENT_AREA and x1 - x0 >= MIN_BOX_SIDE and y1 - y0 >= MIN_BOX_SIDE:
                boxes.append(f"{det_id} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                             f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
            polygons += [polygon_line(seg_id, poly, width, height) for poly in mask_to_polygons(region)]
    sums = {}
    for sem, seg_id, key in ((crop, 0, "crop"), (weed, 1, "weed")):
        drawn = rasterize(polygons, seg_id, width, height)
        sums[key] = (int((drawn & sem).sum()), int((drawn | sem).sum()))
    return boxes, polygons, sums


def collect_peanut(seen: set) -> Tuple[List[Item], dict]:
    images = [p for p in PEANUT_IMAGES.glob("*.jpg") if (PEANUT_MASKS / f"{p.stem}.png").exists()]
    kept, dropped = drop_exact_duplicates(images, seen)
    print(f"peanut: clustering and converting {len(kept)} masks...")
    clusters = source_photo_clusters(kept)
    items, totals = [], Counter()
    for path, cluster in zip(kept, clusters):
        with Image.open(path) as im:
            width, height = im.size
        boxes, polygons, sums = peanut_labels(path.stem, width, height)
        for key, (inter, union) in sums.items():
            totals[f"{key}_inter"] += inter
            totals[f"{key}_union"] += union
        items.append(Item(f"peanut_{path.stem}", path, "peanut", f"peanut{cluster}", "peanut",
                          "converted_from_published_mask", detection=boxes, segmentation=polygons))
    sizes = Counter(clusters).values()
    return items, {
        "images": len(items), "exact_duplicates_dropped": [p.name for p in dropped],
        "source_photo_groups": len(sizes), "largest_group": max(sizes),
        "polygon_vs_mask_iou": {k: round(totals[f"{k}_inter"] / max(1, totals[f"{k}_union"]), 4) for k in ("crop", "weed")},
    }


def collect_closeups(seen: set, refiner: SamRefiner) -> Tuple[List[Item], dict]:
    items, dropped, unusable = [], [], []
    for stage, folder in STAGE_FOLDERS.items():
        images = [p for p in (CLOSEUP_IMAGES / folder).iterdir() if p.is_file()]
        kept, dup = drop_exact_duplicates(images, seen)
        dropped += dup
        print(f"closeup/{stage}: building labels for {len(kept)} images...", flush=True)
        for path in kept:
            labels = weak_labels(read_bgr_gpu(path), GROUNDNUT_ID, 0, refiner)
            if labels is None:
                unusable.append(path.name)
                continue
            items.append(Item(f"closeup_{path.stem}", path, "closeup", dsc_group(path.stem), f"closeup_{stage}",
                              refiner.quality, detection=labels[0], segmentation=labels[1]))
    return items, {"images": len(items), "exact_duplicates_dropped": [p.name for p in dropped],
                   "no_usable_mask": unusable}


def collect_weeds(seen: set, cap: int, seed: int, refiner: SamRefiner) -> Tuple[List[Item], dict]:
    pool = [p for p in WEED_DIR.iterdir() if p.is_file() and re.match(r"DSC\d{5}", p.stem)
            and p.suffix.lower() in (".jpg", ".jpeg")]
    kept, dropped = drop_exact_duplicates(pool, seen)
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in kept:
        groups[dsc_group(path.stem)].append(path)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    items, unusable = [], []
    print(f"weed: building labels for up to {cap} of {len(kept)} images...", flush=True)
    for key in keys:
        if len(items) >= cap:
            break
        for path in groups[key]:
            labels = weak_labels(read_bgr_gpu(path), WEED_ID, 1, refiner)
            if labels is None:
                unusable.append(path.name)
                continue
            items.append(Item(f"weed_{path.stem}", path, "weed", key, "weed", refiner.quality,
                              detection=labels[0], segmentation=labels[1]))
    return items, {"pool": len(kept), "images": len(items), "exact_duplicates_dropped": len(dropped),
                   "no_usable_mask": unusable}


def manual_label_paths(image: Path, root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Where the click-to-label tool saves one image's box and polygon labels."""
    folder = root / image.parent.name if root else image.parent
    return folder / "labels" / f"{image.stem}.txt", folder / "labels_seg" / f"{image.stem}.txt"


def read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_manual_labels(image: Path) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    return tuple(read_lines(p) if p.exists() else None for p in manual_label_paths(image))


def read_machine_labels(image: Path) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Labels written by ``pseudo_label.py``, kept apart from the researcher's own."""
    paths = (image.parent / "labels_pseudo" / f"{image.stem}.txt", image.parent / "labels_pseudo_seg" / f"{image.stem}.txt")
    return tuple(read_lines(p) if p.exists() else None for p in paths)


def place_field_items(mixed: List[Item], field: List[Item], machine: List[Item], seed: int) -> Tuple[List[Item], List[Item]]:
    """Choose splits using each item's source-photo group, and return (field, machine) to keep.

    Mixed frames are always the test split. Other human labels split normally, joining the test
    split when they share a source photo with a mixed frame. Machine labels may only train, so one
    sharing a source photo with any human label in val or test is dropped.
    """
    test_groups = {item.group for item in mixed}
    for item in field:
        if item.group in test_groups:
            item.split = "test"
    assign_groups([item for item in field if not item.split], seed)
    held_out = test_groups | {item.group for item in field if item.split != "train"}
    kept = [item for item in machine if item.group not in held_out]
    for item in kept:
        item.split = "train"
    return field, kept


def collect_manual(seen: set, seed: int) -> Tuple[List[Item], List[Item], List[Item], dict]:
    """Own field photos labelled by the researcher, then those labelled by ``pseudo_label.py``.

    An image saved with no plants was skipped, not reviewed: nearly all of them show plants, and
    an empty label would teach a model those plants are background. Such images only count if
    ``pseudo_label.py`` labelled them. Returns (mixed, field, machine, info).
    """
    label_dir = MIXED_DIR / "labels"
    label_dir.mkdir(exist_ok=True)
    classes = label_dir / "classes.txt"
    if not classes.exists():
        classes.write_text("groundnut\nweed\n", encoding="utf-8")

    mixed_images = sorted(MIXED_DIR.glob("*.jpeg"))
    field_images = [p for batch in sorted(FIELD_DIR.iterdir()) if batch.is_dir()
                    for p in sorted(batch.iterdir()) if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")]
    candidates = [p for p in mixed_images + field_images if read_manual_labels(p)[0] or read_machine_labels(p)[0] is not None]
    kept, _ = drop_exact_duplicates(candidates, seen)
    clusters = source_photo_clusters(kept) if len(kept) > 1 else list(range(len(kept)))

    mixed, field, machine = [], [], []
    for path, cluster in zip(kept, clusters):
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
        folder = re.sub(r"[^A-Za-z0-9_-]+", "_", path.parent.name)
        detection, segmentation = read_manual_labels(path)
        if not detection:
            detection, segmentation = read_machine_labels(path)
            machine.append(Item(f"fieldauto_{folder}_{safe}", path, "field_machine", f"field{cluster}", "field_machine",
                                "machine_from_researcher_examples", detection=detection, segmentation=segmentation))
        elif path.parent == MIXED_DIR:
            mixed.append(Item(f"mixed_{safe}", path, "mixed", f"field{cluster}", "mixed", "manual",
                              detection=detection, segmentation=segmentation, split="test"))
        else:
            field.append(Item(f"field_{folder}_{safe}", path, "field", f"field{cluster}", "field", "manual",
                              detection=detection, segmentation=segmentation))
    field, kept_machine = place_field_items(mixed, field, machine, seed)
    return mixed, field, kept_machine, {
        "mixed_images": len(mixed_images), "mixed_labelled_by_researcher": len(mixed),
        "field_images": len(field_images), "field_labelled_by_researcher": len(field),
        "machine_labelled": len(machine), "machine_used_in_train": len(kept_machine),
        "machine_dropped_sharing_photo_with_val_or_test": len(machine) - len(kept_machine),
    }


def ends_like_jpeg(path: Path) -> bool:
    with open(path, "rb") as f:
        f.seek(-2, os.SEEK_END)
        return f.read() == b"\xff\xd9"


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics re-saves JPEGs with data after the end marker, as many phones write them.
    # Those get a real copy, so that re-save can never reach the original through a hard link.
    if src.suffix.lower() in (".jpg", ".jpeg") and not ends_like_jpeg(src):
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_task(root: Path, task: str, items: Sequence[Item], names: Dict[int, str], dataset: str) -> dict:
    out = root / task
    attr = "detection" if task == "detection" else "segmentation"
    chosen = [item for item in items if getattr(item, attr) is not None]
    counts: Dict[str, Counter] = {s: Counter() for s in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for item in chosen:
        link(item.image, out / "images" / item.split / f"{item.name}{item.image.suffix}")
        lines = getattr(item, attr)
        (out / "labels" / item.split / f"{item.name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[item.split]["images"] += 1
        counts[item.split][f"source_{item.source}"] += 1
        for line in lines:
            counts[item.split][f"objects_{names[int(line.split()[0])]}"] += 1
    yaml_lines = [
        "# Auto-generated by ml/data/build_yolo_datasets.py - see datasets/YOLO_DATASETS.md",
        f'path: "{out.resolve().as_posix()}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {k}: {v}" for k, v in names.items()]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    summary = {split: dict(counts[split]) for split in SPLIT_NAMES}
    (out / "metadata.json").write_text(json.dumps({
        "dataset_type": dataset, "task": task, "class_names": names, "splits": summary,
        "quality": dict(Counter(item.quality for item in chosen)),
    }, indent=2), encoding="utf-8")
    return summary


def prepare_output(dataset: str, force: bool) -> Path:
    root = DATASETS / dataset
    for task in ("detection", "segmentation"):
        out = root / task
        if not out.exists():
            continue
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not created by this script; move it first.")
        if not force:
            raise SystemExit(f"{out} already built (use --force to rebuild).")
        shutil.rmtree(out)
    for task in ("detection", "segmentation"):
        (root / task).mkdir(parents=True)
        (root / task / MARKER).write_text("generated output, safe to rebuild\n", encoding="utf-8")
    return root


def verify(datasets: Dict[str, Sequence[Item]]) -> List[str]:
    """Check every written pair and confirm no group crosses a split."""
    errors = []
    for dataset in datasets:
        for task, min_parts, names in (("detection", 5, DETECTION_NAMES), ("segmentation", 7, SEGMENTATION_NAMES)):
            base = DATASETS / dataset / task
            for split in SPLIT_NAMES:
                images = {p.stem for p in (base / "images" / split).iterdir() if not p.name.endswith(".cache")}
                labels = {p.stem for p in (base / "labels" / split).glob("*.txt")}
                if images != labels:
                    errors.append(f"{dataset}/{task}/{split}: image/label mismatch {sorted(images ^ labels)[:3]}")
                for stem in labels:
                    for number, line in enumerate((base / "labels" / split / f"{stem}.txt").read_text().splitlines(), 1):
                        parts = line.split()
                        values = [float(v) for v in parts[1:]]
                        where = f"{dataset}/{task}/{split}/{stem}:{number}"
                        if len(parts) < min_parts or int(parts[0]) not in names or not all(0 <= v <= 1 for v in values):
                            errors.append(f"{where}: bad line")
                        if task == "detection" and len(parts) != 5:
                            errors.append(f"{where}: expected 5 values")
                        if task == "segmentation" and len(values) % 2:
                            errors.append(f"{where}: odd coordinate count")
    for dataset, items in datasets.items():
        group_splits = defaultdict(set)
        for item in items:
            group_splits[(item.source, item.group)].add(item.split)
        for group, splits in group_splits.items():
            if len(splits) > 1:
                errors.append(f"{dataset}: group {group} spans {sorted(splits)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weed-cap", type=int, default=1500,
                    help="own weed close-ups to use; about equal to the groundnut count so weak "
                         "full-frame labels do not swamp the hybrid set")
    ap.add_argument("--force", action="store_true", help="rebuild output produced by this script")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU available. This script refuses to run on the CPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    tmp = DATASETS / "_yolo_build_tmp"
    tmp.mkdir(exist_ok=True)
    try:
        seen: set = set()
        peanut, peanut_info = collect_peanut(seen)
        agri, agri_info = collect_agri(seen, tmp)
        refiner = SamRefiner()
        closeups, closeup_info = collect_closeups(seen, refiner)
        weeds, weed_info = collect_weeds(seen, args.weed_cap, args.seed, refiner)
        mixed, field, machine, manual_info = collect_manual(seen, args.seed)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Groups are per source, so the same assignment is shared by all three datasets.
    assign_groups(peanut + agri + closeups + weeds, args.seed)
    primary = closeups + weeds + mixed + field + machine
    secondary = agri + peanut
    datasets = {"primary": primary, "secondary": secondary, "hybrid": primary + secondary}

    summary = {}
    for dataset, items in datasets.items():
        root = prepare_output(dataset, args.force)
        summary[dataset] = {
            "detection": write_task(root, "detection", items, DETECTION_NAMES, dataset),
            "segmentation": write_task(root, "segmentation", items, SEGMENTATION_NAMES, dataset),
        }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("name,source,split,group,quality,has_detection,has_segmentation,original\n")
        for item in primary + secondary:
            f.write(f'{item.name},{item.source},{item.split},{item.group},{item.quality},'
                    f'{item.detection is not None},{item.segmentation is not None},"{item.image.relative_to(PROJECT)}"\n')

    errors = verify(datasets)
    report = {"seed": args.seed, "weed_cap": args.weed_cap, "sources": {
        "peanut": peanut_info, "agri": agri_info, "closeup": closeup_info, "weed": weed_info, "manual": manual_info,
        "sam": {"accepted": refiner.accepted, "kept_colour_mask": refiner.rejected},
    }, "datasets": summary, "verification_errors": errors[:50]}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nVerification errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
