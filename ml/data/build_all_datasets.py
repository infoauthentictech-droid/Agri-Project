"""Generate the four dataset types for every trainable algorithm (GPU only).

Run ``ml/data/build_yolo_datasets.py`` first: it produces the labelled primary and
secondary sets and the split manifest this script builds on. Raw data is never changed.

Dataset types (under ``datasets/``):
  primary            own images, unedited
  primary_augmented  primary + 5 edited versions of each TRAIN image; val/test stay original
  secondary          public images only
  hybrid             primary + augmented versions + secondary

Tasks inside each type:
  detection/, segmentation/   YOLO layout with data.yaml   (YOLOv8n, YOLOv8n-Seg)
  classification/{split}/{crop,weed}/                      (the 8 classifiers)
  plant_gate/{split}/{plant,non_plant}/                    (Random Forest gate)
  cyclegan/domain_{a,b}/{train,test}/images/               (CycleGAN)

Aborts when no GPU is available. JPEG decoding, every edit and JPEG encoding run on the
GPU; only file reads and writes and the label arithmetic stay on the CPU.

Usage (from the project root):
    python ml/data/build_all_datasets.py [--factor 5] [--seed 42] [--force]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision.io import encode_jpeg

PROJECT =Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "ml" / "data"))

from gpu_loader import decode_file  # noqa: E402
from build_yolo_datasets import (  # noqa: E402
    DATASETS,
    DETECTION_NAMES,
    MANIFEST,
    MARKER,
    MIXED_DIR,
    OWN,
    SEGMENTATION_NAMES,
    SPLIT_NAMES,
    WEED_DIR,
    Item,
    assign_groups,
    link,
    source_photo_clusters,
)

DATASET_TYPES = ("primary", "primary_augmented", "secondary", "hybrid")
AUG_STORE = DATASETS / "primary_augmented" / "_augmented_images"
REPORT = DATASETS / "dataset_summary.json"
FIELD_DIR = OWN / "crop" / "groundnut_field_images"
NEGATIVE_DIRS = {
    "real": DATASETS / "negatives_real",
    "synthetic": DATASETS / "negatives_synthetic",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
CLASS_OF_SOURCE = {"closeup": "crop", "weed": "weed", "weed_field": "weed"}

RECIPES = ("hflip_dark", "vflip_bright", "rot90_contrast", "rot180_blur", "zoom_saturate")
JPEG_QUALITY = 92
ZOOM = 1.25
PATCH_PAD = 0.10  # extra context around a public box when cutting a classification patch
PATCH_MIN_SIDE = 48  # px; smaller boxes are too blurred to classify
BOX_MIN_VISIBLE = 0.30  # fraction of a box that must survive a zoom
BOX_MIN_SIDE = 0.005  # normalized
POLYGON_MIN_AREA = 1e-5  # normalized


Geometry = Tuple  # ("hflip",) ("vflip",) ("rot90ccw",) ("rot90cw",) ("rot180",) ("zoom", x0, y0, side)


@dataclass
class Sample:
    stem: str
    image: Path
    split: str
    source: str
    base: str  # the original photo this came from, used by the leakage check
    group: str
    klass: Optional[str] = None  # crop / weed for classification and CycleGAN
    detection: Optional[List[str]] = None
    segmentation: Optional[List[str]] = None
    gate: bool = True  # usable as a whole-frame "plant" example


def transform_points(points: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Map normalized (x, y) points through a geometric edit."""
    x, y = points[:, 0], points[:, 1]
    kind = geometry[0]
    if kind == "hflip":
        return np.stack((1 - x, y), axis=1)
    if kind == "vflip":
        return np.stack((x, 1 - y), axis=1)
    if kind == "rot90ccw":
        return np.stack((y, 1 - x), axis=1)
    if kind == "rot90cw":
        return np.stack((1 - y, x), axis=1)
    if kind == "rot180":
        return np.stack((1 - x, 1 - y), axis=1)
    if kind == "zoom":
        _, x0, y0, side = geometry
        return np.stack(((x - x0) / side, (y - y0) / side), axis=1)
    raise ValueError(f"unknown geometry {kind}")


def transform_image(image: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """Apply the same geometric edit to a CxHxW image tensor."""
    kind = geometry[0]
    if kind == "hflip":
        return F.hflip(image)
    if kind == "vflip":
        return F.vflip(image)
    if kind == "rot90ccw":
        return torch.rot90(image, 1, dims=(1, 2))
    if kind == "rot90cw":
        return torch.rot90(image, 3, dims=(1, 2))
    if kind == "rot180":
        return torch.rot90(image, 2, dims=(1, 2))
    if kind == "zoom":
        _, x0, y0, side = geometry
        height, width = image.shape[1:]
        return F.resized_crop(image, round(y0 * height), round(x0 * width), round(side * height),
                              round(side * width), [height, width], antialias=True)
    raise ValueError(f"unknown geometry {kind}")


def transform_box_line(line: str, geometry: Geometry) -> Optional[str]:
    """Transform one YOLO box; None when too little of it remains in frame."""
    parts = line.split()
    cx, cy, w, h = (float(v) for v in parts[1:5])
    corners = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    moved = transform_points(corners, geometry)
    x0, y0 = moved.min(axis=0)
    x1, y1 = moved.max(axis=0)
    full_area = (x1 - x0) * (y1 - y0)
    x0, y0, x1, y1 = max(x0, 0.0), max(y0, 0.0), min(x1, 1.0), min(y1, 1.0)
    new_w, new_h = x1 - x0, y1 - y0
    if new_w < BOX_MIN_SIDE or new_h < BOX_MIN_SIDE or new_w * new_h < BOX_MIN_VISIBLE * full_area:
        return None
    return f"{parts[0]} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {new_w:.6f} {new_h:.6f}"


def clip_polygon(points: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to the unit square."""
    def clip(pts: List[np.ndarray], axis: int, keep_low: bool, limit: float) -> List[np.ndarray]:
        inside = (lambda p: p[axis] >= limit) if keep_low else (lambda p: p[axis] <= limit)
        result = []
        for i, current in enumerate(pts):
            previous = pts[i - 1]
            if inside(current) != inside(previous):
                t = (limit - previous[axis]) / (current[axis] - previous[axis])
                result.append(previous + t * (current - previous))
            if inside(current):
                result.append(current)
        return result

    pts = list(points)
    for axis, keep_low, limit in ((0, True, 0.0), (0, False, 1.0), (1, True, 0.0), (1, False, 1.0)):
        if not pts:
            break
        pts = clip(pts, axis, keep_low, limit)
    return np.array(pts) if pts else np.zeros((0, 2))


def transform_polygon_line(line: str, geometry: Geometry) -> Optional[str]:
    """Transform one YOLO polygon; None when it leaves the frame."""
    parts = line.split()
    points = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
    moved = clip_polygon(transform_points(points, geometry))
    if len(moved) < 3:
        return None
    x, y = moved[:, 0], moved[:, 1]
    if 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) < POLYGON_MIN_AREA:
        return None
    return parts[0] + " " + " ".join(f"{min(max(v, 0.0), 1.0):.6f}" for v in moved.ravel())


def recipe_geometry(recipe: str, rng: random.Random) -> Geometry:
    if recipe == "hflip_dark":
        return ("hflip",)
    if recipe == "vflip_bright":
        return ("vflip",)
    if recipe == "rot90_contrast":
        return (rng.choice(("rot90ccw", "rot90cw")),)
    if recipe == "rot180_blur":
        return ("rot180",)
    if recipe == "zoom_saturate":
        side = 1 / ZOOM
        return ("zoom", rng.uniform(0, 1 - side), rng.uniform(0, 1 - side), side)
    raise ValueError(f"unknown recipe {recipe}")


def recipe_photometric(image: torch.Tensor, recipe: str, rng: random.Random) -> torch.Tensor:
    if recipe == "hflip_dark":
        return F.adjust_brightness(image, rng.uniform(0.65, 0.85))
    if recipe == "vflip_bright":
        return F.adjust_brightness(image, rng.uniform(1.15, 1.35))
    if recipe == "rot90_contrast":
        return F.adjust_contrast(image, rng.uniform(1.15, 1.35))
    if recipe == "rot180_blur":
        return F.gaussian_blur(image, [5, 5], [rng.uniform(0.6, 1.2)] * 2)
    if recipe == "zoom_saturate":
        return F.adjust_saturation(image, rng.uniform(1.1, 1.3))
    raise ValueError(f"unknown recipe {recipe}")


class GpuImages:
    """Decode and encode JPEGs on the GPU, counting any file nvJPEG cannot decode."""

    def __init__(self) -> None:
        self.gpu_decoded = 0
        self.cpu_decoded: List[str] = []

    def read(self, path: Path) -> torch.Tensor:
        image, on_gpu = decode_file(path)
        if on_gpu:
            self.gpu_decoded += 1
        else:
            self.cpu_decoded.append(path.name)
        return image

    @staticmethod
    def write(image: torch.Tensor, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encode_jpeg(image, quality=JPEG_QUALITY).cpu().numpy().tobytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")


def image_files(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_labelled() -> List[Sample]:
    """Read everything build_yolo_datasets.py produced."""
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} not found: run ml/data/build_yolo_datasets.py first.")
    samples = []
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = "secondary" if row["source"] in ("agri", "peanut") else "primary"
            root = DATASETS / dataset
            split, name = row["split"], row["name"]
            detection = read_lines(root / "detection/labels" / split / f"{name}.txt")
            segmentation = None
            if row["has_segmentation"] == "True":
                segmentation = read_lines(root / "segmentation/labels" / split / f"{name}.txt")
            samples.append(Sample(name, PROJECT / row["original"], split, row["source"], name, row["group"],
                                  CLASS_OF_SOURCE.get(row["source"]), detection, segmentation))
    return samples


def split_groups(entries: Sequence[Tuple[str, Path, str, str]], seed: int) -> List[Item]:
    """Assign (name, path, group, stratum) entries to splits, keeping groups whole."""
    items = [Item(name, path, "extra", group, stratum, "") for name, path, group, stratum in entries]
    assign_groups(items, seed)
    return items


def collect_field_images(labelled: Sequence[Sample], seed: int) -> Tuple[List[Sample], List[Sample], dict]:
    """Own photos with no per-object labels: real weed photos, mixed frames, wide field shots.

    Returns (weed photos usable for classification, gate-only photos, info). Photos sharing a
    source photo stay in one split, and exact copies of images already used are dropped.
    """
    seen = {sha256(s.image) for s in labelled}
    pools = [("weed_field", sorted(p for p in WEED_DIR.iterdir() if p.is_file() and p.name.startswith("WhatsApp")
                                   and p.suffix.lower() in IMAGE_SUFFIXES)),
             ("mixed", image_files(MIXED_DIR))]
    pools += [("field", image_files(batch)) for batch in sorted(FIELD_DIR.iterdir()) if batch.is_dir()]
    kept: List[Tuple[str, Path]] = []
    dropped = Counter()
    for source, paths in pools:
        for path in paths:
            digest = sha256(path)
            if digest in seen:
                dropped[source] += 1
                continue
            seen.add(digest)
            kept.append((source, path))

    print(f"own unlabelled photos: grouping {len(kept)} by source photo...", flush=True)
    clusters = source_photo_clusters([p for _, p in kept])
    entries = [(f"{source}_{safe_name(path.stem)}", path, f"own{cluster}", source)
               for (source, path), cluster in zip(kept, clusters)]
    weeds, gate_only = [], []
    for (source, _), item in zip(kept, split_groups(entries, seed)):
        sample = Sample(item.name, item.image, item.split, source, item.name, item.group, CLASS_OF_SOURCE.get(source))
        (weeds if source == "weed_field" else gate_only).append(sample)
    return weeds, gate_only, {"exact_copies_dropped": dict(dropped), "counts": dict(Counter(s for s, _ in kept))}


def collect_negatives(seed: int) -> Tuple[List[Sample], dict]:
    seen, entries, dropped = set(), [], 0
    for kind, folder in NEGATIVE_DIRS.items():
        for path in image_files(folder):
            digest = sha256(path)
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            entries.append((f"{kind}_{safe_name(path.stem)}", path, f"{kind}_{path.stem}", kind))
    items = split_groups(entries, seed)
    samples = [Sample(i.name, i.image, i.split, "negative", i.name, i.group) for i in items]
    return samples, {"images": len(samples), "exact_copies_dropped": dropped}


def augment(originals: Sequence[Sample], factor: int, seed: int, images: GpuImages) -> List[Sample]:
    """Create ``factor`` edited versions of every training image, labels included."""
    recipes = RECIPES[:factor]
    train = [s for s in originals if s.split == "train"]
    print(f"augmenting {len(train)} training images x{factor} on the GPU...", flush=True)
    versions = []
    for index, sample in enumerate(train, 1):
        image = images.read(sample.image)
        for recipe in recipes:
            rng = random.Random(f"{seed}:{sample.stem}:{recipe}")
            geometry = recipe_geometry(recipe, rng)
            stem = f"{sample.stem}__aug_{recipe}"
            path = AUG_STORE / f"{stem}.jpg"
            images.write(recipe_photometric(transform_image(image, geometry), recipe, rng), path)
            detection = segmentation = None
            if sample.detection is not None:
                moved = (transform_box_line(ln, geometry) for ln in sample.detection)
                detection = [ln for ln in moved if ln]
            if sample.segmentation is not None:
                moved = (transform_polygon_line(ln, geometry) for ln in sample.segmentation)
                segmentation = [ln for ln in moved if ln]
            versions.append(replace(sample, stem=stem, image=path, detection=detection, segmentation=segmentation))
        if index % 500 == 0:
            print(f"  {index}/{len(train)}", flush=True)
    return versions


def public_patches(public: Sequence[Sample], seed: int, images: GpuImages) -> Tuple[List[Sample], dict]:
    """Cut crop/weed classification patches from the public images' boxes, balanced per split.

    Public images have no whole-frame class, so the labelled objects become the examples.
    """
    candidates: Dict[Tuple[str, str], List[Tuple[Sample, int, str]]] = defaultdict(list)
    for sample in public:
        with Image.open(sample.image) as probe:
            width, height = probe.size
        for index, line in enumerate(sample.detection or []):
            parts = line.split()
            if float(parts[3]) * width < PATCH_MIN_SIDE or float(parts[4]) * height < PATCH_MIN_SIDE:
                continue
            klass = "crop" if int(parts[0]) == 0 else "weed"
            candidates[(sample.split, klass)].append((sample, index, line))

    rng = random.Random(seed)
    chosen: Dict[str, List[Tuple[Sample, int, str, str]]] = defaultdict(list)
    available = {f"{k[0]}/{k[1]}": len(v) for k, v in candidates.items()}
    for split in SPLIT_NAMES:
        limit = min(len(candidates[(split, "crop")]), len(candidates[(split, "weed")]))
        for klass in ("crop", "weed"):
            for sample, index, line in rng.sample(candidates[(split, klass)], limit):
                chosen[sample.stem].append((sample, index, line, klass))

    print(f"cutting {sum(len(v) for v in chosen.values())} public patches on the GPU...", flush=True)
    patches = []
    store = AUG_STORE.parent / "_public_patches"
    for stem, entries in sorted(chosen.items()):
        image = images.read(entries[0][0].image)
        height, width = image.shape[1:]
        for sample, index, line, klass in entries:
            cx, cy, w, h = (float(v) for v in line.split()[1:5])
            x0, x1 = max(0, int((cx - w * (0.5 + PATCH_PAD)) * width)), min(width, int((cx + w * (0.5 + PATCH_PAD)) * width))
            y0, y1 = max(0, int((cy - h * (0.5 + PATCH_PAD)) * height)), min(height, int((cy + h * (0.5 + PATCH_PAD)) * height))
            name = f"{stem}__p{index}"
            path = store / f"{name}.jpg"
            images.write(image[:, y0:y1, x0:x1], path)
            patches.append(Sample(name, path, sample.split, sample.source, sample.base, sample.group, klass, gate=False))
    return patches, {"available": available, "used_per_split_per_class":
                     {s: sum(1 for p in patches if p.split == s and p.klass == "crop") for s in SPLIT_NAMES}}


def prepare_task(dataset: str, task: str, force: bool) -> Path:
    out = DATASETS / dataset / task
    if out.exists():
        if not (out / MARKER).exists():
            raise SystemExit(f"{out} exists and was not created by these scripts; move it first.")
        if not force:
            raise SystemExit(f"{out} already built (use --force to rebuild).")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("generated output, safe to rebuild\n", encoding="utf-8")
    return out


def write_data_yaml(out: Path, names: Dict[int, str]) -> None:
    lines = ["# Auto-generated by ml/data/build_yolo_datasets.py and build_all_datasets.py",
             f'path: "{out.resolve().as_posix()}"', "train: images/train", "val: images/val", "test: images/test",
             "names:"] + [f"  {k}: {v}" for k, v in names.items()]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(samples: Iterable[Sample], key) -> dict:
    counts: Dict[str, Counter] = {s: Counter() for s in SPLIT_NAMES}
    for sample in samples:
        counts[sample.split][key(sample)] += 1
    return {s: dict(sorted(c.items())) for s, c in counts.items()}


def write_yolo(dataset: str, task: str, samples: Sequence[Sample], force: bool) -> dict:
    attr, names = ("detection", DETECTION_NAMES) if task == "detection" else ("segmentation", SEGMENTATION_NAMES)
    out = prepare_task(dataset, task, force)
    chosen = [s for s in samples if getattr(s, attr) is not None]
    for split in SPLIT_NAMES:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for sample in chosen:
        link(sample.image, out / "images" / sample.split / f"{sample.stem}{sample.image.suffix}")
        lines = getattr(sample, attr)
        (out / "labels" / sample.split / f"{sample.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_data_yaml(out, names)
    summary = summarize(chosen, lambda s: s.source)
    (out / "metadata.json").write_text(json.dumps(
        {"dataset_type": dataset, "task": task, "class_names": names, "images_by_source": summary}, indent=2),
        encoding="utf-8")
    return summary


def write_classification(dataset: str, samples: Sequence[Sample], force: bool) -> dict:
    out = prepare_task(dataset, "classification", force)
    chosen = [s for s in samples if s.klass]
    for sample in chosen:
        link(sample.image, out / sample.split / sample.klass / f"{sample.stem}{sample.image.suffix}")
    summary = summarize(chosen, lambda s: s.klass)
    (out / "metadata.json").write_text(json.dumps(
        {"dataset_type": dataset, "task": "classification", "images": summary,
         "images_by_source": summarize(chosen, lambda s: s.source)}, indent=2), encoding="utf-8")
    return summary


def write_gate(dataset: str, positives: Sequence[Sample], negatives: Sequence[Sample], force: bool) -> dict:
    out = prepare_task(dataset, "plant_gate", force)
    for klass, group in (("plant", positives), ("non_plant", negatives)):
        for sample in group:
            link(sample.image, out / sample.split / klass / f"{sample.stem}{sample.image.suffix}")
    summary = {s: {"plant": sum(p.split == s for p in positives), "non_plant": sum(n.split == s for n in negatives)}
               for s in SPLIT_NAMES}
    (out / "metadata.json").write_text(json.dumps(
        {"dataset_type": dataset, "task": "plant_gate", "images": summary}, indent=2), encoding="utf-8")
    return summary


def write_cyclegan(dataset: str, samples: Sequence[Sample], extra_infested: Sequence[Sample], force: bool) -> dict:
    """Domain A: groundnut. Domain B: weeds, plus own mixed frames (the truest 'weed-infested' photos)."""
    out = prepare_task(dataset, "cyclegan", force)
    counts: Dict[str, Counter] = {"domain_a": Counter(), "domain_b": Counter()}
    domain_of = {"crop": "domain_a", "weed": "domain_b"}
    placed = [(domain_of[s.klass], s) for s in samples if s.klass] + [("domain_b", s) for s in extra_infested]
    for domain, sample in placed:
        if sample.split == "val":
            continue
        link(sample.image, out / domain / sample.split / "images" / f"{sample.stem}{sample.image.suffix}")
        counts[domain][sample.split] += 1
    summary = {d: dict(c) for d, c in counts.items()}
    (out / "metadata.json").write_text(json.dumps(
        {"dataset_type": dataset, "task": "cyclegan", "images": summary}, indent=2), encoding="utf-8")
    return summary


def check_yolo(dataset: str, task: str) -> List[str]:
    errors = []
    base = DATASETS / dataset / task
    min_parts, names = (5, DETECTION_NAMES) if task == "detection" else (7, SEGMENTATION_NAMES)
    for split in SPLIT_NAMES:
        images = {p.stem for p in (base / "images" / split).iterdir() if p.suffix != ".cache"}
        labels = {p.stem for p in (base / "labels" / split).glob("*.txt")}
        if not images:
            errors.append(f"{dataset}/{task}/{split}: empty")
        if images != labels:
            errors.append(f"{dataset}/{task}/{split}: image/label mismatch {sorted(images ^ labels)[:3]}")
        if split != "train" and any("__aug_" in stem for stem in images):
            errors.append(f"{dataset}/{task}/{split}: augmented image outside train")
        for stem in labels:
            for number, line in enumerate(read_lines(base / "labels" / split / f"{stem}.txt"), 1):
                parts = line.split()
                values = [float(v) for v in parts[1:]]
                if (len(parts) < min_parts or int(parts[0]) not in names or not all(0 <= v <= 1 for v in values)
                        or (task == "detection" and len(parts) != 5) or (task != "detection" and len(values) % 2)):
                    errors.append(f"{dataset}/{task}/{split}/{stem}:{number}: bad line")
    return errors


def check_leakage(dataset: str, samples: Iterable[Sample]) -> List[str]:
    """A source photo or group must sit in exactly one split, whatever was derived from it."""
    bases, groups = defaultdict(set), defaultdict(set)
    for sample in samples:
        bases[sample.base].add(sample.split)
        groups[(sample.source, sample.group)].add(sample.split)
    errors = [f"{dataset}: photo {k} in {sorted(v)}" for k, v in bases.items() if len(v) > 1]
    return errors + [f"{dataset}: group {k} in {sorted(v)}" for k, v in groups.items() if len(v) > 1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factor", type=int, default=5, choices=range(1, len(RECIPES) + 1),
                    help="edited versions per training image")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="rebuild output produced by these scripts")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU available. This script refuses to run on the CPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    started = time.time()

    labelled = load_labelled()
    own = [s for s in labelled if s.source not in ("agri", "peanut")]
    public = [s for s in labelled if s.source in ("agri", "peanut")]
    for sample in public:
        sample.gate = True

    weed_field, gate_only, field_info = collect_field_images(labelled, args.seed)
    negatives, negative_info = collect_negatives(args.seed)
    mixed = [s for s in gate_only + own if s.source == "mixed"]

    for store in (AUG_STORE, AUG_STORE.parent / "_public_patches"):
        if store.exists():
            if not args.force:
                raise SystemExit(f"{store} already built (use --force to rebuild).")
            shutil.rmtree(store)
        store.mkdir(parents=True)

    images = GpuImages()
    primary = own + weed_field
    augmented = augment(primary, args.factor, args.seed, images)
    patches, patch_info = public_patches(public, args.seed, images)

    members = {
        "primary": primary,
        "primary_augmented": primary + augmented,
        "secondary": public + patches,
        "hybrid": primary + augmented + public + patches,
    }
    print("linking datasets...", flush=True)
    summary: Dict[str, dict] = {}
    for dataset in DATASET_TYPES:
        samples = members[dataset]
        entry = {}
        for task in ("detection", "segmentation"):
            if dataset in ("primary_augmented", "hybrid"):
                entry[task] = write_yolo(dataset, task, samples, args.force)
            else:  # built and verified by build_yolo_datasets.py
                out = DATASETS / dataset / task
                write_data_yaml(out, DETECTION_NAMES if task == "detection" else SEGMENTATION_NAMES)
                entry[task] = summarize((s for s in samples if getattr(s, task) is not None), lambda s: s.source)
        entry["classification"] = write_classification(dataset, samples, args.force)
        own_frames = gate_only if dataset != "secondary" else []
        positives = [s for s in samples if s.gate] + list(own_frames)
        entry["plant_gate"] = write_gate(dataset, positives, negatives, args.force)
        entry["cyclegan"] = write_cyclegan(dataset, samples, mixed if dataset != "secondary" else [], args.force)
        summary[dataset] = entry

    errors: List[str] = []
    for dataset in DATASET_TYPES:
        for task in ("detection", "segmentation"):
            errors += check_yolo(dataset, task)
        errors += check_leakage(dataset, members[dataset] + (gate_only if dataset != "secondary" else []))

    report = {
        "seed": args.seed, "factor": args.factor, "recipes": RECIPES[:args.factor],
        "gpu": torch.cuda.get_device_name(0), "minutes": round((time.time() - started) / 60, 1),
        "gpu_decoded": images.gpu_decoded, "cpu_decoded_then_gpu_edited": images.cpu_decoded,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() // 2**20,
        "own_unlabelled_photos": field_info, "negatives": negative_info, "public_patches": patch_info,
        "datasets": summary, "verification_errors": errors[:50], "verification_error_count": len(errors),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nVerification errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
