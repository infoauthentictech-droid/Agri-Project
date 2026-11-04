"""Label the remaining own field photos from the researcher's own labelled examples (GPU only).

Each colour outline is compared with the plants the researcher marked, using pretrained
ResNet-50 features and a nearest-neighbour vote, so no model is trained. An image is labelled
only when almost all of its plant area is classified confidently; the rest stay unlabelled.

These are machine labels. They are saved separately in ``labels_pseudo/`` and
``labels_pseudo_seg/`` beside each image folder, never overwrite the researcher's labels, and
``build_yolo_datasets.py`` only ever puts them in the training split.

Usage (from the project root):
    python ml/data/pseudo_label.py --evaluate    measure accuracy on the researcher's labels
    python ml/data/pseudo_label.py               write the labels
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "ml" / "data"))

from build_yolo_datasets import DATASETS, manual_label_paths, read_manual_labels, source_photo_clusters  # noqa: E402
from gpu_loader import IMAGENET_MEAN, IMAGENET_STD, decode_file  # noqa: E402
from label_tool import label_lines, list_images, objects_path, propose_objects  # noqa: E402

REPORT = DATASETS / "pseudo_label_report.json"
CROP_PAD = 0.15  # context around each plant, as a share of its box
NEIGHBOURS = 15
CONFIDENCE = 0.80  # a plant counts as classified at or above this vote share
MIN_CONFIDENT_AREA = 0.90  # an image is labelled only if this share of its plant area is classified
BATCH = 64


@dataclass
class Region:
    image: int
    polygons: List[List[float]]
    label: int = -1


def box_of(polygons: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    points = np.array([v for polygon in polygons for v in polygon]).reshape(-1, 2)
    return (*points.min(axis=0), *points.max(axis=0))


def polygon_area(polygons: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for polygon in polygons:
        pts = np.array(polygon).reshape(-1, 2)
        total += 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    return total


class Embedder:
    def __init__(self) -> None:
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to("cuda").eval()
        self.model.fc = torch.nn.Identity()
        self.prepare = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, images: Sequence[Path], regions: Sequence[Region]) -> torch.Tensor:
        """L2-normalized features for each region, a crop around the plant with some context."""
        features, pending = [], []
        by_image: Dict[int, List[Region]] = {}
        order = {id(r): i for i, r in enumerate(regions)}
        for region in regions:
            by_image.setdefault(region.image, []).append(region)

        def flush() -> None:
            with torch.inference_mode():
                features.append(self.model(torch.stack([c for _, c in pending])))
            ids.extend(i for i, _ in pending)
            pending.clear()

        ids: List[int] = []
        for index, group in by_image.items():
            image, _ = decode_file(images[index])
            height, width = image.shape[1:]
            for region in group:
                x0, y0, x1, y1 = box_of(region.polygons)
                pad_x, pad_y = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
                left, right = max(0, int((x0 - pad_x) * width)), min(width, int((x1 + pad_x) * width) + 1)
                top, bottom = max(0, int((y0 - pad_y) * height)), min(height, int((y1 + pad_y) * height) + 1)
                pending.append((order[id(region)], self.prepare(image[:, top:bottom, left:right])))
                if len(pending) == BATCH:
                    flush()
        if pending:
            flush()
        stacked = torch.cat(features)
        out = torch.zeros_like(stacked)
        out[torch.tensor(ids, device="cuda")] = stacked
        return torch.nn.functional.normalize(out, dim=1)


def vote(queries: torch.Tensor, references: torch.Tensor, labels: torch.Tensor,
         exclude: torch.Tensor = None) -> torch.Tensor:
    """Class-balanced nearest-neighbour vote. Returns each query's share of weed votes.

    ``exclude`` is a query x reference boolean matrix of neighbours that must not vote, used to
    keep every photo out of its own evaluation.
    """
    weights = torch.where(labels == 1, 0.5 / (labels == 1).sum(), 0.5 / (labels == 0).sum())
    similarity = queries @ references.T
    if exclude is not None:
        similarity = similarity.masked_fill(exclude, -1.0)
    top, index = similarity.topk(NEIGHBOURS, dim=1)
    votes = weights[index] * top.clamp(min=0) ** 4
    weed = (votes * (labels[index] == 1)).sum(dim=1)
    return weed / votes.sum(dim=1).clamp(min=1e-12)


def human_regions(images: Sequence[Path]) -> List[Region]:
    regions = []
    for index, path in enumerate(images):
        saved = objects_path(manual_label_paths(path)[1])
        if saved.exists():
            regions += [Region(index, obj["polygons"], obj["cls"]) for obj in json.loads(saved.read_text(encoding="utf-8"))]
    return regions


def evaluate(regions: Sequence[Region], features: torch.Tensor, clusters: Sequence[int], images: Sequence[Path]) -> dict:
    """Leave out every photo, and any near-copy of it, then predict its plants from the others."""
    labels = torch.tensor([r.label for r in regions], device="cuda")
    group = torch.tensor([clusters[r.image] for r in regions], device="cuda")
    exclude = group[:, None] == group[None, :]
    scores = vote(features, features, labels, exclude).cpu().numpy()
    truth = labels.cpu().numpy()
    areas = np.array([polygon_area(r.polygons) for r in regions])
    predicted = (scores >= 0.5).astype(int)
    confidence = np.maximum(scores, 1 - scores)

    def summary(mask: np.ndarray) -> dict:
        def recall(cls: int) -> float:
            hit = mask & (truth == cls)
            return round(float((predicted[hit] == cls).mean()), 3) if hit.any() else float("nan")
        return {"plants": int(mask.sum()), "accuracy": round(float((predicted[mask] == truth[mask]).mean()), 3),
                "groundnut_recall": recall(0), "weed_recall": recall(1),
                "area_weighted_accuracy": round(float(areas[mask][predicted[mask] == truth[mask]].sum() / areas[mask].sum()), 3)}

    result = {"all_plants": summary(np.ones_like(truth, dtype=bool)),
              f"confidence_{CONFIDENCE}": summary(confidence >= CONFIDENCE)}
    folders = np.array([images[r.image].parent.name for r in regions])
    result["by_folder"] = {str(f): summary(folders == f) for f in sorted(set(folders))}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate", action="store_true", help="only measure accuracy; write no labels")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU available. This script refuses to run on the CPU.")

    images = list_images()
    print(f"{len(images)} images; grouping by source photo...", flush=True)
    clusters = source_photo_clusters(images)
    reference = human_regions(images)
    print(f"reference: {len(reference)} plants you marked, {dict(Counter(r.label for r in reference))}", flush=True)
    embedder = Embedder()
    ref_features = embedder(images, reference)
    held_out = evaluate(reference, ref_features, clusters, images)
    print(json.dumps(held_out, indent=2), flush=True)
    if args.evaluate:
        return 0

    ref_labels = torch.tensor([r.label for r in reference], device="cuda")
    written, skipped = [], []
    for index, path in enumerate(images):
        if read_manual_labels(path)[0]:
            continue
        proposals = [Region(index, polygons) for polygons in propose_objects(path)]
        if not proposals:
            skipped.append(path.name)
            continue
        scores = vote(embedder(images, proposals), ref_features, ref_labels).cpu().numpy()
        areas = np.array([polygon_area(r.polygons) for r in proposals])
        confident = np.maximum(scores, 1 - scores) >= CONFIDENCE
        if areas[confident].sum() < MIN_CONFIDENT_AREA * areas.sum():
            skipped.append(path.name)
            continue
        objects = [{"cls": int(score >= 0.5), "polygons": region.polygons}
                   for region, score, keep in zip(proposals, scores, confident) if keep]
        boxes, polygons = label_lines(objects)
        for folder, lines in (("labels_pseudo", boxes), ("labels_pseudo_seg", polygons)):
            out = path.parent / folder / f"{path.stem}.txt"
            out.parent.mkdir(exist_ok=True)
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path.name)
    print(f"labelled {len(written)}, left unlabelled {len(skipped)}", flush=True)
    REPORT.write_text(json.dumps({"reference_plants": len(reference), "held_out_accuracy": held_out,
                                  "labelled": written, "left_unlabelled": skipped}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
