"""Dataset validation orchestrator.

Read-only over the raw data. Produces a serializable per-dataset validation
record containing image integrity, duplicates, dimensions, label/mask checks
and class statistics, plus a ``path -> fingerprint`` map for cross-dataset
leakage detection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .common import file_fingerprint
from .duplicates import group_duplicates
from .image_utils import inspect_image
from .labels_yolo import parse_yolo_label
from .labels_seg import inspect_mask


def _image_files(images_dir: Path, exts: List[str]) -> List[Path]:
    allowed = {e.lower() for e in exts}
    return sorted(
        f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed
    )


def validate_dataset(cfg: Dict, key: str, logger: logging.Logger) -> Tuple[Dict, Dict]:
    """Validate one dataset; returns ``(record, fingerprint_map)``.

    ``fingerprint_map`` maps absolute image paths to content fingerprints so the
    orchestrator can detect cross-dataset duplicates afterwards.
    """
    d = cfg["datasets"][key]
    raw_root = Path(cfg["raw_root"])
    exts = d["image_ext"] if isinstance(d["image_ext"], list) else [d["image_ext"]]
    images_dir = raw_root / d["images_dir"]
    files = _image_files(images_dir, exts)
    logger.info("[%s] validating %d image files in %s", key, len(files), images_dir)

    rec: Dict = {
        "dataset": key,
        "task": d["task"],
        "total_images": len(files),
        "valid_images": 0,
        "invalid_images": 0,
        "corrupted": [],
        "dimensions": {},
        "modes": {},
        "duplicate_groups": [],
        "duplicate_map": {},
        "duplicate_extra_copies": 0,
        "labels": None,
        "masks": None,
        "issues": [],
        "file_map": {f.stem: str(f) for f in files},
    }

    for f in files:
        info = inspect_image(f)
        if info.valid:
            rec["valid_images"] += 1
            dim = f"{info.width}x{info.height}"
            rec["dimensions"][dim] = rec["dimensions"].get(dim, 0) + 1
            rec["modes"][info.mode] = rec["modes"].get(info.mode, 0) + 1
        else:
            rec["invalid_images"] += 1
            rel = str(f.relative_to(raw_root)).replace("\\", "/")
            rec["corrupted"].append({"path": rel, "error": info.error})
            logger.warning("[%s] corrupted image %s :: %s", key, rel, info.error)

    fps = {f: file_fingerprint(f) for f in files}
    for fp, paths in group_duplicates(files).items():
        rec["duplicate_groups"].append(
            {"fingerprint": fp[:16], "files": [str(p) for p in paths]}
        )
        for p in paths:
            rec["duplicate_map"][Path(p).stem] = fp
    rec["duplicate_extra_copies"] = sum(
        len(g["files"]) - 1 for g in rec["duplicate_groups"]
    )

    task = d["task"]
    if task == "detection":
        rec["labels"] = _validate_detection(cfg, d, raw_root, files)
        lab = rec["labels"]
        if lab["labels_without_image"]:
            rec["issues"].append(
                f"{len(lab['labels_without_image'])} label files without matching image"
            )
        if lab["images_without_label"]:
            rec["issues"].append(
                f"{len(lab['images_without_label'])} images without any label"
            )
        if lab["empty_annotations"]:
            rec["issues"].append(f"{len(lab['empty_annotations'])} empty annotations")
        if lab["invalid_labels"]:
            rec["issues"].append(f"{len(lab['invalid_labels'])} invalid label files")
    elif task == "segmentation":
        rec["masks"] = _validate_segmentation(cfg, d, raw_root, files)
        msk = rec["masks"]
        if msk["images_without_mask"]:
            rec["issues"].append(f"{len(msk['images_without_mask'])} images without mask")
        if msk["masks_without_image"]:
            rec["issues"].append(f"{len(msk['masks_without_image'])} masks without image")
        if msk["size_mismatches"]:
            rec["issues"].append(f"{len(msk['size_mismatches'])} mask size mismatches")
        if msk["empty_masks"]:
            rec["issues"].append(f"{len(msk['empty_masks'])} empty masks (no foreground)")
    else:
        logger.info("[%s] task '%s': image-only inventory", key, task)

    for issue in rec["issues"]:
        logger.warning("[%s] %s", key, issue)
    logger.info(
        "[%s] total=%d valid=%d invalid=%d dup_extra_copies=%d",
        key,
        rec["total_images"],
        rec["valid_images"],
        rec["invalid_images"],
        rec["duplicate_extra_copies"],
    )
    return rec, fps



def _validate_detection(cfg: Dict, d: Dict, raw_root: Path, files: List[Path]) -> Dict:
    """Validate YOLO detection annotations against the image set."""
    labels_dir = raw_root / d["labels_dir"]
    l_ext = d["label_ext"].lstrip(".").lower()
    label_files: Dict[str, Path] = {}
    if labels_dir.exists():
        for f in labels_dir.rglob(f"*.{l_ext}"):
            if f.is_file():
                label_files[f.stem] = f

    image_stems = {f.stem for f in files}
    num_classes = d.get("num_classes") or len(d.get("classes", {}))

    out: Dict = {
        "label_files": len(label_files),
        "num_classes": num_classes,
        "images_with_labels": 0,
        "images_without_label": sorted(image_stems - set(label_files)),
        "labels_without_image": sorted(set(label_files) - image_stems),
        "total_boxes": 0,
        "boxes_per_image": [],
        "class_boxes": {},
        "class_images": {},
        "empty_annotations": [],
        "invalid_labels": [],
        "malformed_lines": [],
    }
    for stem in sorted(image_stems):
        if stem not in label_files:
            continue
        res = parse_yolo_label(label_files[stem], num_classes)
        out["images_with_labels"] += 1
        out["total_boxes"] += res.box_count
        out["boxes_per_image"].append(res.box_count)
        for cid, cnt in res.classes.items():
            key = str(cid)
            out["class_boxes"][key] = out["class_boxes"].get(key, 0) + cnt
            out["class_images"][key] = out["class_images"].get(key, 0) + 1
        if res.empty:
            out["empty_annotations"].append(stem)
        if not res.valid:
            out["invalid_labels"].append({"image": stem, "errors": res.errors})
            for err in res.errors:
                if err.startswith("line "):
                    out["malformed_lines"].append({"image": stem, "error": err})
    return out


def _validate_segmentation(cfg: Dict, d: Dict, raw_root: Path, files: List[Path]) -> Dict:
    """Validate segmentation masks against the image set (Peanut RGB masks)."""
    masks_dir = raw_root / d["masks_dir"]
    m_ext = d["mask_ext"].lstrip(".").lower()
    mask_files: Dict[str, Path] = {}
    if masks_dir.exists():
        for f in masks_dir.rglob(f"*.{m_ext}"):
            if f.is_file():
                mask_files[f.stem] = f

    img_by_stem = {f.stem: f for f in files}
    out: Dict = {
        "mask_files": len(mask_files),
        "images_with_mask": 0,
        "images_without_mask": [],
        "masks_without_image": [],
        "size_mismatches": [],
        "mode_issues": [],
        "empty_masks": [],
        "class_presence": {"background": 0, "crop": 0, "weed": 0},
        "invalid_masks": [],
    }
    for stem in sorted(set(img_by_stem) | set(mask_files)):
        if stem not in img_by_stem:
            out["masks_without_image"].append(stem)
            continue
        if stem not in mask_files:
            out["images_without_mask"].append(stem)
            continue
        res = inspect_mask(img_by_stem[stem], mask_files[stem])
        out["images_with_mask"] += 1
        if not res.size_match:
            out["size_mismatches"].append(
                {
                    "image": stem,
                    "image_size": res.image_size,
                    "mask_size": res.mask_size,
                }
            )
        if res.mode_error:
            out["mode_issues"].append({"image": stem, "mode": res.mode})
        if res.empty:
            out["empty_masks"].append(stem)
        for cls in res.class_presence:
            if cls in out["class_presence"]:
                out["class_presence"][cls] += 1
        if not res.valid and not res.empty:
            out["invalid_masks"].append({"image": stem, "errors": res.errors})
    return out