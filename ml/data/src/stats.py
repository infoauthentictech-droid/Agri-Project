"""Dataset statistics aggregation (no fabricated numbers — everything is
derived from the validated records produced by the pipeline).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .common import utc_now


def _imbalance(values: List[int]):
    """Return max/min ratio (None when undefined)."""
    if not values:
        return None
    m = min(values)
    if m <= 0:
        return None
    return round(max(values) / m, 3)


def build_statistics(records: Dict[str, Dict[str, Any]], overlaps: List[dict]) -> Dict[str, Any]:
    """Aggregate per-dataset statistics from validation records."""
    stats: Dict[str, Any] = {"generated_at": utc_now(), "datasets": {}}

    for key, rec in records.items():
        s: Dict[str, Any] = {
            "task": rec["task"],
            "total_images": rec["total_images"],
            "valid_images": rec["valid_images"],
            "invalid_images": rec["invalid_images"],
            "corrupted_files": len(rec["corrupted"]),
            "duplicate_groups": len(rec["duplicate_groups"]),
            "duplicate_extra_copies": rec["duplicate_extra_copies"],
            "dimensions": rec["dimensions"],
            "modes": rec["modes"],
        }

        if rec.get("labels") is not None:
            lab = rec["labels"]
            s["annotation_type"] = "yolo_detection"
            s["images_with_labels"] = lab["images_with_labels"]
            s["images_without_label"] = len(lab["images_without_label"])
            s["labels_without_image"] = len(lab["labels_without_image"])
            s["total_boxes"] = lab["total_boxes"]
            bpi = lab["boxes_per_image"]
            s["boxes_per_image_min"] = min(bpi) if bpi else 0
            s["boxes_per_image_mean"] = round(sum(bpi) / len(bpi), 3) if bpi else 0
            s["boxes_per_image_max"] = max(bpi) if bpi else 0
            s["class_boxes"] = lab["class_boxes"]
            s["images_per_class"] = lab["class_images"]
            s["class_imbalance_ratio"] = _imbalance(list(lab["class_boxes"].values()))
            s["empty_annotations"] = len(lab["empty_annotations"])
            s["invalid_labels"] = len(lab["invalid_labels"])
            s["malformed_lines"] = len(lab["malformed_lines"])

        if rec.get("masks") is not None:
            msk = rec["masks"]
            s["annotation_type"] = "segmentation"
            s["images_with_mask"] = msk["images_with_mask"]
            s["images_without_mask"] = len(msk["images_without_mask"])
            s["masks_without_image"] = len(msk["masks_without_image"])
            s["size_mismatches"] = len(msk["size_mismatches"])
            s["mode_issues"] = len(msk["mode_issues"])
            s["empty_masks"] = len(msk["empty_masks"])
            s["mask_class_presence"] = msk["class_presence"]
            s["class_imbalance_ratio"] = _imbalance(
                [v for k, v in msk["class_presence"].items() if k != "background"]
            )

        stats["datasets"][key] = s

    stats["cross_dataset_overlaps"] = overlaps
    return stats
