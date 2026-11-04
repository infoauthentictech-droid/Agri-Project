"""Annotation Quality Checks

Validates bounding boxes, class IDs, segmentation masks, empty-label cases,
image-mask dimensions, and annotation quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AnnotationIssue(str, Enum):
    """Types of annotation issues."""
    INVALID_BBOX = "invalid_bbox"
    INVALID_CLASS_ID = "invalid_class_id"
    MASK_MISMATCH = "mask_mismatch"
    EMPTY_LABEL = "empty_label"
    DIMENSION_MISMATCH = "dimension_mismatch"
    OUT_OF_BOUNDS = "out_of_bounds"
    TOO_SMALL = "too_small"
    TOO_LARGE = "too_large"


@dataclass
class AnnotationValidationResult:
    """Result of annotation validation."""
    is_valid: bool
    issues: List[Tuple[AnnotationIssue, str]]
    statistics: Dict
    file_path: str


class AnnotationValidator:
    """Validator for dataset annotations."""

    def __init__(self):
        """Initialize annotation validator."""
        self.min_bbox_area = 100  # Minimum bounding box area in pixels
        self.max_bbox_area = 90 * 90  # Maximum bounding box area (90% of 100x100)

    def validate_detection_annotation(
        self, image_path: Path, label_path: Path
    ) -> AnnotationValidationResult:
        """Validate YOLO detection annotation.

        Args:
            image_path: Path to image
            label_path: Path to YOLO label file

        Returns:
            AnnotationValidationResult
        """
        issues = []
        stats = {"num_boxes": 0, "class_distribution": {}}

        # Check image exists
        if not image_path.exists():
            return AnnotationValidationResult(
                is_valid=False,
                issues=[(AnnotationIssue.INVALID_BBOX, f"Image not found: {image_path}")],
                statistics=stats,
                file_path=str(image_path),
            )

        # Check label exists
        if not label_path.exists():
            return AnnotationValidationResult(
                is_valid=False,
                issues=[(AnnotationIssue.INVALID_BBOX, f"Label not found: {label_path}")],
                statistics=stats,
                file_path=str(image_path),
            )

        # Load image dimensions
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size
        except Exception as e:
            return AnnotationValidationResult(
                is_valid=False,
                issues=[(AnnotationIssue.INVALID_BBOX, f"Failed to load image: {e}")],
                statistics=stats,
                file_path=str(image_path),
            )

        # Parse YOLO label
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                issues.append(
                    (AnnotationIssue.INVALID_BBOX, f"Line {line_num}: Invalid format (expected 5+ values)")
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                issues.append(
                    (AnnotationIssue.INVALID_BBOX, f"Line {line_num}: Invalid numeric values")
                )
                continue

            # Validate class ID
            if class_id not in [0, 1]:
                issues.append(
                    (AnnotationIssue.INVALID_CLASS_ID, f"Line {line_num}: Invalid class ID {class_id}")
                )

            # Validate normalized coordinates
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                issues.append(
                    (AnnotationIssue.OUT_OF_BOUNDS, f"Line {line_num}: Center out of bounds")
                )

            if not (0 < width <= 1 and 0 < height <= 1):
                issues.append(
                    (AnnotationIssue.INVALID_BBOX, f"Line {line_num}: Invalid width/height")
                )

            # Convert to absolute coordinates
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            # Validate bbox is within image bounds
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                issues.append(
                    (AnnotationIssue.OUT_OF_BOUNDS, f"Line {line_num}: Bbox out of image bounds")
                )

            # Validate bbox has positive area
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                issues.append(
                    (AnnotationIssue.INVALID_BBOX, f"Line {line_num}: Invalid bbox (non-positive area)")
                )

            # Validate bbox size
            if area < self.min_bbox_area:
                issues.append(
                    (AnnotationIssue.TOO_SMALL, f"Line {line_num}: Bbox too small ({area} px)")
                )

            if area > self.max_bbox_area:
                issues.append(
                    (AnnotationIssue.TOO_LARGE, f"Line {line_num}: Bbox too large ({area} px)")
                )

            # Update statistics
            stats["num_boxes"] += 1
            stats["class_distribution"][str(class_id)] = (
                stats["class_distribution"].get(str(class_id), 0) + 1
            )

        # Check for empty label
        if stats["num_boxes"] == 0:
            issues.append((AnnotationIssue.EMPTY_LABEL, "No bounding boxes in label"))

        is_valid = len(issues) == 0

        return AnnotationValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=stats,
            file_path=str(image_path),
        )

    def validate_segmentation_annotation(
        self, image_path: Path, label_path: Path, mask_path: Path
    ) -> AnnotationValidationResult:
        """Validate segmentation annotation.

        Args:
            image_path: Path to image
            label_path: Path to YOLO label file
            mask_path: Path to segmentation mask

        Returns:
            AnnotationValidationResult
        """
        issues = []
        stats = {"num_objects": 0, "mask_coverage": 0.0}

        # Validate detection annotation first
        detection_result = self.validate_detection_annotation(image_path, label_path)
        issues.extend(detection_result.issues)

        # Validate mask dimensions match image
        try:
            img = Image.open(image_path)
            img_w, img_h = img.size

            if mask_path.exists():
                mask = Image.open(mask_path)
                mask_w, mask_h = mask.size

                if mask_w != img_w or mask_h != img_h:
                    issues.append(
                        (
                            AnnotationIssue.DIMENSION_MISMATCH,
                            f"Mask dimensions ({mask_w}x{mask_h}) don't match image ({img_w}x{img_h})",
                        )
                    )

                # Calculate mask coverage
                mask_array = np.array(mask)
                if len(mask_array.shape) == 3:
                    # RGB mask - assume non-zero pixels are foreground
                    mask_coverage = (mask_array > 0).sum() / (mask_w * mask_h * 3)
                else:
                    # Grayscale mask
                    mask_coverage = (mask_array > 0).sum() / (mask_w * mask_h)

                stats["mask_coverage"] = float(mask_coverage)

        except Exception as e:
            issues.append((AnnotationIssue.DIMENSION_MISMATCH, f"Failed to validate mask: {e}"))

        is_valid = len(issues) == 0

        return AnnotationValidationResult(
            is_valid=is_valid,
            issues=issues,
            statistics=stats,
            file_path=str(image_path),
        )

    def validate_dataset(
        self, dataset_root: Path, dataset_type: str = "detection"
    ) -> Dict:
        """Validate entire dataset.

        Args:
            dataset_root: Root path to dataset
            dataset_type: Type of dataset (detection, segmentation)

        Returns:
            Dictionary with validation results
        """
        results = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "total_issues": 0,
            "issue_types": {},
            "invalid_files": [],
        }

        if dataset_type == "detection":
            images_dir = dataset_root / "images"
            labels_dir = dataset_root / "labels"

            if not images_dir.exists() or not labels_dir.exists():
                return results

            for img_path in images_dir.glob("*.jpg"):
                results["total_images"] += 1
                label_path = labels_dir / f"{img_path.stem}.txt"

                validation_result = self.validate_detection_annotation(img_path, label_path)

                if validation_result.is_valid:
                    results["valid_images"] += 1
                else:
                    results["invalid_images"] += 1
                    results["invalid_files"].append(str(img_path))

                results["total_issues"] += len(validation_result.issues)

                for issue_type, _ in validation_result.issues:
                    results["issue_types"][issue_type.value] = (
                        results["issue_types"].get(issue_type.value, 0) + 1
                    )

        elif dataset_type == "segmentation":
            images_dir = dataset_root / "images"
            labels_dir = dataset_root / "labels"
            masks_dir = dataset_root / "masks"

            if not images_dir.exists() or not labels_dir.exists():
                return results

            for img_path in images_dir.glob("*.jpg"):
                results["total_images"] += 1
                label_path = labels_dir / f"{img_path.stem}.txt"
                mask_path = masks_dir / f"{img_path.stem}.png"

                validation_result = self.validate_segmentation_annotation(
                    img_path, label_path, mask_path
                )

                if validation_result.is_valid:
                    results["valid_images"] += 1
                else:
                    results["invalid_images"] += 1
                    results["invalid_files"].append(str(img_path))

                results["total_issues"] += len(validation_result.issues)

                for issue_type, _ in validation_result.issues:
                    results["issue_types"][issue_type.value] = (
                        results["issue_types"].get(issue_type.value, 0) + 1
                    )

        return results


def get_annotation_validator() -> AnnotationValidator:
    """Get the global annotation validator.

    Returns:
        AnnotationValidator instance
    """
    return AnnotationValidator()
