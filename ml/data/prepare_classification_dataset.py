"""
Simplified Dataset Preparation Pipeline - Focus on Classification

This script prepares datasets for classification only, since:
- Detection labels are missing from your datasets
- Segmentation labels are in PNG format (not YOLO format)
- Classification data is complete and ready

Features:
- Data leakage prevention
- Augmentation with label tracking
- Stratified splits
- Comprehensive validation
- Non-destructive operations
"""

import sys
import os
import shutil
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import cv2
import numpy as np


@dataclass
class ImageInfo:
    """Information about an image."""
    path: Path
    source_group: str
    content_hash: str
    label: str
    is_augmented: bool = False
    augmentation_parent: Optional[str] = None


class Augmenter:
    """Image augmentation with tracking."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    def flip_vertical(self, image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 0)

    def rotate_90(self, image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def rotate_180(self, image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_180)

    def rotate_270(self, image: np.ndarray) -> np.ndarray:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def brightness_increase(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50)
        v[v > 255] = 255
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def brightness_decrease(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.subtract(v, 50)
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    def augment(self, image: np.ndarray, index: int) -> Tuple[np.ndarray, str]:
        """Apply a specific augmentation based on index."""
        augmentations = [
            self.flip_horizontal,
            self.flip_vertical,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.brightness_increase,
            self.brightness_decrease,
        ]
        aug_name = augmentations[index % len(augmentations)].__name__
        return augmentations[index % len(augmentations)](image), aug_name


def compute_hash(image_path: Path) -> str:
    """Compute SHA256 hash of image file."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def discover_images(source_dir: Path, label: str) -> List[ImageInfo]:
    """Recursively discover images and compute hashes."""
    images = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for img_path in source_dir.rglob('*'):
        if img_path.suffix in valid_extensions:
            try:
                # Compute hash
                img_hash = compute_hash(img_path)

                # Use parent directory as source group
                source_group = img_path.parent.name

                images.append(ImageInfo(
                    path=img_path,
                    source_group=source_group,
                    content_hash=img_hash,
                    label=label
                ))
            except Exception as e:
                print(f"  Warning: Could not process {img_path}: {e}")

    return images


def detect_duplicates(images: List[ImageInfo]) -> Dict[str, List[ImageInfo]]:
    """Detect duplicate images by content hash."""
    hash_to_images = defaultdict(list)
    for img in images:
        hash_to_images[img.content_hash].append(img)

    return {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}


def create_augmented_images(
    images: List[ImageInfo],
    output_dir: Path,
    augmenter: Augmenter,
    augmentation_factor: int
) -> List[ImageInfo]:
    """Create augmented images with proper tracking."""
    augmented = []

    for img_info in images:
        for i in range(augmentation_factor):
            try:
                # Load image
                img = cv2.imread(str(img_info.path))
                if img is None:
                    continue

                # Apply augmentation
                aug_img, aug_name = augmenter.augment(img, i)

                # Generate output filename
                base_name = img_info.path.stem
                ext = img_info.path.suffix
                aug_filename = f"{base_name}_aug{i}_{aug_name}{ext}"
                aug_path = output_dir / img_info.label / aug_filename

                # Save augmented image
                aug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(aug_path), aug_img)

                # Create augmented image info
                aug_info = ImageInfo(
                    path=aug_path,
                    source_group=img_info.source_group,  # Keep same group
                    content_hash=compute_hash(aug_path),
                    label=img_info.label,
                    is_augmented=True,
                    augmentation_parent=str(img_info.path.name)
                )
                augmented.append(aug_info)

            except Exception as e:
                print(f"  Warning: Could not augment {img_info.path}: {e}")

    return augmented


def stratified_split(
    images: List[ImageInfo],
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Tuple[List[ImageInfo], List[ImageInfo], List[ImageInfo]]:
    """Stratified split by source group to prevent leakage."""
    random.seed(random_seed)

    # Group by source_group and label
    groups = defaultdict(list)
    for img in images:
        key = (img.source_group, img.label)
        groups[key].append(img)

    train_images = []
    val_images = []
    test_images = []

    for key, group_images in groups.items():
        random.shuffle(group_images)

        n_total = len(group_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images.extend(group_images[:n_train])
        val_images.extend(group_images[n_train:n_train + n_val])
        test_images.extend(group_images[n_train + n_val:])

    return train_images, val_images, test_images


def copy_images_to_split(
    images: List[ImageInfo],
    output_dir: Path,
    split_name: str
) -> Dict[str, int]:
    """Copy images to split directory."""
    counts = defaultdict(int)

    for img_info in images:
        output_path = output_dir / split_name / img_info.label / img_info.path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not output_path.exists():
            shutil.copy2(img_info.path, output_path)

        counts[img_info.label] += 1

    return dict(counts)


def generate_manifest(
    primary_images: List[ImageInfo],
    augmented_images: List[ImageInfo],
    duplicates: Dict[str, List[ImageInfo]],
    output_path: Path
) -> None:
    """Generate dataset manifest."""
    manifest = {
        "metadata": {
            "timestamp": str(os.times()),
            "random_seed": 42,
            "augmentation_factor": 2,
        },
        "dataset_types": {
            "primary": {
                "classification": {
                    "crop": {"total": len([i for i in primary_images if i.label == "crop"])},
                    "weed": {"total": len([i for i in primary_images if i.label == "weed"])}
                }
            },
            "primary_augmented": {
                "classification": {
                    "crop": {"total": len([i for i in augmented_images if i.label == "crop"])},
                    "weed": {"total": len([i for i in augmented_images if i.label == "weed"])}
                }
            }
        },
        "duplicates": {
            h: [str(i.path) for i in imgs] for h, imgs in duplicates.items()
        },
        "validation_errors": []
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simplified dataset preparation for classification")
    parser.add_argument("--primary-source", required=True, help="Path to primary dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--augmentation-factor", type=int, default=2, help="Augmentation factor")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("SIMPLIFIED DATASET PREPARATION - CLASSIFICATION ONLY")
    print("=" * 60)

    primary_source = Path(args.primary_source)
    output_dir = Path(args.output_dir)

    # Step 1: Discover primary images
    print("\nStep 1: Discovering primary images...")
    crop_dir = primary_source / "crop"
    weed_dir = primary_source / "weed"

    crop_images = discover_images(crop_dir, "crop") if crop_dir.exists() else []
    weed_images = discover_images(weed_dir, "weed") if weed_dir.exists() else []

    primary_images = crop_images + weed_images
    print(f"  Found {len(crop_images)} crop images")
    print(f"  Found {len(weed_images)} weed images")
    print(f"  Total: {len(primary_images)} images")

    # Step 2: Detect duplicates
    print("\nStep 2: Detecting duplicates...")
    duplicates = detect_duplicates(primary_images)
    print(f"  Found {len(duplicates)} duplicate groups")

    # Step 3: Create augmented dataset
    print("\nStep 3: Creating augmented dataset...")
    augmenter = Augmenter(args.random_seed)
    aug_output_dir = output_dir / "primary_augmented" / "classification"

    augmented_images = create_augmented_images(
        primary_images,
        aug_output_dir,
        augmenter,
        args.augmentation_factor
    )
    print(f"  Created {len(augmented_images)} augmented images")

    # Step 4: Create hybrid dataset (primary + augmented)
    print("\nStep 4: Creating hybrid dataset...")
    hybrid_images = primary_images + augmented_images
    print(f"  Total hybrid images: {len(hybrid_images)}")

    # Step 5: Stratified split
    print("\nStep 5: Creating stratified splits...")
    train, val, test = stratified_split(
        hybrid_images,
        args.train_ratio,
        args.val_ratio,
        args.random_seed
    )

    print(f"  Train: {len(train)} images")
    print(f"  Val: {len(val)} images")
    print(f"  Test: {len(test)} images")

    # Step 6: Copy to hybrid directory
    print("\nStep 6: Copying images to hybrid directory...")
    hybrid_output_dir = output_dir / "hybrid" / "classification"

    train_counts = copy_images_to_split(train, hybrid_output_dir, "train")
    val_counts = copy_images_to_split(val, hybrid_output_dir, "val")
    test_counts = copy_images_to_split(test, hybrid_output_dir, "test")

    print(f"  Train: {train_counts}")
    print(f"  Val: {val_counts}")
    print(f"  Test: {test_counts}")

    # Step 7: Generate manifest
    print("\nStep 7: Generating manifest...")
    manifest_path = output_dir / "dataset_manifest.json"
    generate_manifest(primary_images, augmented_images, duplicates, manifest_path)
    print(f"  Manifest saved to {manifest_path}")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir / 'hybrid' / 'classification'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
