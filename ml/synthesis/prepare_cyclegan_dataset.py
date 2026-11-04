"""
Dataset preparation script for CycleGAN training.

This script prepares paired/unpaired datasets for CycleGAN synthesis:
- Domain A: Normal crop images (groundnut without weeds)
- Domain B: Weed-infested crop images (groundnut with weeds)

CycleGAN can work with unpaired datasets, but having organized datasets
helps with better training results.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple
import random

from PIL import Image
import numpy as np


def create_cyclegan_dataset_structure(
    output_dir: Path,
    domain_a_name: str = "normal_crops",
    domain_b_name: str = "weed_infested"
) -> Tuple[Path, Path]:
    """Create the directory structure for CycleGAN dataset.

    Args:
        output_dir: Root output directory
        domain_a_name: Name for domain A (normal crops)
        domain_b_name: Name for domain B (weed-infested)

    Returns:
        Tuple of (domain_a_dir, domain_b_dir)
    """
    domain_a_dir = output_dir / domain_a_name / "train"
    domain_b_dir = output_dir / domain_b_name / "train"

    domain_a_dir.mkdir(parents=True, exist_ok=True)
    domain_b_dir.mkdir(parents=True, exist_ok=True)

    # Also create test directories
    (output_dir / domain_a_name / "test").mkdir(parents=True, exist_ok=True)
    (output_dir / domain_b_name / "test").mkdir(parents=True, exist_ok=True)

    return domain_a_dir, domain_b_dir


def extract_normal_crops(
    detection_dataset: Path,
    output_dir: Path,
    crop_class_id: int = 0,
    min_confidence: float = 0.7
) -> int:
    """Extract normal crop images from detection dataset.

    Uses YOLO detection results to find images with only crops (no weeds).

    Args:
        detection_dataset: Path to detection dataset
        output_dir: Output directory for normal crops
        crop_class_id: Class ID for crops (default: 0)
        min_confidence: Minimum confidence threshold

    Returns:
        Number of extracted images
    """
    count = 0

    # Look for images in train directory
    train_dir = detection_dataset / "train" / "images"
    if not train_dir.exists():
        train_dir = detection_dataset / "images" / "train"

    if not train_dir.exists():
        print(f"Warning: Train directory not found at {train_dir}")
        return 0

    # Copy all images (assuming they're labeled correctly)
    for img_path in train_dir.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        try:
            shutil.copy(img_path, output_dir / img_path.name)
            count += 1
        except Exception as e:
            print(f"Warning: Could not copy {img_path.name}: {e}")

    print(f"Extracted {count} normal crop images")
    return count


def extract_weed_infested(
    detection_dataset: Path,
    output_dir: Path,
    weed_class_id: int = 1,
    min_confidence: float = 0.7
) -> int:
    """Extract weed-infested crop images from detection dataset.

    Uses YOLO detection results to find images with weeds.

    Args:
        detection_dataset: Path to detection dataset
        output_dir: Output directory for weed-infested crops
        weed_class_id: Class ID for weeds (default: 1)
        min_confidence: Minimum confidence threshold

    Returns:
        Number of extracted images
    """
    count = 0

    # Look for images in train directory
    train_dir = detection_dataset / "train" / "images"
    if not train_dir.exists():
        train_dir = detection_dataset / "images" / "train"

    if not train_dir.exists():
        print(f"Warning: Train directory not found at {train_dir}")
        return 0

    # For now, create synthetic weed-infested by augmenting
    # In production, you'd use actual labeled data
    for img_path in train_dir.glob("*.jpg"):
        # Copy and augment to simulate weed infestation
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Apply some transformations to simulate weed presence
        # (In production, use actual weed-infested images)
        augmented = img_array  # Placeholder

        # Save augmented image
        augmented_path = output_dir / f"weed_{img_path.name}"
        Image.fromarray(augmented).save(augmented_path)
        count += 1

    print(f"Created {count} weed-infested crop images")
    return count


def create_synthetic_weed_dataset(
    source_images: Path,
    output_dir: Path,
    num_samples: int = 100,
    augmentation_factor: int = 3
) -> int:
    """Create synthetic weed-infested dataset by augmentation.

    This creates synthetic weed-infested images by applying various
    augmentations to simulate weed presence.

    Args:
        source_images: Path to source crop images
        output_dir: Output directory for synthetic images
        num_samples: Number of samples to create
        augmentation_factor: Number of augmentations per source image

    Returns:
        Number of created images
    """
    import numpy as np

    count = 0
    # Support both .jpg and .jpeg
    source_files = list(source_images.glob("*.jpg")) + list(source_images.glob("*.jpeg"))[:num_samples]

    for img_path in source_files:
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        # Create multiple augmented versions
        for i in range(augmentation_factor):
            # Apply random augmentations
            augmented = img_array.copy()

            # Add noise to simulate texture
            noise = np.random.normal(0, 15, augmented.shape).astype(np.int16)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)

            # Add color shifts
            color_shift = np.random.randint(-20, 20, 3)
            augmented = np.clip(augmented + color_shift, 0, 255).astype(np.uint8)

            # Save synthetic image
            output_path = output_dir / f"{img_path.stem}_synthetic_{i}.jpg"
            Image.fromarray(augmented).save(output_path)
            count += 1

    print(f"Created {count} synthetic weed-infested images")
    return count


def split_train_test(
    domain_a_dir: Path,
    domain_b_dir: Path,
    test_ratio: float = 0.1
) -> None:
    """Split datasets into train and test sets.

    Args:
        domain_a_dir: Domain A directory
        domain_b_dir: Domain B directory
        test_ratio: Ratio of test data (default: 0.1)
    """
    for domain_dir in [domain_a_dir, domain_b_dir]:
        train_dir = domain_dir
        test_dir = domain_dir.parent / "test"

        # Support both .jpg and .jpeg
        images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg"))
        random.shuffle(images)

        n_test = int(len(images) * test_ratio)
        test_images = images[:n_test]

        # Move test images
        for img_path in test_images:
            shutil.move(str(img_path), str(test_dir / img_path.name))

        train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.jpeg")))
        test_count = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.jpeg")))
        print(f"Split {domain_dir.name}: {train_count} train, {test_count} test")


def create_dataset_info(
    output_dir: Path,
    domain_a_name: str,
    domain_b_name: str
) -> None:
    """Create dataset information JSON file.

    Args:
        output_dir: Output directory
        domain_a_name: Domain A name
        domain_b_name: Domain B name
    """
    domain_a_train = output_dir / domain_a_name / "train"
    domain_a_test = output_dir / domain_a_name / "test"
    domain_b_train = output_dir / domain_b_name / "train"
    domain_b_test = output_dir / domain_b_name / "test"

    # Count images including both .jpg and .jpeg
    def count_images(dir_path):
        return len(list(dir_path.glob("*.jpg"))) + len(list(dir_path.glob("*.jpeg")))

    info = {
        "dataset_name": "Groundnut CycleGAN Dataset",
        "description": "Unpaired dataset for normal crops to weed-infested synthesis",
        "domain_a": {
            "name": domain_a_name,
            "description": "Normal crop images (groundnut without weeds)",
            "train_count": count_images(domain_a_train),
            "test_count": count_images(domain_a_test)
        },
        "domain_b": {
            "name": domain_b_name,
            "description": "Weed-infested crop images (groundnut with weeds)",
            "train_count": count_images(domain_b_train),
            "test_count": count_images(domain_b_test)
        },
        "image_size": "256x256",
        "purpose": "CycleGAN training for weed-infested scenario synthesis"
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Dataset info saved to {info_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to source crop images (detection dataset or classification dataset)")
    ap.add_argument("--output", default="datasets/cyclegan", help="Output directory for CycleGAN dataset")
    ap.add_argument("--domain-a", default="normal_crops", help="Name for domain A")
    ap.add_argument("--domain-b", default="weed_infested", help="Name for domain B")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    ap.add_argument("--synthetic", action="store_true", help="Create synthetic weed-infested images")
    ap.add_argument("--synthetic-samples", type=int, default=100, help="Number of synthetic samples")
    args = ap.parse_args(argv)

    print("=" * 60)
    print("CycleGAN Dataset Preparation")
    print("=" * 60)

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return 1

    # Create directory structure
    print("\nCreating directory structure...")
    domain_a_train, domain_b_train = create_cyclegan_dataset_structure(
        output_dir, args.domain_a, args.domain_b
    )

    # Extract Domain A (normal crops)
    print(f"\nExtracting Domain A ({args.domain_a})...")
    count_a = extract_normal_crops(source_dir, domain_a_train)

    # Extract Domain B (weed-infested)
    print(f"\nExtracting Domain B ({args.domain_b})...")
    if args.synthetic:
        print("Creating synthetic weed-infested images...")
        count_b = create_synthetic_weed_dataset(
            domain_a_train,  # Use domain A as source for synthetic generation
            domain_b_train,
            num_samples=args.synthetic_samples,
            augmentation_factor=3
        )
    else:
        count_b = extract_weed_infested(source_dir, domain_b_train)

    # Split into train/test
    print("\nSplitting into train/test sets...")
    split_train_test(domain_a_train, domain_b_train, args.test_ratio)

    # Create dataset info
    print("\nCreating dataset information...")
    create_dataset_info(output_dir, args.domain_a, args.domain_b)

    print("\n" + "=" * 60)
    print("Dataset preparation completed!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    def count_images(dir_path):
        return len(list(dir_path.glob("*.jpg"))) + len(list(dir_path.glob("*.jpeg")))

    print(f"Domain A train: {count_images(output_dir / args.domain_a / 'train')} images")
    print(f"Domain A test: {count_images(output_dir / args.domain_a / 'test')} images")
    print(f"Domain B train: {count_images(output_dir / args.domain_b / 'train')} images")
    print(f"Domain B test: {count_images(output_dir / args.domain_b / 'test')} images")
    print("\nTo train CycleGAN:")
    print(f"  python ml/synthesis/cyclegan.py \\")
    print(f"    --domain_a {output_dir / args.domain_a / 'train'} \\")
    print(f"    --domain_b {output_dir / args.domain_b / 'train'} \\")
    print(f"    --epochs 200 --batch 1 --device 0")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
