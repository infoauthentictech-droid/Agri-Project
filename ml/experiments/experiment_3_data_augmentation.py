"""Experiment 3: Improve Training Data

Provides guidance and tools for:
1. Analyzing current training data distribution
2. Suggesting augmentation strategies for sparse/young/partially hidden weeds
3. Applying realistic augmentations to training data

Note: Collecting new field images with sparse, young, or partially hidden weeds
is a manual process. This script helps with augmentation.

Usage:
    python ml/experiments/experiment_3_data_augmentation.py --analyze
    python ml/experiments/experiment_3_data_augmentation.py --augment --output datasets/augmented
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "data_augmentation"


def analyze_data_distribution(data_path: str) -> Dict:
    """Analyze the distribution of training data.

    Args:
        data_path: Path to data.yaml

    Returns:
        Statistics about data distribution
    """
    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_path).parent
    stats = {"splits": {}}

    for split in ["train", "val", "test"]:
        images_dir = base_path / split / "images"
        labels_dir = base_path / split / "labels"

        if not images_dir.exists():
            continue

        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.png"))

        total_boxes = 0
        weed_boxes = 0
        crop_boxes = 0
        box_sizes = []
        weed_sizes = []

        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        # Calculate area in normalized coordinates
                        area = width * height
                        box_sizes.append(area)
                        total_boxes += 1

                        if class_id == 0:
                            crop_boxes += 1
                        elif class_id == 1:
                            weed_boxes += 1
                            weed_sizes.append(area)

        stats["splits"][split] = {
            "num_images": len(image_files),
            "total_boxes": total_boxes,
            "crop_boxes": crop_boxes,
            "weed_boxes": weed_boxes,
            "avg_box_size": np.mean(box_sizes) if box_sizes else 0,
            "avg_weed_size": np.mean(weed_sizes) if weed_sizes else 0,
            "weed_ratio": weed_boxes / total_boxes if total_boxes > 0 else 0,
        }

    return stats


def suggest_augmentations(stats: Dict) -> List[str]:
    """Suggest augmentation strategies based on data analysis.

    Args:
        stats: Data statistics

    Returns:
        List of augmentation suggestions
    """
    suggestions = []

    train_stats = stats.get("splits", {}).get("train", {})
    avg_weed_size = train_stats.get("avg_weed_size", 0)
    weed_ratio = train_stats.get("weed_ratio", 0)

    # Small weed detection
    if avg_weed_size < 0.05:  # Average weed < 5% of image
        suggestions.append(
            "⚠️  Small weeds detected (avg size: {:.2%}). "
            "Use higher resolution training (832 or 960) or small object augmentation.".format(avg_weed_size)
        )

    # Class imbalance
    if weed_ratio < 0.3:
        suggestions.append(
            "⚠️  Weed class underrepresented ({:.1%}). "
            "Use class weights or oversample weed images.".format(weed_ratio)
        )

    # General suggestions
    suggestions.extend(
        [
            "✓ Apply HSV color augmentation (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4) for lighting variation",
            "✓ Use geometric transforms (degrees=10, translate=0.1, scale=0.5, shear=2.0)",
            "✓ Enable mosaic and mixup for better generalization",
            "✓ Add copy-paste augmentation for rare weed patterns",
            "✓ Consider perspective transform for different camera angles",
            "✓ Use random brightness/contrast for different field conditions",
        ]
    )

    return suggestions


def apply_augmentation(
    image: np.ndarray,
    augment_type: str = "geometric",
) -> np.ndarray:
    """Apply augmentation to an image.

    Args:
        image: Input image
        augment_type: Type of augmentation

    Returns:
        Augmented image
    """
    h, w = image.shape[:2]

    if augment_type == "geometric":
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))

        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip

    elif augment_type == "color":
        # HSV augmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Hue
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180

        # Saturation
        hsv[:, :, 1] = hsv[:, :, 1] * np.random.uniform(0.7, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # Value (brightness)
        hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(0.8, 1.2)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif augment_type == "noise":
        # Add Gaussian noise
        noise = np.random.normal(0, 10, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def augment_dataset(
    data_path: str,
    output_dir: Path,
    augment_factor: int = 2,
    augment_types: List[str] = None,
) -> None:
    """Apply augmentation to training dataset.

    Args:
        data_path: Path to data.yaml
        output_dir: Output directory for augmented data
        augment_factor: How many augmented versions per image
        augment_types: Types of augmentations to apply
    """
    if augment_types is None:
        augment_types = ["geometric", "color", "noise"]

    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_path).parent
    train_images = base_path / "train" / "images"
    train_labels = base_path / "train" / "labels"

    output_images = output_dir / "train" / "images"
    output_labels = output_dir / "train" / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.jpeg")) + sorted(train_images.glob("*.png"))

    print(f"Augmenting {len(image_files)} training images...")

    for img_path in image_files:
        # Copy original
        image = cv2.imread(str(img_path))
        label_path = train_labels / f"{img_path.stem}.txt"

        # Save original
        cv2.imwrite(str(output_images / img_path.name), image)
        if label_path.exists():
            import shutil

            shutil.copy(label_path, output_labels / label_path.name)

        # Create augmented versions
        for i in range(augment_factor):
            augment_type = augment_types[i % len(augment_types)]
            aug_image = apply_augmentation(image, augment_type)

            aug_name = f"{img_path.stem}_aug_{i}_{augment_type}{img_path.suffix}"
            cv2.imwrite(str(output_images / aug_name), aug_image)

            # Copy label (augmentation doesn't change labels for these transforms)
            if label_path.exists():
                aug_label_name = f"{label_path.stem}_aug_{i}_{augment_type}{label_path.suffix}"
                import shutil

                shutil.copy(label_path, output_labels / aug_label_name)

    print(f"Augmented dataset saved to: {output_dir}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 3: Improve Training Data via Augmentation"
    )
    ap.add_argument(
        "--data",
        default=str(DATA_YAML),
        help="Path to data.yaml",
    )
    ap.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze current data distribution",
    )
    ap.add_argument(
        "--augment",
        action="store_true",
        help="Apply augmentation to training data",
    )
    ap.add_argument(
        "--augment-factor",
        type=int,
        default=2,
        help="Number of augmented versions per image",
    )
    args = ap.parse_args(argv)

    print(f"\n{'='*70}")
    print("Experiment 3: Improve Training Data")
    print(f"{'='*70}")
    print(f"Data: {args.data}")
    print(f"{'='*70}\n")

    if args.analyze:
        print("Analyzing data distribution...")
        stats = analyze_data_distribution(args.data)

        print(f"\nData Distribution:")
        for split, split_stats in stats.get("splits", {}).items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {split_stats['num_images']}")
            print(f"  Total boxes: {split_stats['total_boxes']}")
            print(f"  Crop boxes: {split_stats['crop_boxes']}")
            print(f"  Weed boxes: {split_stats['weed_boxes']}")
            print(f"  Avg box size: {split_stats['avg_box_size']:.4f}")
            print(f"  Avg weed size: {split_stats['avg_weed_size']:.4f}")
            print(f"  Weed ratio: {split_stats['weed_ratio']:.2%}")

        print(f"\n{'='*70}")
        print("Augmentation Suggestions:")
        print(f"{'='*70}")
        suggestions = suggest_augmentations(stats)
        for suggestion in suggestions:
            print(suggestion)

        # Save analysis
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        analysis_file = output_path / "data_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump({"statistics": stats, "suggestions": suggestions}, f, indent=2)
        print(f"\nAnalysis saved to: {analysis_file}")

    if args.augment:
        print("Applying augmentation to training data...")
        augment_dataset(args.data, Path(args.output), args.augment_factor)

    if not args.analyze and not args.augment:
        print("Please specify --analyze or --augment")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
