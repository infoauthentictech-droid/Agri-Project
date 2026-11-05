"""Experiment 2: Investigate Missed Weeds

Analyzes false-negative examples, especially small, distant, overlapping,
or partially obscured weeds. Generates statistics and visualizations.

Usage:
    python ml/experiments/experiment_2_missed_weeds.py
    python ml/experiments/experiment_2_missed_weeds.py --model experiments/yolo/agri_yolov8n/weights/best.pt --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "missed_weeds_analysis"


def load_ground_truth(image_path: Path, label_path: Path) -> List[Dict]:
    """Load ground truth annotations from YOLO format.

    Args:
        image_path: Path to image
        label_path: Path to label file

    Returns:
        List of ground truth detections
    """
    if not label_path.exists():
        return []

    gt_detections = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Convert from normalized to absolute coordinates
                img = cv2.imread(str(image_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                gt_detections.append(
                    {
                        "class_id": class_id,
                        "bbox": [x1, y1, x2, y2],
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )

    return gt_detections


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def analyze_missed_detections(
    model: YOLO,
    data_path: str,
    split: str = "val",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "0",
) -> Dict:
    """Analyze missed detections on a dataset split.

    Args:
        model: YOLO model
        data_path: Path to data.yaml
        split: Dataset split
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for matching
        device: Device to use

    Returns:
        Analysis results
    """
    import yaml

    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    base_path = Path(data_path).parent
    images_dir = base_path / split / "images"
    labels_dir = base_path / split / "labels"

    print(f"  Analyzing {split} split...")
    print(f"  Images directory: {images_dir}")

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.png"))
    print(f"  Found {len(image_files)} images")

    stats = {
        "total_images": len(image_files),
        "total_gt_weeds": 0,
        "total_detected_weeds": 0,
        "missed_weeds": 0,
        "missed_by_size": {"small": 0, "medium": 0, "large": 0},
        "size_distribution": {"small": 0, "medium": 0, "large": 0},
        "missed_examples": [],
    }

    for img_path in image_files[:100]:  # Limit to 100 images for speed
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Load ground truth
        gt_detections = load_ground_truth(img_path, label_path)
        weed_gt = [d for d in gt_detections if d["class_id"] == 1]  # class 1 = weed

        if not weed_gt:
            continue

        stats["total_gt_weeds"] += len(weed_gt)

        # Run inference
        results = model(str(img_path), conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)

        if not results or len(results) == 0:
            # All weeds missed
            stats["missed_weeds"] += len(weed_gt)
            for weed in weed_gt:
                size_category = categorize_size(weed["area"])
                stats["missed_by_size"][size_category] += 1
                stats["size_distribution"][size_category] += 1
            continue

        # Get predicted detections
        pred_boxes = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    if int(cls) == 1:  # weed
                        pred_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        stats["total_detected_weeds"] += len(pred_boxes)

        # Match predictions to ground truth
        matched_gt = set()
        for gt_idx, gt_box in enumerate(weed_gt):
            for pred_box in pred_boxes:
                iou = calculate_iou(gt_box["bbox"], pred_box)
                if iou >= iou_threshold:
                    matched_gt.add(gt_idx)
                    break

        # Count missed weeds
        for gt_idx, gt_box in enumerate(weed_gt):
            if gt_idx not in matched_gt:
                stats["missed_weeds"] += 1
                size_category = categorize_size(gt_box["area"])
                stats["missed_by_size"][size_category] += 1
                stats["size_distribution"][size_category] += 1

                # Save example (first 20 missed weeds)
                if len(stats["missed_examples"]) < 20:
                    stats["missed_examples"].append(
                        {
                            "image": str(img_path.name),
                            "bbox": gt_box["bbox"],
                            "area": gt_box["area"],
                            "size_category": size_category,
                        }
                    )

    # Calculate recall by size
    stats["recall_by_size"] = {}
    for size in ["small", "medium", "large"]:
        total = stats["size_distribution"][size]
        missed = stats["missed_by_size"][size]
        detected = total - missed
        stats["recall_by_size"][size] = {
            "total": total,
            "detected": detected,
            "missed": missed,
            "recall": detected / total if total > 0 else 0.0,
        }

    return stats


def categorize_size(area: int) -> str:
    """Categorize bounding box area into size categories."""
    if area < 32 * 32:  # Very small
        return "small"
    elif area < 96 * 96:  # Medium
        return "medium"
    else:  # Large
        return "large"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 2: Investigate Missed Weeds"
    )
    ap.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to YOLO model weights",
    )
    ap.add_argument(
        "--data",
        default=str(DATA_YAML),
        help="Path to data.yaml",
    )
    ap.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to analyze",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for matching",
    )
    ap.add_argument(
        "--device",
        default="0",
        help="GPU device or 'cpu'",
    )
    args = ap.parse_args(argv)

    # Check GPU availability
    import torch

    if args.device != "cpu":
        device_idx = int(args.device)
        if torch.cuda.is_available():
            if device_idx >= torch.cuda.device_count():
                print(f"Error: GPU {device_idx} not available")
                return 1
            print(f"Using GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
        else:
            print("CUDA not available, falling back to CPU")
            args.device = "cpu"

    print(f"\n{'='*70}")
    print("Experiment 2: Investigate Missed Weeds")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load model
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return 1

    model = YOLO(args.model)

    # Convert path to forward slashes for YOLO compatibility
    data_path = str(Path(args.data).as_posix())

    # Analyze missed detections
    stats = analyze_missed_detections(model, data_path, args.split, args.conf, args.iou, args.device)

    # Print results
    print(f"\n{'='*70}")
    print("Analysis Results")
    print(f"{'='*70}")
    print(f"Total images analyzed: {stats['total_images']}")
    print(f"Total ground truth weeds: {stats['total_gt_weeds']}")
    print(f"Total detected weeds: {stats['total_detected_weeds']}")
    print(f"Missed weeds: {stats['missed_weeds']}")
    print(f"Overall recall: {(stats['total_detected_weeds'] / stats['total_gt_weeds'] * 100):.2f}%")

    print(f"\nRecall by size:")
    print(f"{'Size':<10} {'Total':<10} {'Detected':<10} {'Missed':<10} {'Recall':<10}")
    print("-" * 50)
    for size in ["small", "medium", "large"]:
        recall = stats["recall_by_size"][size]
        print(
            f"{size:<10} {recall['total']:<10} {recall['detected']:<10} "
            f"{recall['missed']:<10} {recall['recall']*100:<10.2f}%"
        )

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"missed_weeds_{args.split}.json"
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "statistics": stats,
    }

    results_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
