"""YOLOv8-Seg / YOLO11-Seg training script for semantic and instance segmentation.

YOLO-Seg combines both semantic and instance segmentation into a single,
high-speed, pixel-precise mask engine. This is the industry standard for
real-time segmentation on agricultural edge-computing devices.

Dataset Types:
- primary: Original field images
- primary_augmented: Augmented versions of primary dataset
- secondary: External public datasets
- hybrid: Combined dataset (default for training)

Usage:
    python ml/segmentation/train_yoloseg.py --epochs 100 --batch 16 --device 0 --dataset_type hybrid
    python ml/segmentation/train_yoloseg.py --model yolo11n-seg.pt --dataset_type primary_augmented
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ultralytics import YOLO
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "hybrid" / "segmentation" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "yoloseg" / "agri_yolov8n_seg"


def validate_yaml(yaml_path: Path) -> tuple[bool, str, Optional[Dict]]:
    """Validate YOLO data.yaml file"""
    if not yaml_path.exists():
        return False, f"YAML file not found: {yaml_path}", None

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Check required fields
        required_fields = ['path', 'train', 'val', 'names']
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}", data

        # Check train/val/test paths exist
        base_path = Path(data['path'])
        for split in ['train', 'val']:
            split_path = base_path / data[split]
            if not split_path.exists():
                return False, f"Split directory not found: {split_path}", data

        # Check that test split exists if specified
        if 'test' in data:
            test_path = base_path / data['test']
            if not test_path.exists():
                return False, f"Test directory not found: {test_path}", data

        # Validate class names
        if not isinstance(data['names'], dict):
            return False, "Names must be a dictionary", data

        if len(data['names']) != 2:
            return False, f"Expected 2 classes, found {len(data['names'])}", data

        return True, "Valid", data

    except Exception as e:
        return False, f"YAML parse error: {e}", None


def validate_segmentation_annotations(yaml_data: Dict) -> tuple[int, int, list]:
    """Validate that images have corresponding segmentation annotations"""
    base_path = Path(yaml_data['path'])
    validation_errors = []

    total_images = 0
    total_labels = 0

    for split in ['train', 'val', 'test']:
        if split not in yaml_data:
            continue

        images_dir = base_path / yaml_data[split]
        labels_dir = base_path / 'labels' / split

        if not images_dir.exists():
            continue

        # Count images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        total_images += len(image_files)

        # Check for corresponding labels
        if labels_dir.exists():
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    total_labels += 1
                    # Validate segmentation label format (polygons)
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            # Segmentation requires at least 5 values (class + 2 pairs of coordinates)
                            if len(parts) < 5:
                                validation_errors.append(f"{split}/{img_file.name}: Line {line_num} - Segmentation requires at least 5 values (class + polygon coordinates)")
                            elif (len(parts) - 1) % 2 != 0:
                                validation_errors.append(f"{split}/{img_file.name}: Line {line_num} - Segmentation requires even number of coordinates")
                            else:
                                class_id = int(parts[0])
                                if class_id not in yaml_data['names'].keys():
                                    validation_errors.append(f"{split}/{img_file.name}: Line {line_num} - Invalid class ID {class_id}")
                    except (OSError, ValueError) as e:
                        validation_errors.append(f"{split}/{img_file.name}: Could not read label - {e}")
                else:
                    validation_errors.append(f"{split}/{img_file.name}: Missing label file")

    return total_images, total_labels, validation_errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Train YOLO-Seg segmentation model for groundnut vs weed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--device", default="0", help="GPU device (0, 1, ...) or 'cpu'")
    ap.add_argument("--model", default="yolov8n-seg.pt",
                    help="YOLO-Seg model variant (yolov8n-seg.pt, yolo11n-seg.pt)")
    ap.add_argument("--data", default=str(DATA_YAML), help="Path to data.yaml")
    ap.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--optimizer", default="AdamW", help="Optimizer")
    ap.add_argument("--dataset-type", type=str, default="hybrid",
                    choices=["primary", "primary_augmented", "secondary", "hybrid"],
                    help="Dataset type to use")
    ap.add_argument("--skip-validation", action="store_true", help="Skip annotation validation")
    ap.add_argument("--resume", action="store_true", help="Continue from the last checkpoint in --output, if there is one")
    args = ap.parse_args(argv)

    # Validate hyperparameters
    if args.epochs <= 0:
        print(f"Error: epochs must be positive, got {args.epochs}")
        return 1
    if args.batch <= 0:
        print(f"Error: batch_size must be positive, got {args.batch}")
        return 1
    if args.imgsz <= 0:
        print(f"Error: image_size must be positive, got {args.imgsz}")
        return 1
    if args.lr0 <= 0:
        print(f"Error: learning_rate must be positive, got {args.lr0}")
        return 1
    if args.patience < 0:
        print(f"Error: patience must be non-negative, got {args.patience}")
        return 1

    # Validate model is a segmentation model
    if 'seg' not in args.model.lower():
        print(f"WARNING: '{args.model}' may not be a segmentation model. Ensure it supports segmentation.")

    # Update data.yaml path based on dataset type
    data_yaml_path = PROJECT_ROOT / "datasets" / args.dataset_type / "segmentation" / "data.yaml"
    args.data = str(data_yaml_path)

    print(f"Using dataset: {args.dataset_type}")
    print(f"Data YAML: {args.data}")

    # Validate YAML file
    is_valid, msg, yaml_data = validate_yaml(data_yaml_path)
    if not is_valid:
        print(f"Error: YAML validation failed: {msg}")
        return 1
    print(f"YAML validation: {msg}")

    # Validate segmentation annotations
    if not args.skip_validation:
        print("\nValidating segmentation annotations...")
        total_images, total_labels, validation_errors = validate_segmentation_annotations(yaml_data)
        print(f"Total images: {total_images}")
        print(f"Total labels: {total_labels}")
        print(f"Images without labels: {total_images - total_labels}")
        print(f"Validation errors: {len(validation_errors)}")

        if validation_errors:
            print("\nFirst 10 validation errors:")
            for err in validation_errors[:10]:
                print(f"  - {err}")

            if len(validation_errors) > 10:
                print(f"  ... and {len(validation_errors) - 10} more errors")

            # Ask user if they want to continue
            response = input("\nContinue training despite validation errors? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled by user")
                return 1

    # GPU only: never fall back to the CPU
    try:
        device_idx = int(args.device)
    except ValueError:
        print(f"Invalid device: {args.device}. Training is GPU-only, use a device number (0, 1, ...)")
        return 1
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Training is GPU-only.")
        return 1
    if device_idx >= torch.cuda.device_count():
        print(f"Error: GPU {device_idx} not available. Available GPUs: {torch.cuda.device_count()}")
        return 1
    print(f"Using GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device_idx).total_memory / 1024**3:.2f} GB")

    print(f"\n{'='*60}")
    print("YOLO-Seg Segmentation Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Learning rate: {args.lr0}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Early stopping patience: {args.patience}")
    print(f"{'='*60}\n")

    # Load model, or the last checkpoint when continuing an interrupted run
    last_checkpoint = Path(args.output).resolve() / "weights" / "last.pt"
    resuming = args.resume and last_checkpoint.exists()
    if resuming:
        print(f"Continuing from {last_checkpoint}")
    try:
        model = YOLO(str(last_checkpoint) if resuming else args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Training arguments
    data_path = str(Path(args.data).as_posix())
    output_dir = Path(args.output).resolve()

    train_args = {
        "data": data_path,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "patience": args.patience,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "verbose": True,
        "save": True,
        "plots": True,
        "val": True,
        "augment": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cos_lr": True,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
    }

    # Train
    print("Starting training...")
    try:
        results = model.train(resume=True) if resuming else model.train(exist_ok=True, **train_args)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Validate on best checkpoint
    print("\nValidating model on best checkpoint...")
    try:
        metrics = model.val()
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1

    # Get actual epochs trained
    actual_epochs = getattr(results, 'epochs', args.epochs)

    # CRITICAL FIX: Distinguish between box and mask metrics
    # Only report mask metrics if they are actually available
    has_mask_metrics = hasattr(metrics, 'seg') and metrics.seg is not None

    if has_mask_metrics:
        box_map50 = float(metrics.box.map50)
        box_map = float(metrics.box.map)
        mask_map50 = float(metrics.seg.map50)
        mask_map = float(metrics.seg.map)
    else:
        box_map50 = float(metrics.box.map50)
        box_map = float(metrics.box.map)
        mask_map50 = None
        mask_map = None
        print("WARNING: Mask metrics not available. Only box detection metrics are reported.")
        print("This may indicate the model is not a segmentation model or segmentation labels are invalid.")

    # Save training summary
    output_path = output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "dataset_type": args.dataset_type,
        "data_yaml": str(args.data),
        "epochs_requested": args.epochs,
        "epochs_trained": actual_epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "device": args.device,
        "learning_rate": args.lr0,
        "optimizer": args.optimizer,
        "early_stopping_patience": args.patience,
        "has_mask_metrics": has_mask_metrics,
        "final_metrics": {
            "box_mAP50": box_map50,
            "box_mAP50-95": box_map,
            "mask_mAP50": mask_map50,
            "mask_mAP50-95": mask_map,
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "validation_errors_count": len(validation_errors) if not args.skip_validation else 0,
    }

    summary_path = output_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Epochs requested: {args.epochs}")
    print(f"Epochs trained: {actual_epochs}")
    print(f"Box mAP50: {box_map50:.4f}")
    print(f"Box mAP50-95: {box_map:.4f}")
    if has_mask_metrics:
        print(f"Mask mAP50: {mask_map50:.4f}")
        print(f"Mask mAP50-95: {mask_map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    # Evaluate on test set
    print(f"{'='*60}")
    print("Evaluating on TEST set...")
    print(f"{'='*60}")
    try:
        test_metrics = model.val(
            data=data_path,
            split='test',
            device=args.device,
            verbose=True
        )

        has_test_mask = hasattr(test_metrics, 'seg') and test_metrics.seg is not None

        if has_test_mask:
            test_box_map50 = float(test_metrics.box.map50)
            test_box_map = float(test_metrics.box.map)
            test_mask_map50 = float(test_metrics.seg.map50)
            test_mask_map = float(test_metrics.seg.map)
        else:
            test_box_map50 = float(test_metrics.box.map50)
            test_box_map = float(test_metrics.box.map)
            test_mask_map50 = None
            test_mask_map = None
            print("WARNING: Test mask metrics not available.")

        test_results = {
            "test_box_mAP50": test_box_map50,
            "test_box_mAP50-95": test_box_map,
            "test_mask_mAP50": test_mask_map50,
            "test_mask_mAP50-95": test_mask_map,
            "test_precision": float(test_metrics.box.mp),
            "test_recall": float(test_metrics.box.mr),
            "has_mask_metrics": has_test_mask,
        }

        print(f"\nTEST SET RESULTS:")
        print(f"  Box mAP50: {test_results['test_box_mAP50']:.4f}")
        print(f"  Box mAP50-95: {test_results['test_box_mAP50-95']:.4f}")
        if has_test_mask:
            print(f"  Mask mAP50: {test_results['test_mask_mAP50']:.4f}")
            print(f"  Mask mAP50-95: {test_results['test_mask_mAP50-95']:.4f}")
        print(f"  Precision: {test_results['test_precision']:.4f}")
        print(f"  Recall: {test_results['test_recall']:.4f}")

        # Add test results to summary
        summary['test_metrics'] = test_results
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    except Exception as e:
        print(f"Warning: Could not evaluate on test set: {e}")
        summary['test_metrics'] = None
        summary['test_error'] = str(e)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # After a GPU error a normal exit can hang forever and stall the training queue
