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
import sys
from pathlib import Path

from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "hybrid" / "segmentation" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "yoloseg" / "agri_yolov8n_seg"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="GPU device (0, 1, ...) or 'cpu'")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="YOLO-Seg model variant (yolov8n-seg.pt, yolo11n-seg.pt)")
    ap.add_argument("--data", default=str(DATA_YAML), help="Path to data.yaml")
    ap.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    ap.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--optimizer", default="AdamW", help="Optimizer")
    ap.add_argument("--dataset_type", type=str, default="hybrid", choices=["primary", "primary_augmented", "secondary", "hybrid"], help="Dataset type to use")
    args = ap.parse_args(argv)

    # Update data.yaml path based on dataset type
    args.data = str(PROJECT_ROOT / "datasets" / args.dataset_type / "segmentation" / "data.yaml")
    print(f"Using dataset: {args.dataset_type}")
    print(f"Data YAML: {args.data}")

    # Check GPU availability
    if args.device != "cpu":
        device_idx = int(args.device)
        if torch.cuda.is_available():
            if device_idx >= torch.cuda.device_count():
                print(f"Error: GPU {device_idx} not available. Available GPUs: {torch.cuda.device_count()}")
                return 1
            print(f"Using GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
        else:
            print("CUDA not available, falling back to CPU")
            args.device = "cpu"

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

    # Load YOLO-Seg model
    model = YOLO(args.model)

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
        # Augmentation settings
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
        # Learning rate scheduler
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
    results = model.train(**train_args)

    # Validate
    print("\nValidating model...")
    metrics = model.val()

    # Save training summary
    output_path = output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "epochs_trained": args.epochs,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "device": args.device,
        "final_metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "mask_mAP50": float(getattr(metrics, 'mask', metrics.box).map50) if hasattr(metrics, 'mask') else float(metrics.box.map50),
            "mask_mAP50-95": float(getattr(metrics, 'mask', metrics.box).map) if hasattr(metrics, 'mask') else float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
    }

    summary_path = output_path / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Final mAP50: {metrics.box.map50:.4f}")
    print(f"Final mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, 'mask'):
        print(f"Final Mask mAP50: {metrics.mask.map50:.4f}")
        print(f"Final Mask mAP50-95: {metrics.mask.map:.4f}")
    print(f"Final Precision: {metrics.box.mp:.4f}")
    print(f"Final Recall: {metrics.box.mr:.4f}")
    print(f"Model saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}\n")

    # Evaluate on test set
    print(f"{'='*60}")
    print("Evaluating on TEST set...")
    print(f"{'='*60}")
    test_metrics = model.val(
        data=data_path,
        split='test',
        device=args.device,
        verbose=True
    )

    test_results = {
        "test_mAP50": float(test_metrics.box.map50),
        "test_mAP50-95": float(test_metrics.box.map),
        "test_mask_mAP50": float(getattr(test_metrics, 'mask', test_metrics.box).map50) if hasattr(test_metrics, 'mask') else float(test_metrics.box.map50),
        "test_mask_mAP50-95": float(getattr(test_metrics, 'mask', test_metrics.box).map) if hasattr(test_metrics, 'mask') else float(test_metrics.box.map),
        "test_precision": float(test_metrics.box.mp),
        "test_recall": float(test_metrics.box.mr),
    }

    print(f"\nTEST SET RESULTS:")
    print(f"  mAP50: {test_results['test_mAP50']:.4f}")
    print(f"  mAP50-95: {test_results['test_mAP50-95']:.4f}")
    print(f"  Mask mAP50: {test_results['test_mask_mAP50']:.4f}")
    print(f"  Mask mAP50-95: {test_results['test_mask_mAP50-95']:.4f}")
    print(f"  Precision: {test_results['test_precision']:.4f}")
    print(f"  Recall: {test_results['test_recall']:.4f}")

    # Add test results to summary
    summary['test_metrics'] = test_results
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
