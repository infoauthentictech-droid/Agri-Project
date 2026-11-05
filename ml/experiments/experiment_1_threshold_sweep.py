"""Experiment 1: Confidence Threshold Sweep for YOLO Detection

Evaluates YOLO detection at different confidence thresholds to find the optimal
balance between precision and recall.

Thresholds to test: 0.10, 0.15, 0.20, 0.25, 0.30

Usage:
    python ml/experiments/experiment_1_threshold_sweep.py
    python ml/experiments/experiment_1_threshold_sweep.py --model experiments/yolo/agri_yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "threshold_sweep"


def evaluate_at_threshold(
    model: YOLO,
    data_path: str,
    conf_threshold: float,
    split: str = "val",
    device: str = "0",
) -> Dict:
    """Evaluate model at a specific confidence threshold.

    Args:
        model: YOLO model instance
        data_path: Path to data.yaml
        conf_threshold: Confidence threshold to use
        split: Dataset split to evaluate (val/test)
        device: Device to use

    Returns:
        Dictionary with metrics
    """
    print(f"  Evaluating at conf={conf_threshold:.2f}...")

    try:
        metrics = model.val(
            data=data_path,
            split=split,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )

        return {
            "conf_threshold": conf_threshold,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-10),
        }
    except Exception as e:
        print(f"  Error at conf={conf_threshold:.2f}: {e}")
        return {
            "conf_threshold": conf_threshold,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 1: Confidence Threshold Sweep for YOLO Detection"
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
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25, 0.30],
        help="Confidence thresholds to test",
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
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
    print("Experiment 1: Confidence Threshold Sweep")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load model
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return 1

    model = YOLO(args.model)

    # Convert path to forward slashes for YOLO compatibility
    data_path = str(Path(args.data).as_posix())

    # Evaluate at each threshold
    results = []
    for threshold in args.thresholds:
        result = evaluate_at_threshold(model, data_path, threshold, args.split, args.device)
        results.append(result)

    # Print results table
    print(f"\n{'Threshold':<12} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['conf_threshold']:<12.2f} "
            f"{r['mAP50']:<10.4f} "
            f"{r['mAP50_95']:<10.4f} "
            f"{r['precision']:<10.4f} "
            f"{r['recall']:<10.4f} "
            f"{r['f1']:<10.4f}"
        )

    # Find optimal threshold based on F1 score
    best_result = max(results, key=lambda x: x["f1"])
    print(f"\nOptimal threshold based on F1: {best_result['conf_threshold']:.2f}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"threshold_sweep_{args.split}.json"
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "thresholds_tested": args.thresholds,
        "results": results,
        "best_threshold": best_result["conf_threshold"],
        "best_f1": best_result["f1"],
    }

    results_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
