"""Experiment 4: Test Image Resolution

Compares different input resolutions (640, 832, 960) to find the optimal balance
between small-weed recall and inference latency.

Usage:
    python ml/experiments/experiment_4_resolution_comparison.py
    python ml/experiments/experiment_4_resolution_comparison.py --model experiments/yolo/agri_yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "resolution_comparison"


def evaluate_at_resolution(
    model: YOLO,
    data_path: str,
    imgsz: int,
    split: str = "val",
    conf_threshold: float = 0.25,
    device: str = "0",
) -> Dict:
    """Evaluate model at a specific resolution.

    Args:
        model: YOLO model instance
        data_path: Path to data.yaml
        imgsz: Image size to use
        split: Dataset split to evaluate
        conf_threshold: Confidence threshold
        device: Device to use

    Returns:
        Dictionary with metrics and timing
    """
    print(f"  Evaluating at {imgsz}x{imgsz}...")

    try:
        # Measure inference time
        start_time = time.time()
        metrics = model.val(
            data=data_path,
            split=split,
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
        end_time = time.time()

        total_time = end_time - start_time

        return {
            "imgsz": imgsz,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-10),
            "inference_time_seconds": total_time,
        }
    except Exception as e:
        print(f"  Error at {imgsz}x{imgsz}: {e}")
        return {
            "imgsz": imgsz,
            "mAP50": 0.0,
            "mAP50_95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inference_time_seconds": 0.0,
            "error": str(e),
        }


def measure_single_image_latency(
    model: YOLO,
    imgsz: int,
    device: str = "0",
    num_runs: int = 10,
) -> Dict:
    """Measure single-image inference latency at a resolution.

    Args:
        model: YOLO model
        imgsz: Image size
        device: Device to use
        num_runs: Number of inference runs

    Returns:
        Latency statistics
    """
    # Create a dummy image
    import numpy as np

    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        _ = model(dummy_image, imgsz=imgsz, device=device, verbose=False)

    # Measure latency
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_image, imgsz=imgsz, device=device, verbose=False)
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "imgsz": imgsz,
        "avg_latency_ms": np.mean(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "std_latency_ms": np.std(latencies),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 4: Test Image Resolution"
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
        "--resolutions",
        type=int,
        nargs="+",
        default=[640, 832, 960],
        help="Resolutions to test",
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    ap.add_argument(
        "--device",
        default="0",
        help="GPU device or 'cpu'",
    )
    ap.add_argument(
        "--measure-latency",
        action="store_true",
        help="Measure single-image latency",
    )
    args = ap.parse_args(argv)

    # Check GPU availability
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
    print("Experiment 4: Test Image Resolution")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    # Load model
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        return 1

    model = YOLO(args.model)

    # Convert path to forward slashes for YOLO compatibility
    data_path = str(Path(args.data).as_posix())

    # Evaluate at each resolution
    results = []
    latency_results = []

    for imgsz in args.resolutions:
        result = evaluate_at_resolution(model, data_path, imgsz, args.split, args.conf, args.device)
        results.append(result)

        if args.measure_latency:
            latency = measure_single_image_latency(model, imgsz, args.device)
            latency_results.append(latency)

    # Print results table
    print(f"\n{'Resolution':<12} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['imgsz']:<12} "
            f"{r['mAP50']:<10.4f} "
            f"{r['mAP50_95']:<10.4f} "
            f"{r['precision']:<10.4f} "
            f"{r['recall']:<10.4f} "
            f"{r['f1']:<10.4f} "
            f"{r['inference_time_seconds']:<10.2f}"
        )

    if latency_results:
        print(f"\n{'Resolution':<12} {'Avg Latency(ms)':<18} {'Min(ms)':<12} {'Max(ms)':<12} {'Std(ms)':<12}")
        print("-" * 70)
        for l in latency_results:
            print(
                f"{l['imgsz']:<12} "
                f"{l['avg_latency_ms']:<18.2f} "
                f"{l['min_latency_ms']:<12.2f} "
                f"{l['max_latency_ms']:<12.2f} "
                f"{l['std_latency_ms']:<12.2f}"
            )

    # Find optimal resolution based on F1 score
    best_result = max(results, key=lambda x: x["f1"])
    print(f"\nOptimal resolution based on F1: {best_result['imgsz']}x{best_result['imgsz']}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  Inference time: {best_result['inference_time_seconds']:.2f}s")

    # Calculate efficiency metric (F1 / time)
    for r in results:
        if r["inference_time_seconds"] > 0:
            r["efficiency"] = r["f1"] / r["inference_time_seconds"]
        else:
            r["efficiency"] = 0

    best_efficiency = max(results, key=lambda x: x["efficiency"])
    print(f"\nMost efficient resolution (F1/time): {best_efficiency['imgsz']}x{best_efficiency['imgsz']}")
    print(f"  Efficiency: {best_efficiency['efficiency']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"resolution_comparison_{args.split}.json"
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "resolutions_tested": args.resolutions,
        "conf_threshold": args.conf,
        "results": results,
        "best_resolution_f1": best_result["imgsz"],
        "best_f1": best_result["f1"],
        "best_resolution_efficiency": best_efficiency["imgsz"],
        "best_efficiency": best_efficiency["efficiency"],
    }

    if latency_results:
        summary["latency_results"] = latency_results

    results_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
