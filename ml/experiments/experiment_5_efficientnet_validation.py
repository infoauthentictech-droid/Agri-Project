"""Experiment 5: Validate EfficientNet Independently

Tests the EfficientNet model to verify it generalizes well and checks for
potential source-image overlap between training and test sets.

This script:
1. Evaluates EfficientNet on the test set with detailed metrics
2. Analyzes per-class performance (precision, recall, F1, calibration)
3. Checks for overfitting signs (large train-test gap)
4. Provides guidance for testing on new field captures

Usage:
    python ml/experiments/experiment_5_efficientnet_validation.py
    python ml/experiments/experiment_5_efficientnet_validation.py --model experiments/classification/efficientnet/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.classifier_service import ClassifierService

DEFAULT_MODEL = "efficientnet"
TEST_DATA_DIR = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "efficientnet_validation"


def load_test_images(test_dir: Path) -> Tuple[List[str], List[int]]:
    """Load test images and their labels.

    Args:
        test_dir: Path to test directory (should have crop/ and weed/ subdirs)

    Returns:
        Tuple of (image_paths, labels)
    """
    image_paths = []
    labels = []

    # Load crop images (label 0)
    crop_dir = test_dir / "crop"
    if crop_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(crop_dir.glob(ext)):
                image_paths.append(str(img_path))
                labels.append(0)

    # Load weed images (label 1)
    weed_dir = test_dir / "weed"
    if weed_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img_path in sorted(weed_dir.glob(ext)):
                image_paths.append(str(img_path))
                labels.append(1)

    return image_paths, labels


def encode_image(image_path: str) -> str:
    """Encode image to base64.

    Args:
        image_path: Path to image

    Returns:
        Base64 encoded image
    """
    import base64

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def evaluate_classifier(
    service: ClassifierService,
    model_name: str,
    image_paths: List[str],
    labels: List[int],
) -> Dict:
    """Evaluate classifier on test set.

    Args:
        service: ClassifierService instance
        model_name: Name of model to use
        image_paths: List of image paths
        labels: Ground truth labels

    Returns:
        Evaluation metrics
    """
    print(f"  Evaluating {len(image_paths)} images...")

    predictions = []
    confidences = []
    all_probabilities = []

    for img_path in image_paths:
        encoded = encode_image(img_path)
        success, result, ms, err = service.classify(encoded, model_name=model_name, top_k=2)

        if success:
            # Convert class_name to label
            class_name = result["class_name"].lower()
            if class_name == "crop" or class_name == "groundnut":
                pred_label = 0
            else:
                pred_label = 1

            predictions.append(pred_label)
            confidences.append(result["confidence"])
            all_probabilities.append(result["probabilities"])
        else:
            print(f"  Error on {img_path}: {err}")
            predictions.append(-1)  # Error marker
            confidences.append(0.0)
            all_probabilities.append([])

    # Calculate metrics
    y_true = np.array(labels)
    y_pred = np.array(predictions)

    # Filter out errors
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) == 0:
        return {"error": "No valid predictions"}

    # Classification report
    report = classification_report(y_true_valid, y_pred_valid, output_dict=True, target_names=["crop", "weed"])

    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pred_valid)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true_valid, y_pred_valid, average=None)

    # Overall accuracy
    accuracy = (y_pred_valid == y_true_valid).mean()

    # Confidence calibration (for positive class - weed)
    weed_indices = y_true_valid == 1
    if weed_indices.sum() > 0:
        weed_probs = []
        weed_true = []
        for i, valid in enumerate(valid_mask):
            if valid and y_true[i] == 1:
                # Get probability for weed class
                probs = all_probabilities[i]
                for prob in probs:
                    if prob["class_name"].lower() in ["weed"]:
                        weed_probs.append(prob["probability"])
                        weed_true.append(1)
                        break

        if len(weed_probs) > 0:
            prob_true, prob_pred = calibration_curve(weed_true, weed_probs, n_bins=10)
        else:
            prob_true, prob_pred = [], []
    else:
        prob_true, prob_pred = [], []

    return {
        "num_images": len(image_paths),
        "num_valid_predictions": len(y_true_valid),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            "crop": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "weed": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "calibration": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


def check_overfitting(train_metrics: Dict, test_metrics: Dict) -> List[str]:
    """Check for signs of overfitting.

    Args:
        train_metrics: Training metrics from results.json
        test_metrics: Test metrics from evaluation

    Returns:
        List of overfitting warnings
    """
    warnings = []

    train_acc = train_metrics.get("test_accuracy", 0)  # EfficientNet results.json uses "test_accuracy"
    test_acc = test_metrics.get("accuracy", 0)

    gap = train_acc - test_acc

    if gap > 0.05:  # 5% gap
        warnings.append(
            f"⚠️  Large train-test accuracy gap: {gap:.2%}. "
            "Possible overfitting or source-image overlap."
        )
    elif gap > 0.02:  # 2% gap
        warnings.append(
            f"⚠️  Moderate train-test accuracy gap: {gap:.2%}. "
            "Monitor for overfitting."
        )

    # Check per-class performance
    train_crop_f1 = train_metrics.get("class_metrics", {}).get("crop", {}).get("f1-score", 0)
    test_crop_f1 = test_metrics.get("per_class_metrics", {}).get("crop", {}).get("f1", 0)

    if train_crop_f1 - test_crop_f1 > 0.05:
        warnings.append(
            f"⚠️  Crop class F1 gap: {train_crop_f1 - test_crop_f1:.2%}. "
            "Check for overfitting on crop images."
        )

    train_weed_f1 = train_metrics.get("class_metrics", {}).get("weed", {}).get("f1-score", 0)
    test_weed_f1 = test_metrics.get("per_class_metrics", {}).get("weed", {}).get("f1", 0)

    if train_weed_f1 - test_weed_f1 > 0.05:
        warnings.append(
            f"⚠️  Weed class F1 gap: {train_weed_f1 - test_weed_f1:.2%}. "
            "Check for overfitting on weed images."
        )

    if not warnings:
        warnings.append("✓ No significant overfitting detected.")

    return warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Experiment 5: Validate EfficientNet Independently"
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name or path (e.g., efficientnet, mobilenet)",
    )
    ap.add_argument(
        "--test-data",
        default=str(TEST_DATA_DIR),
        help="Path to test data directory",
    )
    ap.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )
    ap.add_argument(
        "--train-results",
        help="Path to training results.json for overfitting check",
    )
    args = ap.parse_args(argv)

    print(f"\n{'='*70}")
    print("Experiment 5: Validate EfficientNet Independently")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Test data: {args.test_data}")
    print(f"{'='*70}\n")

    # Initialize classifier service
    service = ClassifierService()

    # Check if model is available
    if args.model not in service.available_models:
        print(f"Error: Model '{args.model}' not available")
        print(f"Available models: {service.available_models}")
        return 1

    # Load model
    print(f"Loading model: {args.model}")
    if not service.load_model(args.model):
        print(f"Error: Failed to load model '{args.model}'")
        return 1

    # Load test images
    test_dir = Path(args.test_data)
    if not test_dir.exists():
        print(f"Error: Test data directory not found: {test_dir}")
        return 1

    image_paths, labels = load_test_images(test_dir)
    print(f"Loaded {len(image_paths)} test images ({labels.count(0)} crop, {labels.count(1)} weed)")

    if len(image_paths) == 0:
        print("Error: No test images found")
        return 1

    # Evaluate
    test_metrics = evaluate_classifier(service, args.model, image_paths, labels)

    # Print results
    print(f"\n{'='*70}")
    print("Test Set Evaluation Results")
    print(f"{'='*70}")
    print(f"Total images: {test_metrics['num_images']}")
    print(f"Valid predictions: {test_metrics['num_valid_predictions']}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 56)
    for class_name in ["crop", "weed"]:
        metrics = test_metrics["per_class_metrics"][class_name]
        print(
            f"{class_name:<10} {metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['support']:<10}"
        )

    print(f"\nConfusion Matrix:")
    cm = test_metrics["confusion_matrix"]
    print(f"                Predicted")
    print(f"              Crop  Weed")
    print(f"Actual Crop   {cm[0][0]:<5} {cm[0][1]:<5}")
    print(f"      Weed    {cm[1][0]:<5} {cm[1][1]:<5}")

    # Check for overfitting if training results provided
    overfitting_warnings = []
    if args.train_results:
        train_results_path = Path(args.train_results)
        if train_results_path.exists():
            with open(train_results_path, "r") as f:
                train_metrics = json.load(f)
            overfitting_warnings = check_overfitting(train_metrics, test_metrics)

            print(f"\n{'='*70}")
            print("Overfitting Analysis")
            print(f"{'='*70}")
            for warning in overfitting_warnings:
                print(warning)
        else:
            print(f"\nWarning: Training results file not found: {args.train_results}")
    else:
        print(f"\nTip: Provide --train-results to check for overfitting")

    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / f"efficientnet_validation_{args.model}.json"
    summary = {
        "model": args.model,
        "test_data": str(args.test_data),
        "test_metrics": test_metrics,
        "overfitting_warnings": overfitting_warnings,
    }

    results_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {results_file}")

    # Guidance for new field captures
    print(f"\n{'='*70}")
    print("Guidance for Testing on New Field Captures")
    print(f"{'='*70}")
    print("1. Collect images from new field sessions (different from training data)")
    print("2. Ensure diverse lighting conditions, camera angles, and growth stages")
    print("3. Include sparse, young, and partially hidden weeds")
    print("4. Organize in similar directory structure (crop/ and weed/ subdirs)")
    print("5. Run this script with --test-data pointing to new data")
    print("6. Compare results with current test set to detect source-image overlap")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
