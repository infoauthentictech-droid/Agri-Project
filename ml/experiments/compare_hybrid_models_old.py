"""Hybrid Model Accuracy Comparison

Compares hybrid models against single-module models to verify that
hybrid models provide higher accuracy as per project objective.

This script:
1. Evaluates all available models on the test set
2. Compares accuracy metrics between single and hybrid models
3. Generates a comparison report
4. Verifies that hybrid models meet the accuracy improvement objective
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from ml.models.registry import build_model
from ml.models.hybrid import list_hybrid_models


class ModelComparator:
    """Compare single and hybrid model accuracy."""

    def __init__(self, test_dataset_path: Path):
        """Initialize model comparator.

        Args:
            test_dataset_path: Path to test dataset
        """
        self.test_dataset_path = test_dataset_path
        self.models_to_test = [
            "mobilenet",
            "efficientnet",
            "resnet50",
            "resnet50_mobilenet",
            "resnet50_efficientnet",
            "resnet50_efficientnet_mobilenet",
        ]
        self.results = {}

    def load_test_images(self) -> List[tuple]:
        """Load test images and labels.

        Returns:
            List of (image_path, label) tuples
        """
        images = []
        test_root = self.test_dataset_path / "test"

        # Load crop images (label 0)
        crop_dir = test_root / "crop"
        if crop_dir.exists():
            for img_path in crop_dir.glob("*.jpg"):
                images.append((img_path, 0))
            for img_path in crop_dir.glob("*.jpeg"):
                images.append((img_path, 0))
            for img_path in crop_dir.glob("*.png"):
                images.append((img_path, 0))

        # Load weed images (label 1)
        weed_dir = test_root / "weed"
        if weed_dir.exists():
            for img_path in weed_dir.glob("*.jpg"):
                images.append((img_path, 1))
            for img_path in weed_dir.glob("*.jpeg"):
                images.append((img_path, 1))
            for img_path in weed_dir.glob("*.png"):
                images.append((img_path, 1))

        return images

    def evaluate_model(self, model_name: str, test_images: List[tuple]) -> Dict:
        """Evaluate a single model on test set.

        Args:
            model_name: Name of model to evaluate
            test_images: List of (image_path, label) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")

        # Build model
        cfg = {
            "model": {
                "name": model_name,
                "num_classes": 2,
            },
            "dataset": {
                "image_size": 128 if model_name in ["mobilenet", "efficientnet"] else 224,
            },
        }

        try:
            model = build_model(cfg)
            model.eval()
        except Exception as e:
            print(f"  Error building model: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Check if checkpoint exists
        checkpoint_path = PROJECT_ROOT / "experiments" / "classification" / model_name / "best_model.pth"
        if not checkpoint_path.exists():
            checkpoint_path = PROJECT_ROOT / "experiments" / "classification" / model_name / "best_model.pt"

        if not checkpoint_path.exists():
            print(f"  Checkpoint not found for {model_name}")
            return {
                "model_name": model_name,
                "error": "Checkpoint not found",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Load checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    model.load_state_dict(ckpt["state_dict"])
                elif "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            return {
                "model_name": model_name,
                "error": f"Checkpoint load error: {e}",
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Evaluate on test set
        correct = 0
        total = 0
        tp = 0  # True positives (weed)
        fp = 0  # False positives
        fn = 0  # False negatives

        for img_path, label in test_images:
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert("RGB")
                img_size = cfg["dataset"]["image_size"]
                img = img.resize((img_size, img_size))
                img_array = np.asarray(img).astype(np.float32) / 255.0
                img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()

                total += 1
                if predicted_class == label:
                    correct += 1

                # For precision/recall (treating weed as positive)
                if label == 1:  # Actual weed
                    if predicted_class == 1:
                        tp += 1
                    else:
                        fn += 1
                elif predicted_class == 1:  # Predicted weed but actual crop
                    fp += 1

            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")

        return {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_samples": total,
        }

    def compare_models(self) -> Dict:
        """Compare all models and generate report.

        Returns:
            Dictionary with comparison results
        """
        print("Loading test images...")
        test_images = self.load_test_images()
        print(f"Loaded {len(test_images)} test images")

        # Evaluate all models
        for model_name in self.models_to_test:
            result = self.evaluate_model(model_name, test_images)
            self.results[model_name] = result

        # Generate comparison report
        report = self._generate_report()

        return report

    def _generate_report(self) -> Dict:
        """Generate comparison report.

        Returns:
            Dictionary with comparison results
        """
        single_models = ["mobilenet", "efficientnet", "resnet50"]
        hybrid_models = ["resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"]

        # Get best single model accuracy
        single_accuracies = [
            self.results[m]["accuracy"] for m in single_models
            if "error" not in self.results[m]
        ]
        best_single_accuracy = max(single_accuracies) if single_accuracies else 0.0
        best_single_model = max(
            single_models,
            key=lambda m: self.results[m].get("accuracy", 0.0),
        )

        # Get hybrid model accuracies
        hybrid_accuracies = {
            m: self.results[m].get("accuracy", 0.0)
            for m in hybrid_models
            if "error" not in self.results[m]
        }

        # Check if any hybrid model beats the best single model
        best_hybrid_accuracy = max(hybrid_accuracies.values()) if hybrid_accuracies else 0.0
        best_hybrid_model = max(hybrid_accuracies, key=hybrid_accuracies.get) if hybrid_accuracies else None

        improvement = best_hybrid_accuracy - best_single_accuracy
        improvement_percentage = (improvement / best_single_accuracy * 100) if best_single_accuracy > 0 else 0.0

        # Verify objective
        objective_met = best_hybrid_accuracy > best_single_accuracy

        report = {
            "objective": "Hybrid model should provide higher accuracy than single module",
            "objective_met": objective_met,
            "best_single_model": best_single_model,
            "best_single_accuracy": float(best_single_accuracy),
            "best_hybrid_model": best_hybrid_model,
            "best_hybrid_accuracy": float(best_hybrid_accuracy),
            "improvement": float(improvement),
            "improvement_percentage": float(improvement_percentage),
            "all_results": self.results,
            "conclusion": (
                f"Hybrid model objective {'MET' if objective_met else 'NOT MET'}. "
                f"Best hybrid ({best_hybrid_model}): {best_hybrid_accuracy:.4f} vs "
                f"Best single ({best_single_model}): {best_single_accuracy:.4f} "
                f"({improvement_percentage:+.2f}% improvement)"
            ),
        }

        return report

    def save_report(self, report: Dict, output_path: Path) -> None:
        """Save comparison report to file.

        Args:
            report: Comparison report
            output_path: Path to save report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")


def main():
    """Main function to run model comparison."""
    test_dataset_path = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed"

    comparator = ModelComparator(test_dataset_path)
    report = comparator.compare_models()

    # Save report
    output_path = PROJECT_ROOT / "experiments" / "evaluation" / "hybrid_comparison_report.json"
    comparator.save_report(report, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("HYBRID MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(report["conclusion"])
    print("=" * 80)


if __name__ == "__main__":
    main()
