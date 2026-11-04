"""
Evaluate YOLO-Seg segmentation model on test set.

This script evaluates the YOLOv8-Seg model on the test split and reports:
- mAP50, mAP50-95 for masks
- Precision and recall for masks
- Per-class metrics
"""

import sys
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YOLOSEG_MODEL_PATH = PROJECT_ROOT / "experiments" / "yoloseg" / "agri_yolov8n_seg-6" / "weights" / "best.pt"
YOLOSEG_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_segmentation" / "data.yaml"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation"
EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_yoloseg():
    """Evaluate YOLO-Seg model on test set."""
    print("=" * 60)
    print("EVALUATING YOLO-SEG SEGMENTATION MODEL")
    print("=" * 60)

    from ultralytics import YOLO
    import torch

    # Determine device
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading YOLO-Seg model from: {YOLOSEG_MODEL_PATH}")
    model = YOLO(str(YOLOSEG_MODEL_PATH))

    # Evaluate on test set
    print("Evaluating on TEST set...")
    metrics = model.val(
        data=str(YOLOSEG_DATA_PATH),
        split='test',
        device=device,
        verbose=True,
        save_json=True
    )

    # Extract metrics
    results = {
        "model": "YOLOv8n-Seg",
        "split": "test",
        "mAP50": float(metrics.box.map50),  # Box mAP50
        "mAP50-95": float(metrics.box.map),  # Box mAP50-95
        "mask_mAP50": float(metrics.seg.map50),  # Mask mAP50
        "mask_mAP50-95": float(metrics.seg.map),  # Mask mAP50-95
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "class_metrics": {
            "crop": {
                "precision": float(metrics.box.p[0]),
                "recall": float(metrics.box.r[0]),
                "mask_precision": float(metrics.seg.p[0]),
                "mask_recall": float(metrics.seg.r[0]),
            },
            "weed": {
                "precision": float(metrics.box.p[1]),
                "recall": float(metrics.box.r[1]),
                "mask_precision": float(metrics.seg.p[1]),
                "mask_recall": float(metrics.seg.r[1]),
            }
        }
    }

    print("\n" + "=" * 60)
    print("YOLO-SEG TEST SET RESULTS")
    print("=" * 60)
    print(f"Box mAP50: {results['mAP50']:.4f}")
    print(f"Box mAP50-95: {results['mAP50-95']:.4f}")
    print(f"Mask mAP50: {results['mask_mAP50']:.4f}")
    print(f"Mask mAP50-95: {results['mask_mAP50-95']:.4f}")
    print(f"Box Precision: {results['precision']:.4f}")
    print(f"Box Recall: {results['recall']:.4f}")
    print(f"Mask Precision: {results['mask_precision']:.4f}")
    print(f"Mask Recall: {results['mask_recall']:.4f}")
    print(f"\nClass-wise metrics:")
    print(f"  Crop - Box P={results['class_metrics']['crop']['precision']:.4f}, R={results['class_metrics']['crop']['recall']:.4f}")
    print(f"  Crop - Mask P={results['class_metrics']['crop']['mask_precision']:.4f}, R={results['class_metrics']['crop']['mask_recall']:.4f}")
    print(f"  Weed - Box P={results['class_metrics']['weed']['precision']:.4f}, R={results['class_metrics']['weed']['recall']:.4f}")
    print(f"  Weed - Mask P={results['class_metrics']['weed']['mask_precision']:.4f}, R={results['class_metrics']['weed']['mask_recall']:.4f}")

    # Save results
    output_file = EVALUATION_OUTPUT_DIR / "yoloseg_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    sys.exit(evaluate_yoloseg())
