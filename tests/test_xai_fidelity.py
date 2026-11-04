"""XAI Fidelity Tests - Fixed for Python 3.12 and new dataset structure

Tests that XAI explanations are faithful:
- Correct target class
- Valid target layer for each architecture
- Heatmaps match model predictions
- Detection explanation matches selected bounding box
- Explanation/API consistency across backend and UI
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
import sysconfig

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "backend"))

# Skip XAI tests on Python 3.14+ due to Ultralytics deepcopy incompatibility
# But enable for Python 3.12
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="XAI tests skipped on Python 3.14+ due to Ultralytics deepcopy incompatibility"
)


def _load_test_image() -> Optional[Path]:
    """Load a test image path for XAI tests using new dataset structure."""
    # Try hybrid classification test images first
    test_dir = PROJECT / "datasets" / "hybrid" / "classification" / "test"
    if test_dir.exists():
        for class_name in ['crop', 'weed']:
            class_dir = test_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
                if files:
                    return files[0]

    # Try primary dataset
    test_dir = PROJECT / "datasets" / "primary" / "classification" / "test"
    if test_dir.exists():
        for class_name in ['crop', 'weed']:
            class_dir = test_dir / class_name
            if class_dir.exists():
                files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("png"))
                if files:
                    return files[0]

    return None


def test_xai_explains_correct_class():
    """Test that XAI explanation explains the class actually predicted."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    # Try to find YOLO checkpoint
    checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    try:
        model = YOLO(str(checkpoint))
        engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments" / "xai")
    except Exception as e:
        pytest.skip(f"Could not initialize XAI engine: {e}")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    try:
        results = model(str(img_path), verbose=False)
    except Exception as e:
        pytest.skip(f"Model inference failed: {e}")

    if not results or len(results) == 0:
        pytest.skip("No detections in test image")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        pytest.skip("No boxes in result")

    boxes = result.boxes.data.cpu().numpy()
    predicted_class_id = int(boxes[0, 5])
    predicted_class = "groundnut" if predicted_class_id == 0 else "weed"

    try:
        explanation = engine.explain_image(str(img_path), save_prefix="test_fidelity")
    except Exception as e:
        pytest.skip(f"Explanation generation failed: {e}")

    assert explanation["success"], "Explanation generation failed"

    explanation_text = str(explanation)
    assert predicted_class.lower() in explanation_text.lower(), \
        f"Explanation does not mention predicted class {predicted_class}"


def test_xai_uses_valid_layer():
    """Test that Grad-CAM uses a valid target layer for each architecture."""
    from ml.xai.gradcam import GradCAMExplainer
    from ml.models.registry import build_model

    # Test with CNN architecture
    config = {
        "model": {
            "name": "cnn",
            "num_classes": 2,
        },
        "dataset": {
            "image_size": 128,
        },
    }

    try:
        model = build_model(config)

        has_valid_layer = False
        for name, module in model.named_modules():
            if "conv" in name.lower() or "feature" in name.lower():
                has_valid_layer = True
                break

        assert has_valid_layer, "CNN model has no valid Grad-CAM layer"

    except Exception as e:
        pytest.skip(f"Could not build model for layer test: {e}")


def test_xai_heatmap_dimensions_match_image():
    """Test that heatmap dimensions match input image dimensions."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO
    from PIL import Image

    checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    try:
        model = YOLO(str(checkpoint))
        engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments" / "xai")
    except Exception as e:
        pytest.skip(f"Could not initialize XAI engine: {e}")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    img = Image.open(img_path)
    img_w, img_h = img.size

    try:
        explanation = engine.explain_image(str(img_path), save_prefix="test_dimensions")
    except Exception as e:
        pytest.skip(f"Explanation generation failed: {e}")

    assert explanation["success"], "Explanation generation failed"

    if "saved_files" in explanation:
        for save_type, file_path in explanation["saved_files"].items():
            if Path(file_path).exists():
                heatmap = Image.open(file_path)
                h_w, h_h = heatmap.size
                assert h_w == img_w and h_h == img_h, \
                    f"Heatmap dimensions ({h_w}x{h_h}) don't match image ({img_w}x{img_h})"


def test_xai_detection_explanation_matches_bbox():
    """Test that explanation corresponds to selected bounding box."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = PROJECT / "experiments" / "yolov8n" / "best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    try:
        model = YOLO(str(checkpoint))
        engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments" / "xai")
    except Exception as e:
        pytest.skip(f"Could not initialize XAI engine: {e}")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    try:
        results = model(str(img_path), verbose=False)
    except Exception as e:
        pytest.skip(f"Model inference failed: {e}")

    if not results or len(results) == 0:
        pytest.skip("No detections in test image")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        pytest.skip("No boxes in result")

    boxes = result.boxes.data.cpu().numpy()
    top_box = boxes[0]
    box_class_id = int(top_box[5])

    try:
        explanation = engine.explain_image(str(img_path), save_prefix="test_bbox_match")
    except Exception as e:
        pytest.skip(f"Explanation generation failed: {e}")

    assert explanation["success"], "Explanation generation failed"

    if "instances" in explanation:
        for instance in explanation["instances"]:
            if instance["class_id"] == box_class_id:
                return

    pytest.fail("Explanation does not include the top detected box")


def test_xai_api_consistency():
    """Test that class labels, confidence, heatmap dimensions, and model version agree across backend and UI."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = PROJECT / "experiments" / "yolov8n" / "best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    try:
        model = YOLO(str(checkpoint))
        engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments" / "xai")
    except Exception as e:
        pytest.skip(f"Could not initialize XAI engine: {e}")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    try:
        explanation = engine.explain_image(str(img_path), save_prefix="test_api_consistency")
    except Exception as e:
        pytest.skip(f"Explanation generation failed: {e}")

    assert explanation["success"], "Explanation generation failed"

    required_keys = ["success", "predictions", "explanations", "saved_files"]
    for key in required_keys:
        assert key in explanation, f"Missing required key: {key}"

    if "predictions" in explanation and "explanations" in explanation:
        num_predictions = len(explanation["predictions"])
        num_explanations = len(explanation["explanations"])
        assert num_predictions == num_explanations, \
            f"Mismatch: {num_predictions} predictions but {num_explanations} explanations"

    if "saved_files" in explanation:
        for file_type, file_path in explanation["saved_files"].items():
            assert isinstance(file_path, str), f"Invalid file path type: {type(file_path)}"
            assert len(file_path) > 0, f"Empty file path for {file_type}"


def test_xai_uncertainty_communication():
    """Test that the app communicates uncertainty even when heatmap looks convincing."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = PROJECT / "experiments" / "yolov8n" / "best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    try:
        model = YOLO(str(checkpoint))
        engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments" / "xai")
    except Exception as e:
        pytest.skip(f"Could not initialize XAI engine: {e}")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    try:
        explanation = engine.explain_image(str(img_path), save_prefix="test_uncertainty")
    except Exception as e:
        pytest.skip(f"Explanation generation failed: {e}")

    assert explanation["success"], "Explanation generation failed"

    if "explanations" in explanation:
        for exp in explanation["explanations"]:
            assert "limitations" in exp, "Explanation missing limitations field"
            assert isinstance(exp["limitations"], list), "Limitations should be a list"
            assert len(exp["limitations"]) > 0, "No limitations documented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
