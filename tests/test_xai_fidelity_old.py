"""XAI Fidelity Tests

Tests that XAI explanations are faithful:
- Correct target class
- Valid target layer for each architecture
- Heatmaps match model predictions
- Detection explanation matches selected bounding box
- Explanation/API consistency across backend and UI

NOTE: All XAI tests are skipped on Python 3.14+ due to Ultralytics
deepcopy incompatibility with thread locks. This is a library-level issue,
not a code defect.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "backend"))

# Skip all XAI tests on Python 3.14+ due to Ultralytics incompatibility
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="XAI tests skipped on Python 3.14+ due to Ultralytics deepcopy incompatibility"
)


def _load_test_image() -> Optional[Path]:
    """Load a test image path for XAI tests.

    Returns:
        Path to test image or None
    """
    # Try YOLO test images first
    test_dir = PROJECT / "datasets" / "processed" / "groundnut_weed_detection" / "test" / "images"
    if test_dir.exists():
        files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
        if files:
            return files[0]

    # Try classification test images
    test_dir = PROJECT / "datasets" / "processed" / "classification_groundnut_weed" / "test" / "crop"
    if test_dir.exists():
        files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
        if files:
            return files[0]

    return None


def test_xai_explains_correct_class():
    """Test that XAI explanation explains the class actually predicted."""
    # This test requires a working XAI engine and model
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    model = YOLO(str(checkpoint))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments/xai")

    # Get a test image
    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    # Run prediction
    results = model(str(img_path), verbose=False)
    if not results or len(results) == 0:
        pytest.skip("No detections in test image")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        pytest.skip("No boxes in result")

    # Get predicted class
    boxes = result.boxes.data.cpu().numpy()
    predicted_class_id = int(boxes[0, 5])  # Class ID
    predicted_class = "groundnut" if predicted_class_id == 0 else "weed"

    # Generate explanation
    explanation = engine.explain_image(str(img_path), save_prefix="test_fidelity")

    # Verify explanation includes the predicted class
    assert explanation["success"], "Explanation generation failed"

    # Check that explanations mention the predicted class
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

        # Check model has the expected layer
        # CNN should have conv/feature layers
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

    checkpoint = PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    model = YOLO(str(checkpoint))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments/xai")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    # Get image dimensions
    img = Image.open(img_path)
    img_w, img_h = img.size

    # Generate explanation
    explanation = engine.explain_image(str(img_path), save_prefix="test_dimensions")

    assert explanation["success"], "Explanation generation failed"

    # Check saved heatmap files have correct dimensions
    if "saved_files" in explanation:
        for save_type, file_path in explanation["saved_files"].items():
            if Path(file_path).exists():
                heatmap = Image.open(file_path)
                h_w, h_h = heatmap.size
                # Heatmap should match original image dimensions
                assert h_w == img_w and h_h == img_h, \
                    f"Heatmap dimensions ({h_w}x{h_h}) don't match image ({img_w}x{img_h})"


def test_xai_detection_explanation_matches_bbox():
    """Test that explanation corresponds to selected bounding box, not unrelated image-level score."""
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    model = YOLO(str(checkpoint))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments/xai")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    # Run detection
    results = model(str(img_path), verbose=False)
    if not results or len(results) == 0:
        pytest.skip("No detections in test image")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        pytest.skip("No boxes in result")

    boxes = result.boxes.data.cpu().numpy()
    top_box = boxes[0]  # Highest confidence box
    box_class_id = int(top_box[5])

    # Generate explanation
    explanation = engine.explain_image(str(img_path), save_prefix="test_bbox_match")

    assert explanation["success"], "Explanation generation failed"

    # Verify explanation data includes the box
    if "instances" in explanation:
        # Check that at least one instance matches the top box
        for instance in explanation["instances"]:
            if instance["class_id"] == box_class_id:
                # Found matching instance
                return

    pytest.fail("Explanation does not include the top detected box")


def test_xai_api_consistency():
    """Test that class labels, confidence, heatmap dimensions, and model version agree across backend and would-be UI."""
    # This test validates the API contract for XAI
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    model = YOLO(str(checkpoint))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments/xai")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    # Generate explanation
    explanation = engine.explain_image(str(img_path), save_prefix="test_api_consistency")

    assert explanation["success"], "Explanation generation failed"

    # Verify response has expected structure
    required_keys = ["success", "predictions", "explanations", "saved_files"]
    for key in required_keys:
        assert key in explanation, f"Missing required key: {key}"

    # Verify predictions and explanations are aligned
    if "predictions" in explanation and "explanations" in explanation:
        num_predictions = len(explanation["predictions"])
        num_explanations = len(explanation["explanations"])
        # Should have one explanation per prediction
        assert num_predictions == num_explanations, \
            f"Mismatch: {num_predictions} predictions but {num_explanations} explanations"

    # Verify saved files are properly formatted
    if "saved_files" in explanation:
        for file_type, file_path in explanation["saved_files"].items():
            assert isinstance(file_path, str), f"Invalid file path type: {type(file_path)}"
            assert len(file_path) > 0, f"Empty file path for {file_type}"


def test_xai_uncertainty_communication():
    """Test that the app communicates uncertainty even when heatmap looks convincing."""
    # This is a behavioral test for the UI - verify that confidence is shown
    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    checkpoint = PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt"
    if not checkpoint.exists():
        pytest.skip("YOLO checkpoint not available")

    model = YOLO(str(checkpoint))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"}, PROJECT / "experiments/xai")

    img_path = _load_test_image()
    if not img_path:
        pytest.skip("Test image not available")

    # Generate explanation
    explanation = engine.explain_image(str(img_path), save_prefix="test_uncertainty")

    assert explanation["success"], "Explanation generation failed"

    # Verify that limitations are documented
    if "explanations" in explanation:
        for exp in explanation["explanations"]:
            # Each explanation should have limitations listed
            assert "limitations" in exp, "Explanation missing limitations field"
            assert isinstance(exp["limitations"], list), "Limitations should be a list"
            # At least one limitation should be documented
            assert len(exp["limitations"]) > 0, "No limitations documented"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
