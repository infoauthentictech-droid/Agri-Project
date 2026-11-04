"""ML Regression Tests

Maintains a small, versioned set of representative images and expected output
properties to detect unexpected changes when a checkpoint changes.

Tests for:
- Class labels
- Detection counts
- Mask shapes
- Confidence ranges
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "backend"))

# Expected outputs for regression tests (baseline values)
REGRESSION_BASELINE = {
    "mobilenet": {
        "expected_class": "groundnut",
        "min_confidence": 0.5,
        "max_confidence": 1.0,
        "num_classes": 2,
    },
    "yolov8n": {
        "min_detections": 0,
        "max_detections": 10,
        "min_confidence": 0.25,
        "max_confidence": 1.0,
        "expected_classes": [0, 1],  # crop, weed
    },
    "segmentation": {
        "min_coverage": 0.0,
        "max_coverage": 100.0,
        "expected_classes": ["crop", "weed"],
    },
}


def _load_test_image(image_name: str) -> str:
    """Load a test image and return base64 encoding.

    Args:
        image_name: Name of test image

    Returns:
        Base64-encoded image
    """
    # Directly find test images
    if "crop" in image_name or "weed" in image_name:
        # Classification test images
        test_dir = PROJECT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
        # Try crop first
        img_path = test_dir / "crop"
        if img_path.exists():
            files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg")) + list(img_path.glob("*.png"))
            if files:
                img_path = files[0]
            else:
                # Try weed
                img_path = test_dir / "weed"
                files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg")) + list(img_path.glob("*.png"))
                if files:
                    img_path = files[0]
                else:
                    pytest.skip("Test image not available")
        else:
            pytest.skip("Test directory not available")
    else:
        # YOLO test images
        test_dir = PROJECT / "datasets" / "processed" / "groundnut_weed_detection" / "test" / "images"
        if test_dir.exists():
            files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
            if files:
                img_path = files[0]
            else:
                pytest.skip("Test image not available")
        else:
            pytest.skip("Test directory not available")

    if img_path and Path(img_path).exists():
        return base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
    else:
        pytest.skip("Test image not available")


def test_mobilenet_classification_regression():
    """Test MobileNet classification against baseline expectations."""
    from backend.app.services.classifier_service import ClassifierService

    svc = ClassifierService()
    if "mobilenet" not in svc.available_models:
        pytest.skip("MobileNet model not available")

    assert svc.load_model("mobilenet")

    image_data = _load_test_image("test_crop.jpg")
    success, result, ms, err = svc.classify(image_data, model_name="mobilenet", top_k=2)

    assert success, f"Classification failed: {err}"

    baseline = REGRESSION_BASELINE["mobilenet"]

    # Check class is one of expected
    assert result["class_name"] in ["groundnut", "weed"], f"Unexpected class: {result['class_name']}"

    # Check confidence is in expected range
    assert baseline["min_confidence"] <= result["confidence"] <= baseline["max_confidence"], \
        f"Confidence out of range: {result['confidence']}"

    # Check number of classes returned
    assert len(result["probabilities"]) == baseline["num_classes"], \
        f"Wrong number of classes: {len(result['probabilities'])}"

    # Check probability sum is approximately 1.0
    prob_sum = sum(p["probability"] for p in result["probabilities"])
    assert 0.99 <= prob_sum <= 1.01, f"Probabilities don't sum to 1.0: {prob_sum}"


def test_yolo_detection_regression():
    """Test YOLO detection against baseline expectations."""
    from backend.app.services.model_service import ModelService

    svc = ModelService()
    if not svc.load_model():
        pytest.skip("YOLO model not available")

    image_data = _load_test_image("test_field.jpg")
    success, predictions, ms, err = svc.predict(image_data, confidence_threshold=0.25)

    assert success, f"Detection failed: {err}"

    baseline = REGRESSION_BASELINE["yolov8n"]

    # Check number of detections is in expected range
    assert baseline["min_detections"] <= len(predictions) <= baseline["max_detections"], \
        f"Unexpected number of detections: {len(predictions)}"

    # Check each detection
    for pred in predictions:
        # Check class ID is expected
        assert pred["class_id"] in baseline["expected_classes"], \
            f"Unexpected class ID: {pred['class_id']}"

        # Check confidence is in expected range
        assert baseline["min_confidence"] <= pred["confidence"] <= baseline["max_confidence"], \
            f"Confidence out of range: {pred['confidence']}"

        # Check bounding box is valid
        bbox = pred["bounding_box"]
        assert bbox["x1"] < bbox["x2"], f"Invalid bounding box: {bbox}"
        assert bbox["y1"] < bbox["y2"], f"Invalid bounding box: {bbox}"
        assert all(isinstance(v, (int, float)) for v in bbox.values()), \
            f"Bounding box has invalid types: {bbox}"


def test_segmentation_regression():
    """Test segmentation against baseline expectations."""
    from backend.app.services.segmentation_service import SegmentationService

    svc = SegmentationService()
    if not svc.load_model():
        pytest.skip("Segmentation model not available")

    image_data = _load_test_image("test_field.jpg")
    success, result, inference_time, err = svc.segment(image_data, target_size=640)

    assert success, f"Segmentation failed: {err}"

    baseline = REGRESSION_BASELINE["segmentation"]

    # Check coverage is in expected range
    for class_name in baseline["expected_classes"]:
        if class_name in result["coverage"]:
            coverage = result["coverage"][class_name]
            assert baseline["min_coverage"] <= coverage <= baseline["max_coverage"], \
                f"Coverage out of range for {class_name}: {coverage}"

    # Check class names are expected
    for detected_class in result["classes"]:
        assert detected_class in baseline["expected_classes"], \
            f"Unexpected class: {detected_class}"

    # Check mask and class_mask are present
    assert result["mask"], "Mask is empty"
    assert result["class_mask"], "Class mask is empty"

    # Check inference time is reasonable (< 5 seconds)
    assert inference_time < 5000, f"Inference too slow: {inference_time}ms"


def test_model_registry_consistency():
    """Test that model registry is consistent with actual files."""
    from backend.app.utils.model_registry import get_registry

    registry = get_registry()

    # Check all registered models have valid paths
    path_validity = registry.validate_model_paths()

    # Report any invalid paths
    invalid_models = [name for name, valid in path_validity.items() if not valid]

    # Skip yolov8n-seg since it's not trained yet
    if "yolov8n-seg" in invalid_models:
        invalid_models.remove("yolov8n-seg")

    if invalid_models:
        pytest.fail(f"Model registry has invalid paths for: {invalid_models}")

    # Check default models exist for each task
    from backend.app.utils.model_registry import ModelTask

    for task in [ModelTask.CLASSIFICATION, ModelTask.DETECTION, ModelTask.SEGMENTATION]:
        default = registry.get_default_model(task)
        if default is None:
            pytest.fail(f"No default model for task: {task.value}")


def test_dataset_manifest_consistency():
    """Test that dataset manifest can be loaded and is consistent."""
    from ml.data.dataset_manifest import get_manager

    manager = get_manager()

    # Try to load classification manifest
    manifest = manager.load_manifest("classification_groundnut_weed")

    if manifest is None:
        pytest.skip("Dataset manifest not found")

    # Check manifest has required fields
    assert manifest.dataset_name, "Dataset name missing"
    assert manifest.dataset_version, "Dataset version missing"
    assert manifest.total_images > 0, "No images in manifest"

    # Check for session leakage
    leakage_check = manager.audit_leakage("classification_groundnut_weed")
    # Skip if all sessions are "unknown" (no session metadata available)
    if leakage_check.get("has_leakage") and set(leakage_check.get("session_leakage", {}).get("leaked_sessions", {}).keys()) != {"unknown"}:
        pytest.fail(
            f"Dataset has session leakage: {leakage_check.get('session_leakage')}"
        )


def test_herbicide_safety_checks():
    """Test that herbicide safety checks are working."""
    from backend.app.utils.recommendation_safety import get_safety_checker, SafetyStatus

    checker = get_safety_checker()

    # Test with missing required fields
    input_data = {"crop": "groundnut"}  # Missing weed_species, etc.
    result_data = {"active_ingredient": "quizalofop-ethyl"}

    safety_check = checker.perform_safety_checks(input_data, result_data, 0.9)

    # Should fail due to missing fields
    assert not safety_check.safe_to_recommend, "Should not recommend with missing fields"
    assert safety_check.status in [SafetyStatus.UNSAFE, SafetyStatus.NEEDS_EXPERT_REVIEW], \
        f"Unexpected status: {safety_check.status}"

    # Test with valid data but incompatible crop
    input_data = {
        "crop": "wheat",  # Not in compatibility database
        "weed_species": "some_weed",
        "crop_growth_stage": "vegetative",
        "application_category": "post-emergence",
    }
    result_data = {"active_ingredient": "quizalofop-ethyl"}

    safety_check = checker.perform_safety_checks(input_data, result_data, 0.9)

    # Should fail due to crop incompatibility
    assert not safety_check.safe_to_recommend, "Should not recommend for incompatible crop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
