"""Smoke tests for the ML services used by the backend.

These tests exercise the real model/service code paths (no mocks) so that
broken model loading, class mappings or JSON serialization regressions are
caught early. They are intentionally few and focused (YOLO inference is slow).
"""
import base64
import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "backend"))


def _yolo_checkpoint() -> Path:
    # Try multiple possible checkpoint locations
    checkpoints = [
        PROJECT / "experiments/yolo/agri_yolov8n/weights/best.pt",
        PROJECT / "runs/detect/experiments/yolo/agri_yolov8n/weights/best.pt",
    ]
    for ckpt in checkpoints:
        if ckpt.exists():
            return ckpt
    return checkpoints[0]  # Return default for error message


def _yolo_sample_image() -> Path:
    # Test images are in test/images, not images/test
    test_dir = PROJECT / "datasets/processed/groundnut_weed_detection/test/images"
    images = sorted(test_dir.glob("*.jpeg")) + sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
    if not images:
        pytest.skip("no YOLO test images available")
    return images[0]


def _classifier_sample_image() -> Path:
    # Test images are in test/crop and test/weed, not test/groundnut
    test_dirs = [
        PROJECT / "datasets/processed/classification_groundnut_weed/test/crop",
        PROJECT / "datasets/processed/classification_groundnut_weed/test/weed",
    ]
    for test_dir in test_dirs:
        images = sorted(test_dir.glob("*.jpeg")) + sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))
        if images:
            return images[0]
    pytest.skip("no classification test patches available")


def test_load_class_names_reads_data_yaml():
    """Class names must come from the dataset config, not hard-coded guesses."""
    from backend.app.services.model_service import load_class_names

    names = load_class_names(str(PROJECT / "datasets/processed/groundnut_weed_detection/data.yaml"))
    assert isinstance(names, dict)
    assert 0 in names and 1 in names
    assert len(names) == 2


def test_model_service_predict_contract():
    """ModelService.predict returns the documented schema."""
    if not _yolo_checkpoint().exists():
        pytest.skip("YOLO checkpoint not present")

    from backend.app.services.model_service import ModelService

    svc = ModelService()
    # The service is lazy by design (no YOLO checkpoint touched in __init__);
    # load the default variant explicitly before predicting.
    assert svc.load_model()
    assert svc.is_loaded()

    encoded = base64.b64encode(_yolo_sample_image().read_bytes()).decode("ascii")
    ok, predictions, ms, err = svc.predict(encoded, confidence_threshold=0.25)
    assert ok, err
    assert isinstance(predictions, list)
    if predictions:
        p = predictions[0]
        assert {"class_id", "class_name", "confidence", "bounding_box"} <= set(p)
        assert {"x1", "y1", "x2", "y2"} <= set(p["bounding_box"])


def test_xai_explain_result_is_json_serializable():
    """The XAI engine response must be plain JSON (regression: numpy arrays
    previously crashed json.dumps in the API path)."""
    import sys

    # Skip on Python 3.14+ due to ultralytics deepcopy incompatibility with thread locks
    if sys.version_info >= (3, 14):
        pytest.skip("XAI test skipped on Python 3.14+ due to ultralytics deepcopy incompatibility")

    if not _yolo_checkpoint().exists():
        pytest.skip("YOLO checkpoint not present")

    from ml.xai.api import XAIEngine
    from ultralytics import YOLO

    model = YOLO(str(_yolo_checkpoint()))
    engine = XAIEngine(model, {0: "groundnut", 1: "weed"},
                       PROJECT / "experiments/xai")
    result = engine.explain_image(str(_yolo_sample_image()), save_prefix="pytest_xai")
    assert result["success"] is True
    # Must be fully JSON serializable (json.dumps raises on failure)
    json.dumps(result)


def test_production_inference_loads_and_predicts():
    """The classifier service must load its checkpoint correctly and
    return a valid prediction dictionary."""
    from backend.app.services.classifier_service import ClassifierService

    svc = ClassifierService()
    available = svc.available_models
    if not available:
        pytest.skip("No classifier models available")

    # Load the first available model (e.g., mobilenet)
    model_name = available[0]
    assert svc.load_model(model_name)
    assert svc.is_loaded(model_name)

    # Load and encode the test image
    encoded = base64.b64encode(_classifier_sample_image().read_bytes()).decode("ascii")

    # Run classification
    ok, result, ms, err = svc.classify(encoded, model_name=model_name, top_k=2)
    assert ok, err
    assert "class_name" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["probabilities"], list)
    assert len(result["probabilities"]) <= 2