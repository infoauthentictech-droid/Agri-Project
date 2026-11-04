"""API tests for the backend."""

import os

# Backend API tests always exercise the STRICT knowledge-base mode: never let a
# research-mode environment variable (GROUNDNUT_ALLOW_PENDING_RECORDS) change
# the recommendation behaviour these tests assert on.
os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = "0"
# Models load lazily on first use; skip the startup pre-warm in tests.
os.environ.setdefault("GROUNDNUT_PRELOAD_MODELS", "0")

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.utils.security import api_key

client = TestClient(app)
KEY = api_key()
HEADERS = {"X-API-Key": KEY}


@pytest.fixture(scope="module", autouse=True)
def _run_lifespan():
    """Enter the client so the app's lifespan builds ``app.state.services``."""
    with client:
        yield


def _post(path, **kw):
    kw.setdefault("headers", HEADERS)
    return client.post(path, **kw)


def _get(path, **kw):
    kw.setdefault("headers", HEADERS)
    return client.get(path, **kw)


def test_health_check():
    response = _get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "recommendation_engine_loaded" in data
    assert "xai_loaded" in data


def test_auth_required():
    """Requests without a valid API key are rejected with 401."""
    response = client.get("/api/v1/health")
    assert response.status_code == 401


def test_auth_invalid_key():
    response = client.get("/api/v1/health", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_rate_limit():
    """Bursts beyond the per-minute budget receive 429."""
    import app.utils.security as sec
    old = sec.RATE_LIMIT_PER_MINUTE
    sec.RATE_LIMIT_PER_MINUTE = 5
    # Start from an empty window: earlier tests share the same client IP.
    sec._store.clear()
    sec._last_insert_ts.clear()
    try:
        codes = set()
        for _ in range(10):
            r = _get("/api/v1/health")
            codes.add(r.status_code)
        assert 429 in codes
        assert 200 in codes
    finally:
        sec.RATE_LIMIT_PER_MINUTE = old
        sec._store.clear()
        sec._last_insert_ts.clear()


def test_predict_invalid_image():
    response = _post("/api/v1/predict", json={
        "image_data": "invalid_base64_data", "confidence_threshold": 0.5})
    assert response.status_code == 400


def test_detect_invalid_image():
    response = _post("/api/v1/detect", json={
        "image_data": "invalid_base64_data", "confidence_threshold": 0.25,
        "iou_threshold": 0.45})
    assert response.status_code == 400


def test_recommend_missing_crop():
    response = _post("/api/v1/recommend", json={"weed_species": "Echinochloa colona"})
    assert response.status_code == 422  # pydantic validation


def test_explain_invalid_image():
    response = _post("/api/v1/explain", json={
        "image_data": "invalid_base64_data", "include_visualization": True})
    assert response.status_code == 400


def test_recommend_valid_request():
    """Strict knowledge-base mode with validated records must recommend.

    The knowledge base has 37 ``validation_status == passed`` records now, so
    a valid request for a known groundnut weed returns a real recommendation
    (success=True) in strict mode. Pending (unvalidated) records are still
    never used.

    The test forces a FRESH engine with strict mode so it is invariant to the
    order pytest imports the other suites (which set research mode).
    """
    from app.services.recommendation_service import RecommendationService

    previous = os.environ.get("GROUNDNUT_ALLOW_PENDING_RECORDS")
    try:
        os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = "0"
        # Rebuild ONLY the recommendation engine so it reads strict mode.
        client.app.state.services["recommendation"] = RecommendationService()

        response = _post("/api/v1/recommend", json={
            "crop": "groundnut", "weed_species": "Echinochloa colona",
            "crop_growth_stage": "post_emergence"})
        # 200 when engine loaded (200 with success either way), 503 if KB missing
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert data["status"] == "success"
            assert data["candidate_management_options"]  # non-empty list
            assert data["source"] is not None
    finally:
        # Restore so other suites (herbicide/tests) see their own setting.
        if previous is None:
            os.environ.pop("GROUNDNUT_ALLOW_PENDING_RECORDS", None)
        else:
            os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = previous


def _detection_test_images():
    """Held-out images of the dataset the detector was trained on."""
    test_dir = Path(__file__).parent.parent.parent / "datasets/hybrid/detection/images/test"
    return sorted(p for p in test_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def test_predict_real_image_end_to_end():
    """A real dataset image must flow through decode -> validation -> YOLO
    inference without crashing and return predictions in the documented schema."""
    import base64 as b64
    images = _detection_test_images()
    if not images:
        pytest.skip("no test images available")
    image_path = images[0]
    encoded = b64.b64encode(image_path.read_bytes()).decode("ascii")
    response = _post("/api/v1/predict", json={
        "image_data": encoded, "confidence_threshold": 0.25})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data
        assert "num_predictions" in data
        assert "inference_time_ms" in data


def test_detect_real_image_end_to_end():
    import base64 as b64
    images = _detection_test_images()
    if not images:
        pytest.skip("no test images available")
    image_path = images[0]
    encoded = b64.b64encode(image_path.read_bytes()).decode("ascii")
    response = _post("/api/v1/detect", json={
        "image_data": encoded, "confidence_threshold": 0.25,
        "iou_threshold": 0.45})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        if data["detections"]:
            d = data["detections"][0]
            assert "class_id" in d and "class_name" in d and "bounding_box" in d


def test_analyze_rejects_non_agricultural_image():
    """The /analyze domain gate must reject clearly non-field photos (e.g. a
    blank synthetic frame) with an honest message instead of a fake detection.
    A plain 640x480 colour is decoded as an image but has no texture, no
    vegetation and no soil colour, so the gate must return success=False with
    zero detections."""
    import base64 as b64
    import io
    from PIL import Image

    img = Image.new("RGB", (640, 480), (0, 128, 128))  # flat teal frame
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = b64.b64encode(buf.getvalue()).decode("ascii")

    response = _post("/api/v1/analyze", json={
        "image_data": encoded,
        "detection_model": "yolov8n",
        "classification_model": "mobilenet",
    })
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is False
        assert data["num_detections"] == 0
        assert data["detections"] == []
        assert data["error_code"] == "not_plant_image"
        assert data["message"] and "crop-field" in data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
