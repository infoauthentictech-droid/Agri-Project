
import base64
import io

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from PIL import Image
from pydantic import BaseModel

from app.schemas.segmentation import (
    SegmentationRequest,
    SegmentationResponse,
)
from app.schemas.model_catalog import ModelEntry, ModelsResponse
from app.schemas.analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from app.schemas.detection import DetectionRequest, DetectionResponse
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.schemas.explanation import ExplanationRequest
from app.schemas.recommendation import RecommendationRequest
from app.utils.input_validation import get_request_validator
from app.utils.security import rate_limit, require_api_key
from app.services import (
    ModelService,
    ClassifierService,
    RecommendationService,
    XAIService,
    SegmentationService,
)
from app.utils.plant_gate import PlantImageGate
from app.utils.density import compute_density_estimate

# Every /api/v1 route requires the X-API-Key header and is rate limited per
# client IP (see app.utils.security).
router = APIRouter(dependencies=[Depends(require_api_key), Depends(rate_limit)])

#: One gate per process: its ImageNet featurizer is loaded once, not per request.
_plant_gate = PlantImageGate()


class PlantGateRequest(BaseModel):
    """Request for plant gate check."""
    image_data: str


class PlantGateResponse(BaseModel):
    """Response from plant gate check."""
    is_plant: bool
    probability: float
    threshold: float
    reasons: list[str]


def build_services() -> dict:
    """Build and return all service instances.

    Returns:
        Dictionary containing all service instances
    """
    return {
        "model": ModelService(),
        "classifier": ClassifierService(),
        "recommendation": RecommendationService(),
        "xai": XAIService(),
        "segmentation": SegmentationService(),
    }


def warm_up_services(services: dict) -> None:
    """Warm up services by loading models.

    Args:
        services: Dictionary of service instances
    """
    # Warm up the default detection model
    model_service = services["model"]
    if not model_service.is_loaded():
        model_service.load_model()

    # Warm up XAI engine
    xai_service = services["xai"]
    if not xai_service.is_loaded():
        xai_service.load_engine()


def _safe_error(error_msg: str | None) -> str:
    """Return a safe error message for API responses."""
    return error_msg or "Segmentation failed."


def _reject_invalid(is_valid: bool, error_msg: str | None) -> None:
    """Raise 400 for a request that failed input validation."""
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {error_msg}",
        )


def _decode_rgb(image_data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")


@router.post("/segment", response_model=SegmentationResponse)
async def segment(request: Request, body: SegmentationRequest):
    """Run semantic segmentation on image (base64)."""
    # Enhanced validation with security checks
    validator = get_request_validator()
    _reject_invalid(*validator.validate_segmentation_request(
        body.image_data, body.target_size
    ))

    services = request.app.state.services
    seg = services["segmentation"]

    success, result, inference_time, error_msg = seg.segment(
        body.image_data,
        target_size=body.target_size,
    )

    if not success:
        code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "not loaded" in (error_msg or "")
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(
            status_code=code,
            detail=_safe_error(error_msg),
        )

    return SegmentationResponse(
        success=True,
        mask=result["mask"],
        class_mask=result["class_mask"],
        classes=result["classes"],
        coverage=result["coverage"],
        class_names=result["class_names"],
        inference_time_ms=inference_time,
    )


@router.post("/plant_gate", response_model=PlantGateResponse)
async def plant_gate(request: Request, body: PlantGateRequest):
    """Check if an image is a plant image using the plant gate."""
    _reject_invalid(*get_request_validator().validate_detection_request(body.image_data))

    rgb = np.asarray(_decode_rgb(body.image_data))
    result = _plant_gate.assess(rgb)

    return PlantGateResponse(
        is_plant=result["is_plant"],
        probability=result["probability"],
        threshold=result["threshold"],
        reasons=result["reasons"],
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request):
    """Get available models from the backend."""
    services = request.app.state.services

    model_service = services["model"]
    classifier_service = services["classifier"]

    detection_models = [
        ModelEntry(
            name=name,
            available=info["available"],
            loaded=info["loaded"],
            kind="detection",
            family=info["family"],
        )
        for name, info in model_service.model_status().items()
    ]

    classification_models = [
        ModelEntry(
            name=model_name,
            available=True,
            loaded=classifier_service.is_loaded(model_name),
            kind="classification",
            family="pytorch",
            accuracy=classifier_service.test_accuracy(model_name),
            accuracy_basis="held-out test set (results.json)",
        )
        for model_name in classifier_service.available_models
    ]

    return ModelsResponse(
        success=True,
        detection_models=detection_models,
        classification_models=classification_models,
        default_detection_model=model_service.default_model,
        default_classification_model=classifier_service.default_model,
    )


def _detect_or_raise(model_service, image_data, confidence, iou, model_name):
    success, detections, inference_time, error = model_service.detect(
        image_data,
        confidence_threshold=confidence,
        iou_threshold=iou,
        model_name=model_name,
    )
    if not success:
        code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "not loaded" in (error or "")
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=code, detail=error or "Detection failed")
    return detections, inference_time


@router.post("/detect", response_model=DetectionResponse)
async def detect(request: Request, body: DetectionRequest):
    """Run object detection on an image (base64)."""
    _reject_invalid(*get_request_validator().validate_detection_request(
        body.image_data, body.confidence_threshold
    ))
    detections, inference_time = _detect_or_raise(
        request.app.state.services["model"],
        body.image_data,
        body.confidence_threshold,
        body.iou_threshold,
        body.model_name,
    )
    return DetectionResponse(
        success=True,
        detections=detections,
        num_detections=len(detections),
        inference_time_ms=inference_time,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, body: PredictionRequest):
    """Legacy alias of /detect with the older response field names."""
    _reject_invalid(*get_request_validator().validate_detection_request(
        body.image_data, body.confidence_threshold
    ))
    detections, inference_time = _detect_or_raise(
        request.app.state.services["model"],
        body.image_data,
        body.confidence_threshold,
        0.45,
        None,
    )
    return PredictionResponse(
        success=True,
        predictions=detections,
        num_predictions=len(detections),
        inference_time_ms=inference_time,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: Request, body: ClassifyRequest):
    """Classify a single image with one of the registered classifiers."""
    _reject_invalid(*get_request_validator().validate_classification_request(
        body.image_data, body.top_k
    ))
    success, result, inference_time, error = request.app.state.services["classifier"].classify(
        body.image_data, model_name=body.model_name, top_k=body.top_k or 3
    )
    if not success:
        code = (
            status.HTTP_404_NOT_FOUND
            if "not available" in (error or "") or "not yet trained" in (error or "")
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=code, detail=error or "Classification failed")
    return ClassifyResponse(success=True, result=result, inference_time_ms=inference_time)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest):
    """Combined detection + classification with plant gate check."""
    services = request.app.state.services
    model_service = services["model"]
    classifier_service = services["classifier"]

    detection_model = body.detection_model or model_service.default_model
    classification_model = body.classification_model or classifier_service.default_model

    def failure(error_code: str, message: str) -> AnalyzeResponse:
        return AnalyzeResponse(
            success=False,
            detection_model=detection_model,
            classification_model=classification_model,
            num_detections=0,
            detections=[],
            detection_time_ms=0.0,
            error_code=error_code,
            message=message,
        )

    _reject_invalid(*get_request_validator().validate_detection_request(
        body.image_data, body.confidence_threshold
    ))

    image = _decode_rgb(body.image_data)
    rgb = np.asarray(image)

    gate_result = _plant_gate.assess(rgb)
    if not gate_result["is_plant"]:
        return failure(
            "not_plant_image",
            gate_result["reasons"][0] if gate_result["reasons"] else "Not a crop-field photo",
        )

    detection_success, detections, detection_time, detection_error = model_service.detect(
        body.image_data,
        confidence_threshold=body.confidence_threshold,
        iou_threshold=body.iou_threshold,
        model_name=detection_model,
    )
    if not detection_success:
        return failure("detection_failed", detection_error or "Detection failed")

    density_estimate = compute_density_estimate(
        detections, image_width=image.width, image_height=image.height
    )

    # Classify the crop of the top-confidence detection so the label belongs
    # to one specific detected object.
    classification = None
    classification_box = None
    classification_time = 0.0
    error_code = None
    message = None
    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        bb = top["bounding_box"]
        crop = image.crop((int(bb["x1"]), int(bb["y1"]), int(bb["x2"]), int(bb["y2"])))
        if crop.width > 0 and crop.height > 0:
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            ok, result, classification_time, class_error = classifier_service.classify(
                base64.b64encode(buffer.getvalue()).decode("ascii"),
                model_name=classification_model,
                top_k=body.top_k or 3,
            )
            if ok:
                classification = result
                classification_box = {
                    "class_id": top["class_id"],
                    "class_name": top["class_name"],
                    "confidence": top["confidence"],
                }
            else:
                error_code = "classification_unavailable"
                message = class_error
    else:
        error_code = "no_detections"

    return AnalyzeResponse(
        success=True,
        detection_model=detection_model,
        classification_model=classification_model,
        num_detections=len(detections),
        detections=detections,
        detection_time_ms=detection_time,
        classification=classification,
        classification_box=classification_box,
        classification_time_ms=classification_time,
        density_estimate=density_estimate,
        error_code=error_code,
        message=message,
    )


@router.post("/recommend")
async def recommend(request: Request, body: RecommendationRequest):
    """Herbicide recommendation from the validated knowledge base."""
    service = request.app.state.services["recommendation"]
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine not loaded",
        )
    result = service.get_recommendation(**body.model_dump())
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation failed",
        )
    return result


@router.post("/explain")
async def explain(request: Request, body: ExplanationRequest):
    """Grad-CAM explanation for the detector's prediction on an image."""
    _reject_invalid(*get_request_validator().validate_detection_request(body.image_data))
    xai_service = request.app.state.services["xai"]
    if not xai_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XAI engine not loaded",
        )
    result = xai_service.explain_image(body.image_data, bool(body.include_visualization))
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )
    return result


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    services = request.app.state.services

    return {
        "status": "ok",
        "model_loaded": services["model"].is_loaded(),
        "recommendation_engine_loaded": services["recommendation"].is_loaded(),
        "xai_loaded": services["xai"].is_loaded(),
    }
