"""Classification / analyze schemas (multi-model inference)."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ClassifyRequest(BaseModel):
    """Single-image classification request."""
    image_data: str = Field(..., description="Base64 encoded image data")
    model_name: Optional[str] = Field(None, description="Classifier to use, e.g. mobilenet")
    top_k: Optional[int] = Field(3, ge=1, le=5, description="Number of top classes to return")


class ProbabilityItem(BaseModel):
    """One class probability."""
    class_id: int
    class_name: str
    probability: float


class ClassificationResult(BaseModel):
    """Results of one classification pass."""
    model: str
    architecture: str
    class_name: str
    class_id: int
    confidence: float
    probabilities: List[ProbabilityItem]


class ClassifyResponse(BaseModel):
    """Classification response."""
    success: bool
    result: Optional[ClassificationResult] = None
    inference_time_ms: float = 0.0
    message: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Combined detection + classification request.

    The client picks one detection model and one classification model; the
    backend runs detection on the full image and classification on the top
    detection region (crop) so the two tasks share a single upload.
    """
    image_data: str = Field(..., description="Base64 encoded image data")
    detection_model: Optional[str] = Field(None, description="Detection variant, e.g. yolov8n")
    classification_model: Optional[str] = Field(None, description="Classifier, e.g. mobilenet")
    confidence_threshold: Optional[float] = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(3, ge=1, le=5)


class DensityEstimate(BaseModel):
    """Area-based density estimate for the detector's configured weed class.

    Intentionally marked ``calibrated=False``: the bounding-box area coverage
    has NOT been calibrated against field observations, and the class the
    estimate is computed over still has unresolved semantic meaning until the
    detector label mapping is confirmed.
    """
    weed_class_ids: List[int]
    weed_count: int
    coverage_percent: float
    density_category: str  # low | medium | high
    calibrated: bool = False
    note: str = ""


class ClassificationBox(BaseModel):
    """The detection box whose crop was passed to the classifier.

    Lets the UI show the classification as belonging to one specific detected
    object (the top-confidence box) rather than appearing to be a global label
    that can contradict the list of all detections.
    """
    class_id: int
    class_name: str
    confidence: float


class AnalyzeResponse(BaseModel):
    """Response of the combined analyze endpoint."""
    success: bool
    detection_model: str
    classification_model: str
    num_detections: int
    detections: List[dict]
    detection_time_ms: float
    classification: Optional[ClassificationResult] = None
    # The detection box the classification was computed from (top-confidence
    # box). Null when no classification was produced.
    classification_box: Optional[ClassificationBox] = None
    classification_time_ms: float = 0.0
    message: Optional[str] = None
    # Machine-readable failure kind: "not_plant_image" | "no_detections" |
    # "classification_unavailable" | None. The UI uses it to show the right
    # exception screen (e.g. retake/upload for non-plant images).
    error_code: Optional[str] = None
    # Area-based density estimate (uncalibrated) for the weed-class boxes.
    density_estimate: Optional[DensityEstimate] = None
    # Honest detector-label status: until the class_0/class_1 mapping is
    # confirmed the app must not claim it "detected a weed" or "identified
    # groundnut" from the detector alone.
    labels_confirmed: bool = False
    label_note: Optional[str] = None