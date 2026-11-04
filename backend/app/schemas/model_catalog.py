"""Model catalog schemas — describes available detection/classification models."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ModelEntry(BaseModel):
    """One model entry shown to the client.

    ``weights`` intentionally never carries a server-side path: exposing the
    filesystem layout of the model zoo is an information leak.

    ``accuracy`` is the model's held-out TEST-SET accuracy (from
    ``experiments/<name>/metrics.json`` or the detector run report). It is the
    honest basis for any "accuracy" claim the UI makes; it is NOT recomputed
    per image. ``None`` means no evaluation metrics are available.
    """
    name: str
    available: bool
    loaded: bool
    kind: str  # "detection" | "classification"
    family: str = ""
    weights: str = ""
    accuracy: Optional[float] = None
    accuracy_basis: str = ""


class ModelsResponse(BaseModel):
    """Registry response listing every model the client can select.
    
    Note: Only one detection model (yolov8n) is used in production for consistency.
    """
    success: bool
    detection_models: List[ModelEntry]
    classification_models: List[ModelEntry]
    default_detection_model: str
    default_classification_model: str
    message: Optional[str] = None