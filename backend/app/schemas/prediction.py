"""
Prediction schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Prediction request schema."""
    image_data: str = Field(..., description="Base64 encoded image data")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")


class BoundingBox(BaseModel):
    """Bounding box schema."""
    x1: float
    y1: float
    x2: float
    y2: float


class Prediction(BaseModel):
    """Single prediction schema."""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    success: bool
    predictions: List[Prediction]
    num_predictions: int
    inference_time_ms: float
    message: Optional[str] = None
