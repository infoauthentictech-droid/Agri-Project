"""
Detection schemas (object detection).
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class DetectionRequest(BaseModel):
    """Detection request schema."""
    image_data: str = Field(..., description="Base64 encoded image data")
    confidence_threshold: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold")
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    model_name: Optional[str] = Field(None, description="Detection model (single variant: yolov8n)")


class BoundingBox(BaseModel):
    """Bounding box schema."""
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single detection schema."""
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class DetectionResponse(BaseModel):
    """Detection response schema."""
    success: bool
    detections: List[Detection]
    num_detections: int
    inference_time_ms: float
    message: Optional[str] = None
