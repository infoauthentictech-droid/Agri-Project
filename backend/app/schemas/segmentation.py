"""
Segmentation schemas (semantic segmentation).
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class SegmentationRequest(BaseModel):
    """Segmentation request schema."""
    image_data: str = Field(..., description="Base64 encoded image data")
    target_size: Optional[int] = Field(224, ge=64, le=1024, description="Target size for segmentation")


class SegmentationResponse(BaseModel):
    """Segmentation response schema."""
    success: bool
    mask: str = Field(..., description="Base64 encoded RGB mask (background=black, crop=green, weed=red)")
    class_mask: str = Field(..., description="Base64 encoded class index mask")
    classes: List[str] = Field(..., description="List of detected class names")
    coverage: Dict[str, float] = Field(..., description="Percentage coverage per class")
    class_names: List[str] = Field(..., description="All available class names")
    inference_time_ms: float
    message: Optional[str] = None
