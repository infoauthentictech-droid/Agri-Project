"""
Explanation schemas.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ExplanationRequest(BaseModel):
    """Explanation request schema."""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_visualization: Optional[bool] = Field(True, description="Include visualization data")


class Explanation(BaseModel):
    """Single explanation schema."""
    prediction: str
    explanation_text: str
    visualization_description: str
    limitations: List[str]
    # Which method produced the visualization: "gradcam" (genuine gradient
    # Grad-CAM) or "fallback_activation" (clearly-labelled non-gradient map).
    method: str = "gradcam"


class ExplanationResponse(BaseModel):
    """Explanation response schema."""
    success: bool
    predictions: List[dict]
    explanations: List[Explanation]
    saved_files: dict
    message: Optional[str] = None
    # Base64-encoded Grad-CAM visualizations (keys: overlay_0, heatmap_0).
    # Present only when include_visualization was true and a visualization was
    # generated for the first detected object.
    visualizations: Optional[Dict[str, str]] = None
