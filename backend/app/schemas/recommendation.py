"""
Recommendation schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class RecommendationRequest(BaseModel):
    """Recommendation request schema."""
    crop: str = Field(..., description="Crop type (e.g., 'groundnut')")
    weed_species: Optional[str] = Field(None, description="Specific weed species")
    weed_category: Optional[str] = Field(None, description="Weed category")
    crop_growth_stage: Optional[str] = Field(None, description="Crop growth stage")
    season: Optional[str] = Field(None, description="Season")
    region: Optional[str] = Field(None, description="Geographic region")
    weed_density: Optional[str] = Field(None, description="Weed density level")
    application_category: Optional[str] = Field(None, description="Application category: pre-sowing, pre-emergence, or post-emergence")


class ApplicationInformation(BaseModel):
    """Application information schema."""
    application_stage: Optional[str]
    application_category: Optional[str]
    recommended_application_time: Optional[str]
    dose_ai_per_ha: Optional[str]
    dose_formulation_per_ha: Optional[str]
    application_method: Optional[str]
    phi_waiting_period_days: Optional[int]
    recommendation_explanation: Optional[str]


class RecommendationResponse(BaseModel):
    """Recommendation response schema."""
    success: bool
    detected_weed: Optional[str]
    weed_category: Optional[str]
    candidate_management_options: List[str]
    active_ingredient: Optional[str]
    application_information: Optional[ApplicationInformation]
    reason: str
    source: Optional[str]
    warnings: List[str]
    confidence: Optional[str]
    status: str
