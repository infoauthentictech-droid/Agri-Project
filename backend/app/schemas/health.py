"""
Health check schemas.
"""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    recommendation_engine_loaded: bool
    xai_loaded: bool
    version: str
