"""
Pydantic schemas for request/response validation.
"""

from .prediction import PredictionRequest, PredictionResponse
from .detection import DetectionRequest, DetectionResponse
from .recommendation import RecommendationRequest, RecommendationResponse
from .explanation import ExplanationRequest, ExplanationResponse
from .health import HealthResponse
from .model_catalog import ModelsResponse, ModelEntry
from .analysis import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClassifyRequest,
    ClassifyResponse,
    ClassificationResult,
    ProbabilityItem,
    DensityEstimate,
)
from .segmentation import SegmentationRequest, SegmentationResponse

__all__ = [
    'PredictionRequest',
    'PredictionResponse',
    'DetectionRequest',
    'DetectionResponse',
    'RecommendationRequest',
    'RecommendationResponse',
    'ExplanationRequest',
    'ExplanationResponse',
    'HealthResponse',
    'ModelsResponse',
    'ModelEntry',
    'AnalyzeRequest',
    'AnalyzeResponse',
    'ClassifyRequest',
    'ClassifyResponse',
    'ClassificationResult',
    'ProbabilityItem',
    'DensityEstimate',
    'SegmentationRequest',
    'SegmentationResponse',
]
