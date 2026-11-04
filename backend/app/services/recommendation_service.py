"""
Recommendation engine service with safety checks.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from herbicide.database.recommendation_engine import (
    HerbicideRecommendationEngine,
    RecommendationInput
)
from app.utils.recommendation_safety import get_safety_checker

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for herbicide recommendations with safety checks."""

    def __init__(self, knowledge_base_path: Optional[str] = None, enable_safety_checks: bool = True):
        """
        Initialize recommendation service.

        Args:
            knowledge_base_path: Path to normalized knowledge base JSON
            enable_safety_checks: Whether to enable safety checks
        """
        self.engine = None
        self.enable_safety_checks = enable_safety_checks
        self.safety_checker = get_safety_checker() if enable_safety_checks else None
        # Resolve the knowledge base relative to the project root so the engine
        # works regardless of the current working directory at launch time.
        self.knowledge_base_path = knowledge_base_path or str(
            project_root / 'herbicide/processed/herbicide_dataset_normalized.json'
        )
        self.load_engine()
    
    def load_engine(self) -> bool:
        """
        Load recommendation engine.
        
        Returns:
            True if engine loaded successfully, False otherwise
        """
        try:
            kb_path = Path(self.knowledge_base_path)
            if kb_path.exists():
                self.engine = HerbicideRecommendationEngine(kb_path)
                logger.info(f"Recommendation engine loaded from {self.knowledge_base_path}")
                return True
            else:
                logger.warning(f"Knowledge base not found at {self.knowledge_base_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load recommendation engine: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self.engine is not None
    
    def get_recommendation(
        self,
        crop: str,
        weed_species: Optional[str] = None,
        weed_category: Optional[str] = None,
        crop_growth_stage: Optional[str] = None,
        season: Optional[str] = None,
        region: Optional[str] = None,
        weed_density: Optional[str] = None,
        application_category: Optional[str] = None,
        detection_confidence: Optional[float] = None
    ) -> Dict:
        """
        Get herbicide recommendation with safety checks.

        Args:
            crop: Crop type
            weed_species: Specific weed species
            weed_category: Weed category
            crop_growth_stage: Crop growth stage
            season: Season
            region: Geographic region
            weed_density: Weed density level
            application_category: Application category (pre-sowing, pre-emergence, post-emergence)
            detection_confidence: Confidence of weed detection (for safety checks)

        Returns:
            Dictionary with recommendation result and safety metadata
        """
        if not self.is_loaded():
            return {
                'success': False,
                'error': 'Recommendation engine not loaded'
            }

        try:
            input_data = RecommendationInput(
                crop=crop,
                weed_species=weed_species,
                weed_category=weed_category,
                crop_growth_stage=crop_growth_stage,
                season=season,
                region=region,
                weed_density=weed_density,
                application_category=application_category
            )

            result = self.engine.generate_recommendation(input_data)

            recommendation_result = {
                'success': result.status.value == 'success',
                'detected_weed': result.detected_weed,
                'weed_category': result.weed_category,
                'candidate_management_options': result.candidate_management_options,
                'active_ingredient': result.active_ingredient,
                'application_information': result.application_information,
                'reason': result.reason,
                'source': result.source,
                'warnings': result.warnings,
                'confidence': result.confidence,
                'status': result.status.value
            }

            # Apply safety checks if enabled
            if self.enable_safety_checks and self.safety_checker:
                input_dict = {
                    'crop': crop,
                    'weed_species': weed_species,
                    'weed_category': weed_category,
                    'crop_growth_stage': crop_growth_stage,
                    'season': season,
                    'region': region,
                    'weed_density': weed_density,
                    'application_category': application_category,
                }
                safety_check = self.safety_checker.perform_safety_checks(
                    input_dict, recommendation_result, detection_confidence
                )
                recommendation_result = self.safety_checker.add_safety_metadata_to_response(
                    recommendation_result, safety_check
                )

            return recommendation_result

        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            return {
                'success': False,
                'error': f"Recommendation failed: {str(e)}"
            }
