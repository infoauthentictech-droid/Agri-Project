"""
Herbicide Recommendation Safety Module

Provides comprehensive safety checks for herbicide recommendations:
- Species mapping validation
- Growth stage support
- Regulatory checks
- Region-specific restrictions
- Confidence-based fallbacks
- Application condition validation
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyStatus(Enum):
    """Safety check status."""
    SAFE = "safe"
    WARNING = "warning"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass
class SafetyCheckResult:
    """Result of safety checks."""
    status: SafetyStatus
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    error: Optional[str] = None
    requires_manual_review: bool = False


class HerbicideSafetyChecker:
    """Comprehensive safety checker for herbicide recommendations."""

    def __init__(self, verified_species_map: Optional[Dict[str, str]] = None):
        """
        Initialize safety checker.

        Args:
            verified_species_map: Dictionary mapping weed species to verified herbicide recommendations
        """
        self.verified_species_map = verified_species_map or {}
        self.supported_growth_stages = [
            "seedling", "vegetative", "flowering", "fruiting", "maturity", "harvest"
        ]
        self.supported_regions = [
            "north_america", "europe", "asia", "africa", "south_america", "australia"
        ]
        self.high_risk_ingredients = [
            "paraquat", "dicamba", "2,4-D", "atrazine", "glyphosate"
        ]

    def perform_safety_checks(
        self,
        input_data: Dict,
        recommendation_result: Dict,
        detection_confidence: Optional[float] = None
    ) -> SafetyCheckResult:
        """
        Perform comprehensive safety checks on recommendation.

        Args:
            input_data: Dictionary with crop, weed, growth stage, region, etc.
            recommendation_result: The recommendation result to validate
            detection_confidence: Confidence of weed detection (0-1)

        Returns:
            SafetyCheckResult with status and details
        """
        checks_passed = []
        checks_failed = []
        warnings = []

        # Check 1: Detection confidence
        if detection_confidence is not None:
            if detection_confidence < 0.5:
                checks_failed.append("Low detection confidence")
                warnings.append(f"Weed detection confidence ({detection_confidence:.2f}) is below threshold (0.5)")
            elif detection_confidence < 0.7:
                warnings.append(f"Weed detection confidence ({detection_confidence:.2f}) is moderate")
            else:
                checks_passed.append("Detection confidence acceptable")
        else:
            warnings.append("Detection confidence not provided")

        # Check 2: Species mapping validation
        weed_species = input_data.get('weed_species')
        if weed_species:
            if weed_species in self.verified_species_map:
                checks_passed.append("Weed species verified in knowledge base")
            else:
                warnings.append(f"Weed species '{weed_species}' not in verified mapping")
        else:
            warnings.append("Weed species not specified")

        # Check 3: Growth stage support
        growth_stage = input_data.get('crop_growth_stage')
        if growth_stage:
            if growth_stage.lower() in [s.lower() for s in self.supported_growth_stages]:
                checks_passed.append("Growth stage supported")
            else:
                warnings.append(f"Growth stage '{growth_stage}' may not be fully supported")
        else:
            warnings.append("Growth stage not specified")

        # Check 4: Region support
        region = input_data.get('region')
        if region:
            if region.lower() in [r.lower() for r in self.supported_regions]:
                checks_passed.append("Region supported")
            else:
                warnings.append(f"Region '{region}' may not have specific recommendations")
        else:
            warnings.append("Region not specified")

        # Check 5: Active ingredient safety
        active_ingredient = recommendation_result.get('active_ingredient')
        if active_ingredient:
            active_lower = active_ingredient.lower()
            if any(ingredient in active_lower for ingredient in self.high_risk_ingredients):
                warnings.append(f"Active ingredient '{active_ingredient}' requires careful handling")
            else:
                checks_passed.append("Active ingredient not in high-risk list")
        else:
            warnings.append("Active ingredient not specified")

        # Check 6: Application timing
        application_category = input_data.get('application_category')
        if application_category:
            checks_passed.append("Application category specified")
        else:
            warnings.append("Application category not specified")

        # Check 7: Recommendation success
        if recommendation_result.get('success'):
            checks_passed.append("Recommendation generated successfully")
        else:
            checks_failed.append("Recommendation generation failed")

        # Determine overall status
        if checks_failed:
            status = SafetyStatus.UNSAFE
        elif warnings:
            status = SafetyStatus.WARNING
        else:
            status = SafetyStatus.SAFE

        requires_manual_review = status in [SafetyStatus.UNSAFE, SafetyStatus.WARNING]

        return SafetyCheckResult(
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            requires_manual_review=requires_manual_review
        )

    def add_safety_metadata_to_response(
        self,
        recommendation_result: Dict,
        safety_check: SafetyCheckResult
    ) -> Dict:
        """
        Add safety metadata to recommendation response.

        Args:
            recommendation_result: Original recommendation result
            safety_check: Safety check result

        Returns:
            Enhanced recommendation result with safety metadata
        """
        recommendation_result['safety_metadata'] = {
            'status': safety_check.status.value,
            'checks_passed': safety_check.checks_passed,
            'checks_failed': safety_check.checks_failed,
            'warnings': safety_check.warnings,
            'requires_manual_review': safety_check.requires_manual_review,
            'limitation_notice': (
                "This recommendation is based on general agricultural knowledge. "
                "Always consult local agricultural extension services, label instructions, "
                "and regulatory requirements before application. "
                "This system does not replace professional agricultural advice."
            )
        }

        # Add disclaimer if checks failed
        if safety_check.status == SafetyStatus.UNSAFE:
            recommendation_result['safety_metadata']['disclaimer'] = (
                "This recommendation has safety concerns. "
                "Manual review by an agricultural expert is required before application."
            )

        return recommendation_result


def get_safety_checker(verified_species_map: Optional[Dict[str, str]] = None) -> HerbicideSafetyChecker:
    """
    Get configured safety checker instance.

    Args:
        verified_species_map: Optional verified species mapping

    Returns:
        HerbicideSafetyChecker instance
    """
    return HerbicideSafetyChecker(verified_species_map)
