"""Herbicide Recommendation Safety Checks

Implements safety checks for herbicide recommendations to prevent unsafe
recommendations based on ML predictions.

Safety rules:
1. Species identification: distinguish confident from uncertain weed IDs
2. Crop compatibility: reject recommendations not approved for detected crop
3. Knowledge-base provenance: record source, verification date, region, product label
4. Safety constraints: never invent dosage, mixing instructions, or application intervals
5. Uncertainty handling: return "unable to recommend safely" when information is missing
6. Expert review flag: indicate recommendations need expert review
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


class SafetyStatus(str, Enum):
    """Safety check status."""
    SAFE = "safe"
    UNCERTAIN = "uncertain"
    UNSAFE = "unsafe"
    NEEDS_EXPERT_REVIEW = "needs_expert_review"


class ConfidenceLevel(str, Enum):
    """Weed identification confidence."""
    HIGH = "high"  # Confidence >= 0.8
    MEDIUM = "medium"  # Confidence >= 0.6
    LOW = "low"  # Confidence < 0.6
    UNKNOWN = "unknown"  # No confidence provided


@dataclass
class SafetyCheckResult:
    """Result of safety checks on a recommendation."""
    status: SafetyStatus
    confidence_level: ConfidenceLevel
    warnings: List[str]
    errors: List[str]
    requires_expert_review: bool
    safe_to_recommend: bool
    safety_notes: str = ""
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now().isoformat()


class RecommendationSafetyChecker:
    """Safety checker for herbicide recommendations."""

    # Approved crops for each active ingredient (simplified example)
    # In production, this should come from the knowledge base
    CROP_COMPATIBILITY: Dict[str, Set[str]] = {
        "groundnut": {
            "quizalofop-ethyl",
            "imazethapyr",
            "bentazone",
            "acifluorfen",
            "pendimethalin",
            "diclosulam",
            "oxyfluorfen",
            "imazapic",
            "quizalofop-p-ethyl",
        },
        # Add more crops as needed
    }

    # Regions where specific products are registered
    REGIONAL_REGISTRATION: Dict[str, Set[str]] = {
        "india": {
            "quizalofop-ethyl",
            "imazethapyr",
            "bentazone",
            "pendimethalin",
            "diclosulam",
            "oxyfluorfen",
        },
        "usa": {"quizalofop-ethyl", "imazethapyr"},
        # Add more regions
    }

    # Required fields for safe recommendation
    REQUIRED_FIELDS: Set[str] = {
        "crop",
        "weed_species",
        "crop_growth_stage",
        "application_category",
    }

    def __init__(self, knowledge_base_path: Optional[str] = None):
        """Initialize safety checker.

        Args:
            knowledge_base_path: Path to knowledge base for provenance tracking
        """
        self.knowledge_base_path = knowledge_base_path or str(
            PROJECT_ROOT / "herbicide" / "processed" / "herbicide_dataset_normalized.json"
        )
        self.knowledge_base_provenance = self._load_provenance()

    def _load_provenance(self) -> Dict:
        """Load knowledge base provenance information.

        Returns:
            Provenance metadata
        """
        kb_path = Path(self.knowledge_base_path)
        if kb_path.exists():
            # In production, load actual provenance from knowledge base
            return {
                "source": "herbicide/processed/herbicide_dataset_normalized.json",
                "verification_date": "2024-10-24",
                "region": "india",  # Update based on actual data
                "version": "1.0",
                "last_updated": datetime.fromtimestamp(kb_path.stat().st_mtime).isoformat(),
            }
        return {
            "source": "unknown",
            "verification_date": None,
            "region": None,
            "version": None,
            "last_updated": None,
        }

    def check_confidence_level(self, confidence: Optional[float]) -> ConfidenceLevel:
        """Determine confidence level from prediction confidence.

        Args:
            confidence: Prediction confidence (0-1)

        Returns:
            Confidence level
        """
        if confidence is None:
            return ConfidenceLevel.UNKNOWN
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def check_crop_compatibility(
        self, crop: str, active_ingredient: Optional[str], region: Optional[str]
    ) -> tuple[bool, List[str]]:
        """Check if active ingredient is approved for the crop.

        Args:
            crop: Crop type
            active_ingredient: Active ingredient name
            region: Geographic region

        Returns:
            Tuple of (is_compatible, warning_messages)
        """
        warnings = []

        if not active_ingredient:
            return False, ["No active ingredient specified"]

        if crop.lower() not in self.CROP_COMPATIBILITY:
            warnings.append(f"Crop '{crop}' not in compatibility database")
            return False, warnings

        approved_ingredients = self.CROP_COMPATIBILITY[crop.lower()]
        if active_ingredient.lower() not in approved_ingredients:
            warnings.append(
                f"Active ingredient '{active_ingredient}' not approved for crop '{crop}'"
            )
            return False, warnings

        # Check regional registration
        if region and region.lower() in self.REGIONAL_REGISTRATION:
            regional_ingredients = self.REGIONAL_REGISTRATION[region.lower()]
            if active_ingredient.lower() not in regional_ingredients:
                warnings.append(
                    f"Active ingredient '{active_ingredient}' not registered in region '{region}'"
                )
                return False, warnings

        return True, warnings

    def check_required_fields(self, recommendation_input: Dict) -> tuple[bool, List[str]]:
        """Check that all required fields are present.

        Args:
            recommendation_input: Input dictionary

        Returns:
            Tuple of (has_all_fields, missing_fields)
        """
        missing = []
        for field in self.REQUIRED_FIELDS:
            if field not in recommendation_input or not recommendation_input[field]:
                missing.append(field)

        return len(missing) == 0, missing

    def check_uncertainty_handling(
        self, confidence: Optional[float], confidence_level: ConfidenceLevel
    ) -> tuple[bool, List[str]]:
        """Check if uncertainty is handled appropriately.

        Args:
            confidence: Prediction confidence
            confidence_level: Confidence level

        Returns:
            Tuple of (safe_to_proceed, warnings)
        """
        warnings = []

        if confidence_level == ConfidenceLevel.UNKNOWN:
            warnings.append("No confidence score provided - uncertain prediction")
            return False, warnings

        if confidence_level == ConfidenceLevel.LOW:
            warnings.append(f"Low confidence ({confidence:.2f}) - recommendation may be unreliable")
            return False, warnings

        if confidence_level == ConfidenceLevel.MEDIUM:
            warnings.append(f"Medium confidence ({confidence:.2f}) - use with caution")
            return True, warnings

        return True, warnings

    def check_safety_constraints(
        self, application_information: Optional[Dict]
    ) -> tuple[bool, List[str]]:
        """Check that safety constraints are not violated.

        Args:
            application_information: Application information from recommendation

        Returns:
            Tuple of (safe, warnings)
        """
        warnings = []

        if not application_information:
            return False, ["No application information provided"]

        # Check for dosage information (accept dose_ai_per_ha or dose_formulation_per_ha)
        dose_ai = application_information.get("dose_ai_per_ha")
        dose_formulation = application_information.get("dose_formulation_per_ha")

        if not dose_ai and not dose_formulation:
            warnings.append("Dosage information missing - cannot recommend safely")
            return False, warnings

        # Check for invented mixing instructions
        if not application_information.get("application_method"):
            warnings.append("Application method missing - cannot recommend safely")
            return False, warnings

        # Check for application interval (accept PHI or recommended_application_time)
        phi = application_information.get("phi_waiting_period_days")
        app_time = application_information.get("recommended_application_time")

        if not phi and not app_time:
            warnings.append("Application timing information missing - cannot recommend safely")
            return False, warnings

        # If PHI is null but we have app_time, allow with warning
        if not phi and app_time:
            warnings.append("PHI waiting period not specified, but application timing provided")

        return True, warnings

    def perform_safety_checks(
        self,
        recommendation_input: Dict,
        recommendation_result: Dict,
        detection_confidence: Optional[float] = None,
    ) -> SafetyCheckResult:
        """Perform all safety checks on a recommendation.

        Args:
            recommendation_input: Input parameters for recommendation
            recommendation_result: Result from recommendation engine
            detection_confidence: Confidence of weed detection

        Returns:
            Safety check result
        """
        warnings = []
        errors = []
        requires_expert_review = False

        # Check confidence level
        confidence_level = self.check_confidence_level(detection_confidence)
        if confidence_level == ConfidenceLevel.LOW:
            requires_expert_review = True

        # Check required fields
        has_all_fields, missing_fields = self.check_required_fields(recommendation_input)
        if not has_all_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")

        # Check uncertainty handling
        safe_uncertainty, uncertainty_warnings = self.check_uncertainty_handling(
            detection_confidence, confidence_level
        )
        warnings.extend(uncertainty_warnings)
        if not safe_uncertainty:
            requires_expert_review = True

        # Check crop compatibility
        if recommendation_result.get("active_ingredient"):
            crop = recommendation_input.get("crop", "")
            active_ingredient = recommendation_result["active_ingredient"]
            region = recommendation_input.get("region")

            compatible, compat_warnings = self.check_crop_compatibility(
                crop, active_ingredient, region
            )
            warnings.extend(compat_warnings)
            if not compatible:
                errors.append(f"Crop compatibility check failed for {active_ingredient}")

        # Check safety constraints
        app_info = recommendation_result.get("application_information")
        safe_constraints, constraint_warnings = self.check_safety_constraints(app_info)
        warnings.extend(constraint_warnings)
        if not safe_constraints:
            errors.append("Safety constraints not met")

        # Determine overall status
        if errors:
            status = SafetyStatus.UNSAFE
            safe_to_recommend = False
        elif requires_expert_review:
            status = SafetyStatus.NEEDS_EXPERT_REVIEW
            safe_to_recommend = False
        elif warnings:
            status = SafetyStatus.UNCERTAIN
            safe_to_recommend = True  # Allow with warnings
        else:
            status = SafetyStatus.SAFE
            safe_to_recommend = True

        # Build safety notes
        safety_notes = self._build_safety_notes(
            status, confidence_level, warnings, errors, self.knowledge_base_provenance
        )

        return SafetyCheckResult(
            status=status,
            confidence_level=confidence_level,
            warnings=warnings,
            errors=errors,
            requires_expert_review=requires_expert_review,
            safe_to_recommend=safe_to_recommend,
            safety_notes=safety_notes,
        )

    def _build_safety_notes(
        self,
        status: SafetyStatus,
        confidence_level: ConfidenceLevel,
        warnings: List[str],
        errors: List[str],
        provenance: Dict,
    ) -> str:
        """Build human-readable safety notes.

        Args:
            status: Safety status
            confidence_level: Confidence level
            warnings: Warning messages
            errors: Error messages
            provenance: Knowledge base provenance

        Returns:
            Safety notes string
        """
        notes = []

        notes.append(f"Safety Status: {status.value}")
        notes.append(f"Confidence Level: {confidence_level.value}")

        if warnings:
            notes.append("\nWarnings:")
            for warning in warnings:
                notes.append(f"  - {warning}")

        if errors:
            notes.append("\nErrors:")
            for error in errors:
                notes.append(f"  - {error}")

        notes.append("\nKnowledge Base Provenance:")
        notes.append(f"  Source: {provenance.get('source', 'unknown')}")
        notes.append(f"  Verification Date: {provenance.get('verification_date', 'unknown')}")
        notes.append(f"  Region: {provenance.get('region', 'unknown')}")
        notes.append(f"  Version: {provenance.get('version', 'unknown')}")

        if status in [SafetyStatus.UNSAFE, SafetyStatus.NEEDS_EXPERT_REVIEW]:
            notes.append("\n⚠️  IMPORTANT: This recommendation requires expert review before use.")
            notes.append("Consult a qualified agronomist or agricultural specialist.")

        return "\n".join(notes)

    def add_safety_metadata_to_response(
        self, recommendation_result: Dict, safety_check: SafetyCheckResult
    ) -> Dict:
        """Add safety metadata to recommendation response.

        Args:
            recommendation_result: Original recommendation result
            safety_check: Safety check result

        Returns:
            Enhanced recommendation result with safety metadata
        """
        result = recommendation_result.copy()
        result["safety"] = {
            "status": safety_check.status.value,
            "confidence_level": safety_check.confidence_level.value,
            "safe_to_recommend": safety_check.safe_to_recommend,
            "requires_expert_review": safety_check.requires_expert_review,
            "warnings": safety_check.warnings,
            "errors": safety_check.errors,
            "safety_notes": safety_check.safety_notes,
            "checked_at": safety_check.checked_at,
        }

        # Override success flag if unsafe
        if not safety_check.safe_to_recommend:
            result["success"] = False
            result["status"] = "unsafe"

        return result


# Global checker instance
_checker: Optional[RecommendationSafetyChecker] = None


def get_safety_checker() -> RecommendationSafetyChecker:
    """Get the global recommendation safety checker.

    Returns:
        RecommendationSafetyChecker instance
    """
    global _checker
    if _checker is None:
        _checker = RecommendationSafetyChecker()
    return _checker
