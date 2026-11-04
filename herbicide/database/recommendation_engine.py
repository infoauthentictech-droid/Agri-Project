"""
Herbicide decision-support engine.

This module provides herbicide recommendations based on validated knowledge base data.
It does NOT invent herbicide names, active ingredients, doses, or application rates.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from herbicide.schemas.herbicide_schema import (
    HerbicideRecord,
    SUPPORTED_CROPS,
    COMMON_WEED_SPECIES,
    ConfidenceLevel,
)


#: Weed categories that mean "the exact species is not known". The automatic
#: app flow sends one of these when the detector/classifier flagged a weed but
#: could not identify the species; the engine then returns ONLY broad-spectrum
#: (general) options — never an invented species-specific product.
AUTO_UNKNOWN_CATEGORIES = frozenset({"not_specified", "unknown", "general"})

#: Record categories considered "general weed spectrum" — the only honest
#: matches for an unidentified weed.
GENERAL_SPECTRUM_CATEGORIES = frozenset({
    "not_specified", "unknown", "general",
    "grasses_broadleaves", "grasses_broadleaves_sedges",
})

#: Species free-text markers that identify a general-spectrum record even when
#: its category field is specific-looking.
GENERAL_SPECTRUM_KEYWORDS = ("general", "not specified", "spectrum")


class RecommendationStatus(Enum):
    """Status of recommendation generation."""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    UNSUPPORTED_CROP = "unsupported_crop"
    UNKNOWN_WEED = "unknown_weed"
    NO_VALID_RECORDS = "no_valid_records"


@dataclass
class RecommendationInput:
    """Input parameters for recommendation engine."""
    crop: str
    weed_species: Optional[str] = None
    weed_category: Optional[str] = None
    crop_growth_stage: Optional[str] = None
    season: Optional[str] = None
    region: Optional[str] = None
    weed_density: Optional[str] = None
    application_category: Optional[str] = None  # Pre-sowing / Pre-emergence / Post-emergence


@dataclass
class RecommendationOutput:
    """Output from recommendation engine."""
    detected_weed: Optional[str]
    weed_category: Optional[str]
    candidate_management_options: List[str]
    active_ingredient: Optional[str]
    application_information: Optional[Dict]
    reason: str
    source: Optional[str]
    warnings: List[str]
    status: RecommendationStatus
    confidence: Optional[str] = None


class HerbicideRecommendationEngine:
    """
    Decision-support engine for herbicide recommendations.
    
    This engine:
    - Validates detected weed against knowledge base
    - Confirms crop is supported (groundnut)
    - Searches validated knowledge base
    - Filters incompatible records
    - Matches weed species/category
    - Considers crop growth stage, season, region if available
    - Ranks valid candidate records by confidence
    - Returns source-backed information only
    - Returns warnings when information is insufficient
    """
    
    def __init__(self, knowledge_base_path: Path):
        """
        Initialize recommendation engine with knowledge base.
        
        Args:
            knowledge_base_path: Path to normalized knowledge base JSON file
        """
        self.knowledge_base_path = knowledge_base_path
        self.records: List[HerbicideRecord] = []
        self.load_knowledge_base()
    
    def load_knowledge_base(self) -> None:
        """Load the normalized knowledge base.

        Only records whose ``validation_status`` is ``passed`` are used to
        give recommendations. ``pending`` records are research-stage data that
        has NOT been reviewed/agronomically confirmed, so recommending
        herbicide from them is unsafe. For development/testing you may opt in
        to pending records with ``GROUNDNUT_ALLOW_PENDING_RECORDS=1``; every
        such hit then also emits a warning.
        """
        import json

        allowed = ["passed"]
        self.include_pending = False
        if os.environ.get("GROUNDNUT_ALLOW_PENDING_RECORDS", "0") == "1":
            allowed.append("pending")
            self.include_pending = True

        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to HerbicideRecord objects
        self.records = [HerbicideRecord(**record) for record in data]

        # Filter only validated ("passed") records. Pending records are only
        # included when explicitly opted in via the env flag above.
        self.records = [r for r in self.records if r.validation_status in allowed]
        skipped = sum(1 for r in data if r.get("validation_status") not in allowed)
        self.pending_count = skipped

        print(f"Loaded {len(self.records)} validated records from knowledge base "
              f"({skipped} skipped: not 'passed')")
        if self.include_pending:
            print("WARNING: GROUNDNUT_ALLOW_PENDING_RECORDS=1 is set - pending "
                  "(unconfirmed) records are being used for recommendations.")

    def filter_by_region_season(self, records: List[HerbicideRecord], region: Optional[str],
                                season: Optional[str]) -> List[HerbicideRecord]:
        """Filter candidates by region/season when both record field and input exist.

        A record with no ``region``/``season`` field is treated as location- and
        season-agnostic and therefore kept; a record whose field does NOT match
        the provided input is dropped.
        """
        if not region and not season:
            return records
        filtered = []
        for r in records:
            if region and r.region and r.region.strip() != "":
                if region.strip().lower() not in r.region.strip().lower():
                    continue
            if season and r.season and r.season.strip() != "":
                if season.strip().lower() not in r.season.strip().lower():
                    continue
            filtered.append(r)
        return filtered
    
    def validate_crop(self, crop: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that crop is supported.
        
        Args:
            crop: Crop name
        
        Returns:
            Tuple of (is_valid, normalized_crop)
        """
        crop_normalized = crop.strip().lower()
        
        # Normalize crop names
        crop_mapping = {
            'groundnut': 'groundnut',
            'arachis hypogaea': 'groundnut',
            'peanut': 'groundnut',
        }
        
        normalized = crop_mapping.get(crop_normalized, crop_normalized)
        
        return normalized in SUPPORTED_CROPS, normalized
    
    def validate_weed(self, weed_species: Optional[str], weed_category: Optional[str]) -> Tuple[bool, str]:
        """
        Validate weed against knowledge base.
        
        Args:
            weed_species: Specific weed species
            weed_category: Weed category
        
        Returns:
            Tuple of (is_valid, validation_reason)
        """
        # If weed species is provided, check if it's in common list
        if weed_species:
            weed_species_lower = weed_species.strip().lower()
            
            # Check if it's a known species (strict match)
            for known_species in COMMON_WEED_SPECIES:
                if known_species.lower() == weed_species_lower or weed_species_lower == known_species.lower():
                    return True, f"Known weed species: {known_species}"
            
            # Check if it's a general description (NOT acceptable for specific recommendations)
            general_descriptions = [
                'grass', 'broadleaf', 'sedge', 'general', 'weed',
                'grassy weeds', 'broad leaves', 'broadleaf weeds'
            ]
            for desc in general_descriptions:
                if weed_species_lower == desc:
                    return False, f"General weed description not specific enough: {weed_species}"
            
            return False, f"Weed species not in knowledge base: {weed_species}"
        
        # Automatic unknown-species mode (app flow): the detector flagged a
        # weed but the models cannot identify the exact species. An explicit
        # broad-spectrum category is accepted so the engine can return
        # general-spectrum records instead of blocking or inventing a species.
        if not weed_species and weed_category:
            cat = weed_category.strip().lower()
            if cat in AUTO_UNKNOWN_CATEGORIES:
                return True, \
                    f"Automatic mode: species unknown; broad-spectrum request ({cat})"

        # If only category is provided, check if it's valid
        if weed_category:
            valid_categories = [
                'grassy_weeds', 'broadleaf_weeds', 'sedges',
                'grasses_broadleaves', 'grasses_broadleaves_sedges'
            ]
            if weed_category.lower() in valid_categories:
                return True, f"Valid weed category: {weed_category}"
            return False, f"Invalid weed category: {weed_category}"
        
        return False, "No weed information provided"
    
    def filter_by_crop(self, records: List[HerbicideRecord], crop: str) -> List[HerbicideRecord]:
        """
        Filter records by crop.
        
        Args:
            records: List of herbicide records
            crop: Normalized crop name
        
        Returns:
            Filtered records
        """
        return [r for r in records if r.crop == crop]
    
    def filter_by_weed(self, records: List[HerbicideRecord], weed_species: Optional[str], weed_category: Optional[str]) -> List[HerbicideRecord]:
        """
        Filter records by weed species or category.
        
        Args:
            records: List of herbicide records
            weed_species: Specific weed species
            weed_category: Weed category
        
        Returns:
            Filtered records
        """
        if not weed_species and not weed_category:
            return records
        
        filtered = []

        # Automatic unknown-species mode: keep only records that honestly
        # address the general weed spectrum — never a species-specific claim
        # for a weed whose species is unknown.
        if weed_category and not weed_species:
            cat = weed_category.strip().lower()
            if cat in AUTO_UNKNOWN_CATEGORIES:
                for record in records:
                    rec_cat = record.weed_category.strip().lower()
                    rec_sp = record.weed_species.strip().lower()
                    if rec_cat in GENERAL_SPECTRUM_CATEGORIES:
                        filtered.append(record)
                    elif any(k in rec_sp for k in GENERAL_SPECTRUM_KEYWORDS):
                        filtered.append(record)
                return filtered
        
        for record in records:
            record_weed = record.weed_species.lower()
            
            # Match by species
            if weed_species:
                if weed_species.lower() in record_weed:
                    filtered.append(record)
                    continue
            
            # Match by category
            if weed_category:
                if weed_category.lower() == record.weed_category.lower():
                    filtered.append(record)
                    continue
            
            # If record has general weed description and we have specific info, include it
            if 'general' in record_weed or 'not specified' in record_weed:
                filtered.append(record)
        
        return filtered
    
    def filter_by_growth_stage(self, records: List[HerbicideRecord], growth_stage: Optional[str], 
                                application_category: Optional[str] = None) -> List[HerbicideRecord]:
        """
        Filter records by crop growth stage and application category if available.
        
        Args:
            records: List of herbicide records
            growth_stage: Crop growth stage
            application_category: Application category (pre-sowing / pre-emergence / post-emergence)
        
        Returns:
            Filtered records
        """
        if not growth_stage and not application_category:
            return records
        
        filtered = records.copy()
        
        # Filter by application category if provided
        if application_category:
            app_cat_lower = application_category.lower()
            filtered = [r for r in filtered if r.application_category and 
                       r.application_category.lower() == app_cat_lower]
        
        # Filter by growth stage if provided
        if growth_stage:
            growth_stage_lower = growth_stage.lower()
            
            # Map growth stages to application stages
            stage_mapping = {
                'pre_sowing': 'pre_sowing_soil_incorporation',
                'pre_sowing_soil_incorporation': 'pre_sowing_soil_incorporation',
                'emergence': 'pre_emergence',
                'pre_emergence': 'pre_emergence',
                'post_emergence': 'post_emergence',
                'early': 'post_emergence',
                'late': 'late_post_emergence',
                'late_post_emergence': 'late_post_emergence',
            }
            
            target_stage = stage_mapping.get(growth_stage_lower)
            
            if target_stage:
                filtered = [r for r in filtered if target_stage in r.application_stage]
        
        return filtered
    
    def rank_candidates(self, records: List[HerbicideRecord]) -> List[HerbicideRecord]:
        """
        Rank candidate records by confidence level.
        
        Args:
            records: List of herbicide records
        
        Returns:
            Ranked records (highest confidence first)
        """
        confidence_order = {
            ConfidenceLevel.OFFICIAL_REGULATORY.value: 0,
            ConfidenceLevel.RESEARCH_BULLETIN.value: 1,
            ConfidenceLevel.STATE_UNIVERSITY_EXTENSION.value: 2,
            ConfidenceLevel.UNKNOWN.value: 3,
        }
        
        # Sort by confidence, then by conflict flag (non-conflicting first)
        return sorted(
            records,
            key=lambda r: (
                confidence_order.get(r.confidence, 99),
                1 if r.conflict_flag else 0
            )
        )
    
    def generate_recommendation(self, input_data: RecommendationInput) -> RecommendationOutput:
        """
        Generate herbicide recommendation based on input parameters.
        
        Args:
            input_data: RecommendationInput with crop, weed, and other parameters
        
        Returns:
            RecommendationOutput with recommendation or error message
        """
        warnings = []
        
        # Automatic unknown-species mode: the app detected "a weed" but could
        # not identify the species. The engine then returns broad-spectrum
        # options and clearly labels them as such.
        auto_unknown = (
            not input_data.weed_species
            and input_data.weed_category
            and input_data.weed_category.strip().lower() in AUTO_UNKNOWN_CATEGORIES
        )
        
        # Step 1: Validate crop
        crop_valid, normalized_crop = self.validate_crop(input_data.crop)
        if not crop_valid:
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason=f"Unsupported crop: {input_data.crop}",
                source=None,
                warnings=[f"Unsupported crop: {input_data.crop}"],
                status=RecommendationStatus.UNSUPPORTED_CROP
            )
        
        # The knowledge base may simply not be reviewed yet: every shipped record
        # is 'pending'. That is a true, distinct state — say it clearly instead
        # of a generic "no records matched" message.
        if not self.records:
            pending_note = (
                f" ({self.pending_count} record(s) awaiting agronomic review)"
                if self.pending_count else ""
            )
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason=(
                    "No validated herbicide records are available yet"
                    f"{pending_note}. Start the backend with "
                    "GROUNDNUT_ALLOW_PENDING_RECORDS=1 to preview draft "
                    "broad-spectrum options."
                ),
                source=None,
                warnings=[
                    "Herbicide knowledge base has not been agronomically "
                    "reviewed yet; no 'passed' records exist."
                ],
                status=RecommendationStatus.NO_VALID_RECORDS
            )

        # Step 2: Validate weed
        weed_valid, weed_reason = self.validate_weed(input_data.weed_species, input_data.weed_category)
        if not weed_valid:
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason=weed_reason,
                source=None,
                warnings=[weed_reason, "Insufficient validated information for a recommendation"],
                status=RecommendationStatus.UNKNOWN_WEED
            )
        
        # Step 3: Search knowledge base
        candidates = self.records.copy()
        
        # Step 4: Filter by crop
        candidates = self.filter_by_crop(candidates, normalized_crop)
        if not candidates:
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason=f"No records found for crop: {normalized_crop}",
                source=None,
                warnings=[f"No records found for crop: {normalized_crop}"],
                status=RecommendationStatus.NO_VALID_RECORDS
            )
        
        # Step 5: Filter by weed
        candidates = self.filter_by_weed(candidates, input_data.weed_species, input_data.weed_category)
        if not candidates:
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason=f"No records found for weed: {input_data.weed_species or input_data.weed_category}",
                source=None,
                warnings=[f"No records found for weed: {input_data.weed_species or input_data.weed_category}"],
                status=RecommendationStatus.NO_VALID_RECORDS
            )
        
        # Step 6: Filter by growth stage and application category if available
        if input_data.crop_growth_stage or input_data.application_category:
            candidates = self.filter_by_growth_stage(
                candidates, 
                input_data.crop_growth_stage,
                input_data.application_category
            )
            if not candidates:
                stage_msg = input_data.crop_growth_stage or input_data.application_category
                warnings.append(f"No records match growth stage/application: {stage_msg}")
                # Don't return error - continue with all candidates as fallback
                candidates = self.filter_by_weed(
                    self.filter_by_crop(self.records, normalized_crop),
                    input_data.weed_species,
                    input_data.weed_category
                )
        
        # Step 7: Filter by season/region (real filtering — records carrying a
        # season/region field must match; records without such fields are
        # location/season agnostic and are kept).
        if input_data.season or input_data.region:
            candidates = self.filter_by_region_season(
                candidates, input_data.region, input_data.season
            )
            if not candidates:
                return RecommendationOutput(
                    detected_weed=input_data.weed_species,
                    weed_category=input_data.weed_category,
                    candidate_management_options=[],
                    active_ingredient=None,
                    application_information=None,
                    reason="No validated records match the requested region/season.",
                    source=None,
                    warnings=["No validated records match the requested region/season."],
                    status=RecommendationStatus.NO_VALID_RECORDS
                )

        # Weed density is accepted by the API but is not (yet) a knowledge-base
        # selection field. It stays advisory so the caller and the user can see
        # that it did not drive the choice.
        if input_data.weed_density:
            warnings.append(
                f"Weed density '{input_data.weed_density}' is advisory only; "
                "density thresholds are not field-calibrated and did not "
                "change the product selection."
            )

        if not candidates:
            return RecommendationOutput(
                detected_weed=input_data.weed_species,
                weed_category=input_data.weed_category,
                candidate_management_options=[],
                active_ingredient=None,
                application_information=None,
                reason="No validated records matched the provided inputs.",
                source=None,
                warnings=["No validated records matched the provided inputs."],
                status=RecommendationStatus.NO_VALID_RECORDS
            )
        
        # Step 8: Rank candidates by confidence
        ranked_candidates = self.rank_candidates(candidates)
        
        # Step 9: Return best candidate
        best_candidate = ranked_candidates[0]
        
        # If pending records were opted in, always surface that the set is not
        # fully validated.
        if getattr(self, "include_pending", False):
            warnings.append("Knowledge base includes unconfirmed 'pending' records.")

        if auto_unknown:
            warnings.append(
                "Weed species could not be identified automatically from the "
                "image; this is a broad-spectrum (general) recommendation. "
                "Confirm the actual weed species with an agronomist before "
                "applying any herbicide."
            )

        if best_candidate.conflict_flag:
            warnings.append(f"Record has conflict flag: {best_candidate.notes}")
        
        # Build output
        management_options = []
        if best_candidate.herbicide_common_name:
            management_options.append(best_candidate.herbicide_common_name)
        if best_candidate.trade_names:
            management_options.append(f"Trade names: {best_candidate.trade_names}")
        
        application_info = {
            'application_stage': best_candidate.application_stage,
            'application_category': best_candidate.application_category,
            'recommended_application_time': best_candidate.recommended_application_time,
            'dose_ai_per_ha': best_candidate.dose_ai_per_ha,
            'dose_formulation_per_ha': best_candidate.dose_formulation_per_ha,
            'application_method': best_candidate.application_method,
            'phi_waiting_period_days': best_candidate.phi_waiting_period_days,
            'recommendation_explanation': best_candidate.recommendation_explanation,
        }
        
        # Add warnings for missing critical information
        if not best_candidate.phi_waiting_period_days:
            warnings.append("PHI (pre-harvest interval) not specified")
        
        if not best_candidate.dose_ai_per_ha and not best_candidate.dose_formulation_per_ha:
            warnings.append("Application dose not specified")
        
        return RecommendationOutput(
            detected_weed=(
                "Unidentified weed (automatic detection)" if auto_unknown
                else input_data.weed_species
            ),
            weed_category=best_candidate.weed_category,
            candidate_management_options=management_options,
            active_ingredient=best_candidate.active_ingredient,
            application_information=application_info,
            reason=f"Matched {len(candidates)} candidate(s) in knowledge base",
            source=best_candidate.source,
            warnings=warnings,
            status=RecommendationStatus.SUCCESS,
            confidence=best_candidate.confidence
        )


if __name__ == '__main__':
    # Example usage
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    # Test with known weed
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='Echinochloa colona',
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print("Recommendation Result:")
    print(f"Status: {result.status.value}")
    print(f"Detected weed: {result.detected_weed}")
    print(f"Weed category: {result.weed_category}")
    print(f"Management options: {result.candidate_management_options}")
    print(f"Active ingredient: {result.active_ingredient}")
    print(f"Application info: {result.application_information}")
    print(f"Reason: {result.reason}")
    print(f"Source: {result.source}")
    print(f"Confidence: {result.confidence}")
    print(f"Warnings: {result.warnings}")
