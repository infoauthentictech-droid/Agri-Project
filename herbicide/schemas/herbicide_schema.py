"""
Herbicide knowledge base schema definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class CropType(Enum):
    """Normalized crop types."""
    GROUNDNUT = "groundnut"
    GROUNDNUT_PIGEONPEA_INTERCROP = "groundnut_pigeonpea_intercrop"
    GROUNDNUT_SUNFLOWER_INTERCROP = "groundnut_sunflower_intercrop"
    GROUNDNUT_CASTOR_INTERCROP = "groundnut_castor_intercrop"
    UNKNOWN = "unknown"


class CroppingSystem(Enum):
    """Normalized cropping systems."""
    SOLE_CROP = "sole_crop"
    INTERCROP = "intercrop"
    SEQUENCE_ROTATION = "sequence_rotation"
    UNKNOWN = "unknown"


class WeedCategory(Enum):
    """Normalized weed categories."""
    GRASSY_WEEDS = "grassy_weeds"
    BROADLEAF_WEEDS = "broadleaf_weeds"
    SEDGES = "sedges"
    GRASSES_BROADLEAVES = "grasses_broadleaves"
    GRASSES_BROADLEAVES_SEDGES = "grasses_broadleaves_sedges"
    MONOCOT_WEEDS = "monocot_weeds"
    GENERAL = "general"
    NOT_SPECIFIED = "not_specified"
    UNKNOWN = "unknown"


class ApplicationStage(Enum):
    """Normalized application stages."""
    PRE_SOWING_SOIL_INCORPORATION = "pre_sowing_soil_incorporation"
    PRE_EMERGENCE = "pre_emergence"
    POST_EMERGENCE = "post_emergence"
    LATE_POST_EMERGENCE = "late_post_emergence"
    PRE_EARLY_POST_EMERGENCE = "pre_early_post_emergence"
    MANUAL_CONTROL = "manual_control"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Source confidence levels."""
    OFFICIAL_REGULATORY = "official_regulatory"
    RESEARCH_BULLETIN = "research_bulletin"
    STATE_UNIVERSITY_EXTENSION = "state_university_extension"
    UNKNOWN = "unknown"


@dataclass
class HerbicideRecord:
    """
    Normalized herbicide record structure.
    
    This schema maps the original dataset fields to a normalized structure
    for the herbicide knowledge base.
    """
    # Identification
    record_id: str
    
    # Crop and cropping system
    crop: str  # Normalized crop name
    cropping_system: str  # Normalized cropping system
    
    # Weed information
    weed_species: str  # Original weed species description
    weed_category: str  # Normalized weed category
    
    # Herbicide information
    herbicide_common_name: Optional[str] = None  # Normalized herbicide name
    trade_names: Optional[str] = None  # Original trade names (preserved)
    active_ingredient: Optional[str] = None  # Normalized active ingredient
    formulation: Optional[str] = None  # Normalized formulation
    
    # Application information
    application_stage: str = "unknown"  # Normalized application stage
    application_category: Optional[str] = None  # Pre-sowing / Pre-emergence / Post-emergence
    recommended_application_time: Optional[str] = None  # When to apply (e.g., "3-5 days after sowing")
    dose_ai_per_ha: Optional[str] = None  # Original dose (active ingredient)
    dose_formulation_per_ha: Optional[str] = None  # Original dose (formulation)
    application_rate_normalized: Optional[float] = None  # Normalized rate (numeric)
    unit: Optional[str] = None  # Normalized unit
    application_method: Optional[str] = None  # How to apply (e.g., "spray", "soil incorporation")
    
    # Safety and restrictions
    phi_waiting_period_days: Optional[int] = None  # Pre-harvest interval
    mode_of_action: Optional[str] = None  # Extracted from notes
    limitations: Optional[str] = None  # Extracted from notes
    crop_safety_information: Optional[str] = None  # Extracted from notes
    
    # Geographic and temporal context
    region: Optional[str] = None  # Extracted from source or notes
    season: Optional[str] = None  # Extracted from notes or source
    
    # Recommendation explanation
    recommendation_explanation: Optional[str] = None  # Why this herbicide is recommended
    
    # Source and provenance
    source: str = ""  # Original source (preserved)
    page: Optional[str] = None  # Original page reference
    confidence: str = "unknown"  # Normalized confidence level
    conflict_flag: bool = False  # Whether record has conflicts
    notes: Optional[str] = None  # Original notes (preserved)
    
    # Metadata
    normalized_at: Optional[str] = None  # Timestamp of normalization
    validation_status: str = "pending"  # Validation status


@dataclass
class ValidationResult:
    """Result of validation checks."""
    record_id: str
    has_missing_values: bool
    missing_fields: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    has_conflicts: bool = False
    conflict_details: List[str] = field(default_factory=list)
    has_invalid_references: bool = False
    invalid_references: List[str] = field(default_factory=list)
    has_inconsistent_units: bool = False
    unit_issues: List[str] = field(default_factory=list)
    unsupported_crop: bool = False
    unsupported_weed_species: bool = False
    validation_passed: bool = True
    warnings: List[str] = field(default_factory=list)


@dataclass
class FieldMapping:
    """Mapping from original CSV fields to normalized schema."""
    original_field: str
    normalized_field: str
    transformation: str = "direct"  # direct, normalize, extract, derive


# Field mappings from original CSV to normalized schema
FIELD_MAPPINGS = [
    FieldMapping("record_id", "record_id", "direct"),
    FieldMapping("crop", "crop", "normalize"),
    FieldMapping("cropping_system", "cropping_system", "normalize"),
    FieldMapping("weed_species", "weed_species", "direct"),
    FieldMapping("weed_species", "weed_category", "derive"),
    FieldMapping("herbicide_common_name", "herbicide_common_name", "normalize"),
    FieldMapping("trade_names", "trade_names", "direct"),
    FieldMapping("active_ingredient_formulation", "active_ingredient", "extract"),
    FieldMapping("active_ingredient_formulation", "formulation", "extract"),
    FieldMapping("dose_ai_per_ha", "dose_ai_per_ha", "direct"),
    FieldMapping("dose_formulation_per_ha", "dose_formulation_per_ha", "direct"),
    FieldMapping("dose_ai_per_ha", "application_rate_normalized", "parse"),
    FieldMapping("dose_formulation_per_ha", "unit", "extract"),
    FieldMapping("application_stage", "application_stage", "normalize"),
    FieldMapping("phi_waiting_period_days", "phi_waiting_period_days", "parse"),
    FieldMapping("notes", "mode_of_action", "extract"),
    FieldMapping("notes", "limitations", "extract"),
    FieldMapping("notes", "crop_safety_information", "extract"),
    FieldMapping("source", "region", "extract"),
    FieldMapping("source", "season", "extract"),
    FieldMapping("source", "source", "direct"),
    FieldMapping("page", "page", "direct"),
    FieldMapping("confidence", "confidence", "normalize"),
    FieldMapping("conflict_flag", "conflict_flag", "parse"),
    FieldMapping("notes", "notes", "direct"),
]


# Supported crops for validation
SUPPORTED_CROPS = [
    "groundnut",
    "groundnut_pigeonpea_intercrop",
    "groundnut_sunflower_intercrop",
    "groundnut_castor_intercrop",
]

# Common weed species for validation (from dataset)
COMMON_WEED_SPECIES = [
    "echinochloa spp.",
    "echinochloa colona",
    "echinochloa colonum",
    "cyperus difformis",
    "cyperus rotundus",
    "commelina benghalensis",
    "trianthema sp.",
    "trianthema portulacastrum",
    "amaranthus viridis",
    "parthenium hysterophorus",
    "euphorbia geniculata",
    "euphorbia hirta",
    "digera arvensis",
    "acalypha spp.",
    "digitaria longiflora",
    "digitaria sanguinalis",
    "digitaria spp.",
    "eleusine indica",
    "dactyloctenium aegyptium",
    "dactyloctenium sp.",
    "bracharia reptans",
    "dinebra arabica",
    "dinebra retroflexa",
    "cynodon dactylon",
]

# Valid units for validation
VALID_UNITS = [
    "kg/ha",
    "g/ha",
    "l/ha",
    "ml/ha",
    "lit/ha",
    "ltr/ha",
]
