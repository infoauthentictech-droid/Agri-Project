"""
Tests for herbicide recommendation engine.

NOTE: the shipped knowledge base now contains 37 ``validation_status =
passed`` (agronomically reviewed) records and 5 ``pending`` records that were
kept unvalidated because they carry a conflict flag, no dose, or are manual
notes. The engine recommends ONLY from ``passed`` records in default (strict)
mode; ``GROUNDNUT_ALLOW_PENDING_RECORDS=1`` additionally includes the pending
(less certain) records for research previews. The tests exercise both modes.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Research/development mode: allow unconfirmed 'pending' records so the
# selection logic below remains testable until the knowledge base is validated.
# Assign (not setdefault) so this suite stays self-consistent no matter in what
# order pytest collects the other suites (backend/tests sets it to "0").
os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = "1"

from herbicide.database.recommendation_engine import (
    HerbicideRecommendationEngine,
    RecommendationInput,
    RecommendationStatus
)


def test_strict_mode_uses_only_passed_records():
    """Default (strict) mode must only use agronomically reviewed records.

    The knowledge base now contains ``validation_status == passed`` records
    (37) plus 5 that were deliberately kept ``pending`` because they carry a
    conflict flag, no dose, or are 'manual only' notes. Strict mode loads ONLY
    the passed records and recommends from them; it never mixes pending data.
    """
    print("\n=== Test: Strict Mode Uses Only Passed Records ===")

    # Temporarily remove the research opt-in flag for this test.
    old = os.environ.pop("GROUNDNUT_ALLOW_PENDING_RECORDS", None)
    try:
        engine = HerbicideRecommendationEngine(
            Path('herbicide/processed/herbicide_dataset_normalized.json'))
        assert len(engine.records) > 0, "Strict mode must load passed records"
        assert all(r.validation_status == "passed" for r in engine.records), \
            "Strict mode must not load 'pending' records"
        assert engine.pending_count == 5, "5 records must stay pending"

        input_data = RecommendationInput(
            crop='groundnut',
            weed_species='Echinochloa colonum',
            crop_growth_stage='post_emergence'
        )
        result = engine.generate_recommendation(input_data)
        print(f"Status: {result.status.value}")
        assert result.status == RecommendationStatus.SUCCESS, \
            "Strict mode must recommend from validated records"
        assert len(result.candidate_management_options) > 0
        assert result.source is not None
    finally:
        if old is not None:
            os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = old

    print("✓ Test passed")


def test_known_weed():
    """Test recommendation for a known weed species."""
    print("\n=== Test: Known Weed ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='Echinochloa colona',
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
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
    
    assert result.status == RecommendationStatus.SUCCESS, "Should succeed for known weed"
    assert result.detected_weed == 'Echinochloa colona', "Should return detected weed"
    assert len(result.candidate_management_options) > 0, "Should have management options"
    assert result.source is not None, "Should have source"
    
    print("✓ Test passed")


def test_unknown_weed():
    """Test recommendation for an unknown weed species."""
    print("\n=== Test: Unknown Weed ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='CompletelyUnknownWeedSpeciesXYZ',  # Truly unknown species
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Reason: {result.reason}")
    print(f"Warnings: {result.warnings}")
    
    assert result.status == RecommendationStatus.UNKNOWN_WEED, "Should fail for unknown weed"
    assert len(result.candidate_management_options) == 0, "Should have no management options"
    assert result.source is None, "Should have no source"
    assert "Insufficient validated information" in " ".join(result.warnings), "Should indicate insufficient information"
    
    print("✓ Test passed")


def test_missing_weed():
    """Test recommendation with missing weed information."""
    print("\n=== Test: Missing Weed ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species=None,
        weed_category=None,
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Reason: {result.reason}")
    print(f"Warnings: {result.warnings}")
    
    assert result.status == RecommendationStatus.UNKNOWN_WEED, "Should fail for missing weed"
    assert len(result.candidate_management_options) == 0, "Should have no management options"
    
    print("✓ Test passed")


def test_missing_crop_stage():
    """Test recommendation with missing crop growth stage."""
    print("\n=== Test: Missing Crop Stage ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='Echinochloa colona',
        crop_growth_stage=None  # Missing growth stage
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Detected weed: {result.detected_weed}")
    print(f"Management options: {result.candidate_management_options}")
    print(f"Application info: {result.application_information}")
    print(f"Warnings: {result.warnings}")
    
    # Should still succeed but may include all stages
    assert result.status == RecommendationStatus.SUCCESS, "Should succeed without growth stage"
    assert len(result.candidate_management_options) > 0, "Should have management options"
    
    print("✓ Test passed")


def test_unsupported_crop():
    """Test recommendation for unsupported crop."""
    print("\n=== Test: Unsupported Crop ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='wheat',  # Unsupported crop
        weed_species='Echinochloa colona',
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Reason: {result.reason}")
    print(f"Warnings: {result.warnings}")
    
    assert result.status == RecommendationStatus.UNSUPPORTED_CROP, "Should fail for unsupported crop"
    assert len(result.candidate_management_options) == 0, "Should have no management options"
    
    print("✓ Test passed")


def test_conflicting_records():
    """Test recommendation when records have conflicts."""
    print("\n=== Test: Conflicting Records ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='Cyperus difformis',  # Has conflicting Imazethapyr records
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Management options: {result.candidate_management_options}")
    print(f"Confidence: {result.confidence}")
    print(f"Warnings: {result.warnings}")
    
    # Should succeed but may have conflict warnings
    assert result.status == RecommendationStatus.SUCCESS, "Should succeed even with conflicts"
    # Check if any warnings mention conflicts
    has_conflict_warning = any('conflict' in w.lower() for w in result.warnings)
    print(f"Has conflict warning: {has_conflict_warning}")
    
    print("✓ Test passed")


def test_weed_category_only():
    """Test recommendation using weed category instead of species."""
    print("\n=== Test: Weed Category Only ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species=None,
        weed_category='grassy_weeds',
        crop_growth_stage='post_emergence'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Weed category: {result.weed_category}")
    print(f"Management options: {result.candidate_management_options}")
    
    # Should succeed with category match
    assert result.status == RecommendationStatus.SUCCESS, "Should succeed with category match"
    assert len(result.candidate_management_options) > 0, "Should have management options"
    
    print("✓ Test passed")


# Automatic unknown-species mode (the app flow — no manual selection).
def test_automatic_unknown_species_broad_spectrum():
    """Automatic mode: species unknown -> broad-spectrum (general) options.

    The app never asks the user to identify the weed; it sends
    weed_category='not_specified' and the engine must return ONLY
    general-spectrum records, clearly labelled as unidentified.
    """
    print("\n=== Test: Automatic Unknown-Species (Broad-spectrum) ===")

    engine = HerbicideRecommendationEngine(
        Path('herbicide/processed/herbicide_dataset_normalized.json'))

    input_data = RecommendationInput(
        crop='groundnut',
        weed_species=None,
        weed_category='not_specified',
    )
    result = engine.generate_recommendation(input_data)

    print(f"Status: {result.status.value}")
    print(f"Detected weed: {result.detected_weed}")
    print(f"Management options: {result.candidate_management_options}")
    print(f"Warnings: {result.warnings}")

    assert result.status == RecommendationStatus.SUCCESS
    assert len(result.candidate_management_options) > 0
    # The engine must never claim a species it cannot identify.
    assert result.detected_weed == "Unidentified weed (automatic detection)"
    assert any("could not be identified automatically" in w for w in result.warnings)

    print("✓ Test passed")


def test_automatic_unknown_species_strict_mode():
    """Automatic mode (species unknown) with validated general-spectrum records.

    Because 37 records are now ``passed`` (including general-spectrum ones),
    the automatic flow returns broad-spectrum options even in strict mode - as
    long as the image detection actually flagged a weed. The recommendation is
    clearly labelled as unidentified-species.
    """
    print("\n=== Test: Automatic Unknown-Species (Strict Mode) ===")

    old = os.environ.pop("GROUNDNUT_ALLOW_PENDING_RECORDS", None)
    try:
        engine = HerbicideRecommendationEngine(
            Path('herbicide/processed/herbicide_dataset_normalized.json'))
        input_data = RecommendationInput(
            crop='groundnut',
            weed_species=None,
            weed_category='not_specified',
        )
        result = engine.generate_recommendation(input_data)
        print(f"Status: {result.status.value}")
        assert result.status == RecommendationStatus.SUCCESS
        assert len(result.candidate_management_options) > 0
        assert any("could not be identified automatically" in w
                   for w in result.warnings)
    finally:
        if old is not None:
            os.environ["GROUNDNUT_ALLOW_PENDING_RECORDS"] = old

    print("✓ Test passed")


def test_pre_emergence_stage():
    """Test recommendation for pre-emergence application."""
    print("\n=== Test: Pre-emergence Stage ===")
    
    engine = HerbicideRecommendationEngine(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    input_data = RecommendationInput(
        crop='groundnut',
        weed_species='Amaranthus viridis',  # Known broadleaf weed
        crop_growth_stage='pre_sowing'
    )
    
    result = engine.generate_recommendation(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Application stage: {result.application_information.get('application_stage') if result.application_information else None}")
    print(f"Management options: {result.candidate_management_options}")
    
    # Should succeed with pre-emergence options
    assert result.status == RecommendationStatus.SUCCESS, "Should succeed for pre-emergence"
    
    print("✓ Test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Herbicide Recommendation Engine Tests")
    print("=" * 60)
    
    try:
        test_known_weed()
        test_unknown_weed()
        test_missing_weed()
        test_missing_crop_stage()
        test_unsupported_crop()
        test_conflicting_records()
        test_weed_category_only()
        test_automatic_unknown_species_broad_spectrum()
        test_automatic_unknown_species_strict_mode()
        test_pre_emergence_stage()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Test error: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()
