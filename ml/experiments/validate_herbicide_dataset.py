"""Herbicide Dataset Validation

Validates the updated herbicide dataset structure and content.
Checks for required fields, data quality, and consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HerbicideDatasetValidator:
    """Validator for herbicide dataset."""

    # Required fields for safety and recommendation
    REQUIRED_FIELDS = {
        "record_id",
        "crop",
        "weed_species",
        "weed_category",
        "herbicide_common_name",
        "active_ingredient",
        "application_stage",
        "dose_ai_per_ha",
        "application_method",
        "phi_waiting_period_days",
        "region",
    }

    # Fields that can be null but should have a fallback
    FALLBACK_FIELDS = {
        "active_ingredient": "herbicide_common_name",
        "dose_ai_per_ha": "dose_formulation_per_ha",
        "phi_waiting_period_days": "recommended_application_time",
    }

    def __init__(self, dataset_path: Path):
        """Initialize validator.

        Args:
            dataset_path: Path to herbicide dataset JSON
        """
        self.dataset_path = dataset_path
        self.data = []
        self.issues = []
        self.warnings = []

    def load_dataset(self) -> bool:
        """Load herbicide dataset.

        Returns:
            True if loaded successfully
        """
        try:
            with open(self.dataset_path, "r") as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            self.issues.append(f"Failed to load dataset: {e}")
            return False

    def validate_structure(self) -> Dict:
        """Validate dataset structure.

        Returns:
            Dictionary with validation results
        """
        results = {
            "total_records": len(self.data),
            "valid_records": 0,
            "invalid_records": 0,
            "missing_fields": {},
            "null_critical_fields": {},
        }

        field_counts = Counter()
        null_field_counts = Counter()

        for i, record in enumerate(self.data):
            if not isinstance(record, dict):
                self.issues.append(f"Record {i}: Not a dictionary")
                results["invalid_records"] += 1
                continue

            # Check required fields
            for field in self.REQUIRED_FIELDS:
                field_counts[field] += 1
                if field not in record:
                    results["missing_fields"][field] = (
                        results["missing_fields"].get(field, 0) + 1
                    )
                elif record[field] is None:
                    null_field_counts[field] += 1
                    # Check if fallback exists
                    fallback = self.FALLBACK_FIELDS.get(field)
                    if fallback and fallback in record and record[fallback] is not None:
                        self.warnings.append(
                            f"Record {i}: Field '{field}' is null but fallback '{fallback}' exists"
                        )
                    else:
                        results["null_critical_fields"][field] = (
                            results["null_critical_fields"].get(field, 0) + 1
                        )

            results["valid_records"] += 1

        results["field_counts"] = dict(field_counts)
        results["null_field_counts"] = dict(null_field_counts)

        return results

    def validate_content(self) -> Dict:
        """Validate dataset content and consistency.

        Returns:
            Dictionary with content validation results
        """
        results = {
            "unique_crops": set(),
            "unique_weed_species": set(),
            "unique_herbicides": set(),
            "unique_regions": set(),
            "duplicate_record_ids": [],
            "inconsistent_validation_status": [],
        }

        record_ids = set()

        for i, record in enumerate(self.data):
            # Check for duplicate record IDs
            record_id = record.get("record_id")
            if record_id:
                if record_id in record_ids:
                    results["duplicate_record_ids"].append(record_id)
                else:
                    record_ids.add(record_id)

            # Collect unique values
            if record.get("crop"):
                results["unique_crops"].add(record["crop"])
            if record.get("weed_species"):
                results["unique_weed_species"].add(record["weed_species"])
            if record.get("herbicide_common_name"):
                results["unique_herbicides"].add(record["herbicide_common_name"])
            if record.get("region"):
                results["unique_regions"].add(record["region"])

            # Check validation status consistency
            validation_status = record.get("validation_status")
            if validation_status and validation_status != "passed":
                results["inconsistent_validation_status"].append(
                    (record.get("record_id"), validation_status)
                )

        # Convert sets to lists for JSON serialization
        results["unique_crops"] = list(results["unique_crops"])
        results["unique_weed_species"] = list(results["unique_weed_species"])
        results["unique_herbicides"] = list(results["unique_herbicides"])
        results["unique_regions"] = list(results["unique_regions"])

        return results

    def validate_safety_fields(self) -> Dict:
        """Validate safety-critical fields.

        Returns:
            Dictionary with safety validation results
        """
        results = {
            "records_with_active_ingredient": 0,
            "records_with_dose": 0,
            "records_with_phi": 0,
            "records_with_region": 0,
            "records_with_all_safety_fields": 0,
        }

        for record in self.data:
            has_active = record.get("active_ingredient") is not None
            has_dose = (
                record.get("dose_ai_per_ha") is not None
                or record.get("dose_formulation_per_ha") is not None
            )
            has_phi = (
                record.get("phi_waiting_period_days") is not None
                or record.get("recommended_application_time") is not None
            )
            has_region = record.get("region") is not None

            if has_active:
                results["records_with_active_ingredient"] += 1
            if has_dose:
                results["records_with_dose"] += 1
            if has_phi:
                results["records_with_phi"] += 1
            if has_region:
                results["records_with_region"] += 1

            if has_active and has_dose and has_phi and has_region:
                results["records_with_all_safety_fields"] += 1

        return results

    def generate_report(self) -> Dict:
        """Generate complete validation report.

        Returns:
            Dictionary with complete validation results
        """
        if not self.load_dataset():
            return {
                "success": False,
                "error": "Failed to load dataset",
                "issues": self.issues,
            }

        structure_results = self.validate_structure()
        content_results = self.validate_content()
        safety_results = self.validate_safety_fields()

        # Overall assessment
        total_records = structure_results["total_records"]
        has_critical_issues = (
            structure_results["invalid_records"] > 0
            or len(content_results.get("duplicate_record_ids", [])) > 0
        )

        # Data completeness score
        completeness_score = (
            structure_results["valid_records"] / total_records if total_records > 0 else 0
        )

        # Safety score
        safety_score = (
            safety_results["records_with_all_safety_fields"] / total_records
            if total_records > 0
            else 0
        )

        report = {
            "success": True,
            "dataset_path": str(self.dataset_path),
            "total_records": total_records,
            "completeness_score": completeness_score,
            "safety_score": safety_score,
            "structure_validation": structure_results,
            "content_validation": content_results,
            "safety_validation": safety_results,
            "issues": self.issues,
            "warnings": self.warnings,
            "has_critical_issues": has_critical_issues,
            "recommendation": (
                "Dataset has critical issues that need attention"
                if has_critical_issues
                else "Dataset structure is valid but some fields are incomplete"
                if safety_score < 0.5
                else "Dataset is valid and ready for use"
            ),
        }

        return report

    def save_report(self, report: Dict, output_path: Path) -> None:
        """Save validation report to file.

        Args:
            report: Validation report
            output_path: Path to save report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Validation report saved to {output_path}")


def main():
    """Main function to validate herbicide dataset."""
    dataset_path = PROJECT_ROOT / "herbicide" / "processed" / "herbicide_dataset_normalized.json"

    validator = HerbicideDatasetValidator(dataset_path)
    report = validator.generate_report()

    # Print summary
    print("\n" + "=" * 80)
    print("HERBICIDE DATASET VALIDATION REPORT")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Total Records: {report['total_records']}")
    print(f"Completeness Score: {report['completeness_score']:.2%}")
    print(f"Safety Score: {report['safety_score']:.2%}")
    print(f"\nRecommendation: {report['recommendation']}")
    print("=" * 80)

    if report["issues"]:
        print("\nCritical Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"][:10]:  # Show first 10
            print(f"  - {warning}")

    # Safety details
    safety = report["safety_validation"]
    print(f"\nSafety Field Coverage:")
    print(f"  Records with active ingredient: {safety['records_with_active_ingredient']}/{report['total_records']}")
    print(f"  Records with dosage: {safety['records_with_dose']}/{report['total_records']}")
    print(f"  Records with PHI/app time: {safety['records_with_phi']}/{report['total_records']}")
    print(f"  Records with region: {safety['records_with_region']}/{report['total_records']}")
    print(f"  Records with ALL safety fields: {safety['records_with_all_safety_fields']}/{report['total_records']}")

    # Save report
    output_path = PROJECT_ROOT / "experiments" / "evaluation" / "herbicide_validation_report.json"
    validator.save_report(report, output_path)


if __name__ == "__main__":
    main()
