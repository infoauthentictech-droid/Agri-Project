"""
Herbicide dataset validation module.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from herbicide.schemas.herbicide_schema import (
    HerbicideRecord,
    ValidationResult,
    SUPPORTED_CROPS,
    COMMON_WEED_SPECIES,
    VALID_UNITS
)


class HerbicideValidator:
    """
    Validate herbicide dataset records for data quality issues.
    
    Identifies:
    - Missing values
    - Duplicates
    - Conflicting records
    - Invalid references
    - Inconsistent units
    - Unsupported crops
    - Unsupported weed species
    """
    
    def __init__(self, raw_data_path: Path):
        """
        Initialize validator with raw dataset.
        
        Args:
            raw_data_path: Path to raw herbicide CSV file
        """
        self.raw_data_path = raw_data_path
        self.records: List[Dict] = []
        self.validation_results: List[ValidationResult] = []
        self.load_data()
    
    def load_data(self) -> None:
        """Load raw CSV data."""
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.records = list(reader)
        
        print(f"Loaded {len(self.records)} records from {self.raw_data_path}")
    
    def check_missing_values(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Check for missing values in required fields.
        
        Args:
            record: Single record dictionary
        
        Returns:
            Tuple of (has_missing, list_of_missing_fields)
        """
        required_fields = [
            'record_id',
            'crop',
            'herbicide_common_name',
            'source'
        ]
        
        missing_fields = []
        for field in required_fields:
            value = record.get(field, '').strip()
            if not value or value.lower() in ['n/a', 'none', 'not specified', '']:
                missing_fields.append(field)
        
        return len(missing_fields) > 0, missing_fields
    
    def check_duplicates(self) -> List[Tuple[str, List[str]]]:
        """
        Check for duplicate records based on key fields.
        
        Returns:
            List of (duplicate_key, list_of_record_ids)
        """
        # Group by combination of crop + herbicide + application_stage
        key_to_ids = {}
        
        for record in self.records:
            key = f"{record.get('crop', '')}|{record.get('herbicide_common_name', '')}|{record.get('application_stage', '')}"
            record_id = record.get('record_id', '')
            
            if key not in key_to_ids:
                key_to_ids[key] = []
            key_to_ids[key].append(record_id)
        
        # Find duplicates
        duplicates = []
        for key, ids in key_to_ids.items():
            if len(ids) > 1:
                duplicates.append((key, ids))
        
        return duplicates
    
    def check_conflicts(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Check for conflicting information within a record.
        
        Args:
            record: Single record dictionary
        
        Returns:
            Tuple of (has_conflicts, list_of_conflict_details)
        """
        conflicts = []
        
        # Check conflict_flag field
        conflict_flag = record.get('conflict_flag', '').strip()
        if conflict_flag.lower() in ['yes', 'true', '1']:
            conflicts.append(f"Record marked with conflict_flag: {conflict_flag}")
        
        # Check for conflicting dose information
        dose_ai = record.get('dose_ai_per_ha', '').strip()
        dose_form = record.get('dose_formulation_per_ha', '').strip()
        
        # If both doses are present but one is empty/invalid
        if dose_ai and dose_form:
            # Check if doses seem inconsistent (very rough heuristic)
            # This is a basic check - detailed validation requires parsing units
            if 'kg' in dose_ai.lower() and 'ml' in dose_form.lower():
                conflicts.append(f"Potential unit conflict: AI dose in kg, formulation dose in ml")
        
        # Check notes for conflict indicators
        notes = record.get('notes', '').lower()
        if 'conflict' in notes or 'discrepancy' in notes or 'inconsistent' in notes:
            conflicts.append(f"Notes mention conflict/discrepancy: {record.get('notes', '')}")
        
        return len(conflicts) > 0, conflicts
    
    def check_invalid_references(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Check for invalid references (e.g., page numbers without source).
        
        Args:
            record: Single record dictionary
        
        Returns:
            Tuple of (has_invalid_refs, list_of_invalid_ref_details)
        """
        invalid_refs = []
        
        source = record.get('source', '').strip()
        page = record.get('page', '').strip()
        
        # If page is specified but source is not
        if page and not source:
            invalid_refs.append(f"Page reference '{page}' without source")
        
        # If source is generic but page is specific
        if source and page and 'n/a' in source.lower():
            invalid_refs.append(f"Page reference '{page}' with generic source '{source}'")
        
        return len(invalid_refs) > 0, invalid_refs
    
    def check_inconsistent_units(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Check for inconsistent or invalid units.
        
        Args:
            record: Single record dictionary
        
        Returns:
            Tuple of (has_inconsistent_units, list_of_unit_issues)
        """
        unit_issues = []
        
        dose_ai = record.get('dose_ai_per_ha', '').strip()
        dose_form = record.get('dose_formulation_per_ha', '').strip()
        
        # Extract units from dose fields
        def extract_unit(dose_str):
            if not dose_str:
                return None
            # Common unit patterns
            unit_patterns = [
                r'kg/ha',
                r'g/ha',
                r'l/ha',
                r'ml/ha',
                r'lit/ha',
                r'ltr/ha',
                r'EC',
                r'WDG',
                r'SL',
                r'WG',
            ]
            for pattern in unit_patterns:
                if pattern.lower() in dose_str.lower():
                    return pattern
            return None
        
        ai_unit = extract_unit(dose_ai)
        form_unit = extract_unit(dose_form)
        
        # Check if units are recognized
        if ai_unit and ai_unit not in [u.lower() for u in VALID_UNITS + ['EC', 'WDG', 'SL', 'WG']]:
            unit_issues.append(f"Unrecognized AI unit: {ai_unit}")
        
        if form_unit and form_unit not in [u.lower() for u in VALID_UNITS + ['EC', 'WDG', 'SL', 'WG']]:
            unit_issues.append(f"Unrecognized formulation unit: {form_unit}")
        
        # Check for unit inconsistencies between AI and formulation
        if ai_unit and form_unit:
            # AI should be in mass units (kg, g), formulation in volume (l, ml) or concentration
            mass_units = ['kg/ha', 'g/ha']
            volume_units = ['l/ha', 'ml/ha', 'lit/ha', 'ltr/ha']
            
            if ai_unit in volume_units and form_unit in mass_units:
                unit_issues.append(f"Unit type mismatch: AI in volume unit ({ai_unit}), formulation in mass unit ({form_unit})")
        
        return len(unit_issues) > 0, unit_issues
    
    def check_unsupported_crop(self, record: Dict) -> bool:
        """
        Check if crop is in supported list.
        
        Args:
            record: Single record dictionary
        
        Returns:
            True if crop is unsupported
        """
        crop = record.get('crop', '').strip().lower()
        
        # Normalize crop names
        crop_normalizations = {
            'groundnut': 'groundnut',
            'groundnut (sole crop)': 'groundnut',
            'groundnut + pigeonpea (intercrop)': 'groundnut_pigeonpea_intercrop',
            'groundnut + sunflower or + pigeonpea (intercrop, staggered planting)': 'groundnut_sunflower_intercrop',
            'groundnut + castor (intercrop)': 'groundnut_castor_intercrop',
            'groundnut - wheat / mustard sequence': 'groundnut',  # Sequence, treat as groundnut
        }
        
        normalized_crop = crop_normalizations.get(crop, crop)
        
        return normalized_crop not in SUPPORTED_CROPS
    
    def check_unsupported_weed_species(self, record: Dict) -> bool:
        """
        Check if weed species is in common list.
        
        Args:
            record: Single record dictionary
        
        Returns:
            True if weed species is not in common list
        """
        weed_species = record.get('weed_species', '').strip().lower()
        
        # If it's a general description, don't flag
        general_descriptions = [
            'general',
            'not specified',
            'effective weed control',
            'better weed management',
            'grassy weeds',
            'broad leaves',
            'broadleaf weeds',
            'sedges',
            'annual grasses',
            'perennial grasses',
            'annual and perennial grasses',
            'grasses & broad leaves',
            'grasses and broad-leaved weeds',
        ]
        
        for desc in general_descriptions:
            if desc in weed_species:
                return False  # It's a general description, not unsupported
        
        # Check if any known species is mentioned
        for known_species in COMMON_WEED_SPECIES:
            if known_species.lower() in weed_species:
                return False  # Contains known species
        
        # If it contains specific species names but none match known list
        # This is a soft check - we flag for review but don't reject
        if ',' in weed_species or '&' in weed_species:
            # It's a list of species, check if any are known
            return False  # Don't flag lists for now
        
        return True
    
    def validate_record(self, record: Dict) -> ValidationResult:
        """
        Validate a single record.
        
        Args:
            record: Single record dictionary
        
        Returns:
            ValidationResult object
        """
        record_id = record.get('record_id', 'unknown')
        
        # Run all checks
        has_missing, missing_fields = self.check_missing_values(record)
        has_conflicts, conflict_details = self.check_conflicts(record)
        has_invalid_refs, invalid_refs = self.check_invalid_references(record)
        has_unit_issues, unit_issues = self.check_inconsistent_units(record)
        unsupported_crop = self.check_unsupported_crop(record)
        unsupported_weed = self.check_unsupported_weed_species(record)
        
        # Determine overall validation status
        validation_passed = not (
            has_missing or
            has_conflicts or
            has_invalid_refs or
            unsupported_crop
        )
        
        # Collect warnings
        warnings = []
        if unsupported_weed:
            warnings.append("Weed species not in common list - review recommended")
        if has_unit_issues:
            warnings.extend([f"Unit issue: {issue}" for issue in unit_issues])
        
        return ValidationResult(
            record_id=record_id,
            has_missing_values=has_missing,
            missing_fields=missing_fields,
            has_conflicts=has_conflicts,
            conflict_details=conflict_details,
            has_invalid_references=has_invalid_refs,
            invalid_references=invalid_refs,
            has_inconsistent_units=has_unit_issues,
            unit_issues=unit_issues,
            unsupported_crop=unsupported_crop,
            unsupported_weed_species=unsupported_weed,
            validation_passed=validation_passed,
            warnings=warnings
        )
    
    def validate_all(self) -> List[ValidationResult]:
        """
        Validate all records in the dataset.
        
        Returns:
            List of ValidationResult objects
        """
        print("Validating all records...")
        
        # Check for duplicates across all records
        duplicates = self.check_duplicates()
        print(f"Found {len(duplicates)} duplicate groups")
        
        # Validate each record
        for record in self.records:
            result = self.validate_record(record)
            
            # Check if this record is part of a duplicate
            for dup_key, dup_ids in duplicates:
                if result.record_id in dup_ids:
                    result.is_duplicate = True
                    result.warnings.append(f"Duplicate record (key: {dup_key})")
            
            self.validation_results.append(result)
        
        return self.validation_results
    
    def generate_validation_report(self) -> Dict:
        """
        Generate summary validation report.
        
        Returns:
            Dictionary with validation summary
        """
        total_records = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.validation_passed)
        failed = total_records - passed
        
        missing_count = sum(1 for r in self.validation_results if r.has_missing_values)
        conflict_count = sum(1 for r in self.validation_results if r.has_conflicts)
        invalid_ref_count = sum(1 for r in self.validation_results if r.has_invalid_references)
        unit_issue_count = sum(1 for r in self.validation_results if r.has_inconsistent_units)
        unsupported_crop_count = sum(1 for r in self.validation_results if r.unsupported_crop)
        unsupported_weed_count = sum(1 for r in self.validation_results if r.unsupported_weed_species)
        duplicate_count = sum(1 for r in self.validation_results if r.is_duplicate)
        
        # Collect all missing fields
        all_missing_fields = {}
        for result in self.validation_results:
            for field in result.missing_fields:
                all_missing_fields[field] = all_missing_fields.get(field, 0) + 1
        
        # Collect all conflicts
        all_conflicts = {}
        for result in self.validation_results:
            for conflict in result.conflict_details:
                all_conflicts[conflict] = all_conflicts.get(conflict, 0) + 1
        
        return {
            'total_records': total_records,
            'validation_passed': passed,
            'validation_failed': failed,
            'success_rate': f"{(passed/total_records*100):.2f}%" if total_records > 0 else "N/A",
            'missing_value_count': missing_count,
            'conflict_count': conflict_count,
            'invalid_reference_count': invalid_ref_count,
            'unit_issue_count': unit_issue_count,
            'unsupported_crop_count': unsupported_crop_count,
            'unsupported_weed_species_count': unsupported_weed_count,
            'duplicate_count': duplicate_count,
            'missing_fields_breakdown': all_missing_fields,
            'conflicts_breakdown': all_conflicts,
        }
    
    def save_validation_report(self, output_path: Path) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            output_path: Path to save validation report
        """
        import json
        
        report = self.generate_validation_report()
        
        # Add detailed results
        report['detailed_results'] = [
            {
                'record_id': r.record_id,
                'validation_passed': r.validation_passed,
                'missing_fields': r.missing_fields,
                'is_duplicate': r.is_duplicate,
                'conflict_details': r.conflict_details,
                'invalid_references': r.invalid_references,
                'unit_issues': r.unit_issues,
                'unsupported_crop': r.unsupported_crop,
                'unsupported_weed_species': r.unsupported_weed_species,
                'warnings': r.warnings,
            }
            for r in self.validation_results
        ]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Validation report saved to {output_path}")


if __name__ == '__main__':
    # Run validation
    raw_path = Path('herbicide/raw/herbicide_dataset.csv')
    validator = HerbicideValidator(raw_path)
    
    # Validate all records
    validator.validate_all()
    
    # Generate and save report
    report = validator.generate_validation_report()
    
    print("\nValidation Summary:")
    print(f"Total records: {report['total_records']}")
    print(f"Passed: {report['validation_passed']}")
    print(f"Failed: {report['validation_failed']}")
    print(f"Success rate: {report['success_rate']}")
    print(f"\nIssue counts:")
    print(f"  Missing values: {report['missing_value_count']}")
    print(f"  Conflicts: {report['conflict_count']}")
    print(f"  Invalid references: {report['invalid_reference_count']}")
    print(f"  Unit issues: {report['unit_issue_count']}")
    print(f"  Unsupported crops: {report['unsupported_crop_count']}")
    print(f"  Unsupported weed species: {report['unsupported_weed_species_count']}")
    print(f"  Duplicates: {report['duplicate_count']}")
    
    # Save detailed report
    validator.save_validation_report(Path('herbicide/validation/validation_report.json'))
