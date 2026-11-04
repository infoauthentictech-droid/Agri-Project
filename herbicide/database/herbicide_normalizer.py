"""
Herbicide dataset normalization module.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from herbicide.schemas.herbicide_schema import (
    HerbicideRecord,
    CropType,
    CroppingSystem,
    WeedCategory,
    ApplicationStage,
    ConfidenceLevel,
)


class HerbicideNormalizer:
    """
    Normalize herbicide dataset records to standard schema.
    
    This module:
    - Normalizes crop names and cropping systems
    - Normalizes application stages
    - Normalizes confidence levels
    - Extracts structured information from unstructured fields
    - Preserves original data and sources
    """
    
    def __init__(self, raw_data_path: Path):
        """
        Initialize normalizer with raw dataset.
        
        Args:
            raw_data_path: Path to raw herbicide CSV file
        """
        self.raw_data_path = raw_data_path
        self.records: List[Dict] = []
        self.normalized_records: List[HerbicideRecord] = []
        self.load_data()
    
    def load_data(self) -> None:
        """Load raw CSV data."""
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.records = list(reader)
        
        print(f"Loaded {len(self.records)} records from {self.raw_data_path}")
    
    def normalize_crop(self, crop: str) -> str:
        """
        Normalize crop name to standard format.
        
        Args:
            crop: Original crop name
        
        Returns:
            Normalized crop name
        """
        crop = crop.strip().lower()
        
        crop_mapping = {
            'groundnut': CropType.GROUNDNUT.value,
            'groundnut (sole crop)': CropType.GROUNDNUT.value,
            'groundnut + pigeonpea (intercrop)': CropType.GROUNDNUT_PIGEONPEA_INTERCROP.value,
            'groundnut + sunflower or + pigeonpea (intercrop, staggered planting)': CropType.GROUNDNUT_SUNFLOWER_INTERCROP.value,
            'groundnut + castor (intercrop)': CropType.GROUNDNUT_CASTOR_INTERCROP.value,
            'groundnut - wheat / mustard sequence': CropType.GROUNDNUT.value,
        }
        
        return crop_mapping.get(crop, CropType.UNKNOWN.value)
    
    def normalize_cropping_system(self, cropping_system: str) -> str:
        """
        Normalize cropping system to standard format.
        
        Args:
            cropping_system: Original cropping system
        
        Returns:
            Normalized cropping system
        """
        system = cropping_system.strip().lower()
        
        system_mapping = {
            'sole crop': CroppingSystem.SOLE_CROP.value,
            'intercrop': CroppingSystem.INTERCROP.value,
            'sequence/rotation': CroppingSystem.SEQUENCE_ROTATION.value,
        }
        
        return system_mapping.get(system, CroppingSystem.UNKNOWN.value)
    
    def normalize_weed_category(self, weed_species: str) -> str:
        """
        Derive weed category from species description.
        
        Args:
            weed_species: Original weed species description
        
        Returns:
            Normalized weed category
        """
        species = weed_species.strip().lower()
        
        # Check for specific patterns
        if 'grass' in species and 'broad' in species and 'sedge' in species:
            return WeedCategory.GRASSES_BROADLEAVES_SEDGES.value
        elif 'grass' in species and 'broad' in species:
            return WeedCategory.GRASSES_BROADLEAVES.value
        elif 'sedge' in species and 'broad' in species:
            return WeedCategory.BROADLEAF_WEEDS.value  # Primary category
        elif 'grass' in species:
            return WeedCategory.GRASSY_WEEDS.value
        elif 'broad' in species or 'leaf' in species:
            return WeedCategory.BROADLEAF_WEEDS.value
        elif 'sedge' in species:
            return WeedCategory.SEDGES.value
        elif 'monocot' in species:
            return WeedCategory.MONOCOT_WEEDS.value
        elif 'general' in species or 'not specified' in species:
            return WeedCategory.NOT_SPECIFIED.value
        else:
            return WeedCategory.UNKNOWN.value
    
    def normalize_application_stage(self, stage: str) -> str:
        """
        Normalize application stage to standard format.
        
        Args:
            stage: Original application stage description
        
        Returns:
            Normalized application stage
        """
        stage = stage.strip().lower()
        
        stage_mapping = {
            'pre-sowing soil incorporation': ApplicationStage.PRE_SOWING_SOIL_INCORPORATION.value,
            'pre-emergence': ApplicationStage.PRE_EMERGENCE.value,
            'post-emergence': ApplicationStage.POST_EMERGENCE.value,
            'late post-emergence': ApplicationStage.LATE_POST_EMERGENCE.value,
            'pre/early post-emergence': ApplicationStage.PRE_EARLY_POST_EMERGENCE.value,
            'manual control': ApplicationStage.MANUAL_CONTROL.value,
            'n/a': ApplicationStage.NOT_APPLICABLE.value,
        }
        
        # Check for partial matches
        for key, value in stage_mapping.items():
            if key in stage:
                return value
        
        return ApplicationStage.UNKNOWN.value
    
    def normalize_confidence(self, confidence: str) -> str:
        """
        Normalize confidence level to standard format.
        
        Args:
            confidence: Original confidence description
        
        Returns:
            Normalized confidence level
        """
        conf = confidence.strip().lower()
        
        conf_mapping = {
            'official/regulatory': ConfidenceLevel.OFFICIAL_REGULATORY.value,
            'official/regulatory ': ConfidenceLevel.OFFICIAL_REGULATORY.value,
            'research bulletin': ConfidenceLevel.RESEARCH_BULLETIN.value,
            'state/university extension': ConfidenceLevel.STATE_UNIVERSITY_EXTENSION.value,
        }
        
        return conf_mapping.get(conf, ConfidenceLevel.UNKNOWN.value)
    
    def extract_mode_of_action(self, notes: Optional[str]) -> Optional[str]:
        """
        Extract mode of action from notes field.
        
        Args:
            notes: Original notes field
        
        Returns:
            Extracted mode of action or None
        """
        if not notes:
            return None
        
        notes_lower = notes.lower()
        
        # Look for mode of action patterns
        if 'mode of action:' in notes_lower:
            # Extract after "mode of action:"
            match = re.search(r'mode of action:\s*(.+?)(?:;|$)', notes_lower)
            if match:
                return match.group(1).strip()
        
        # Look for common MOA keywords
        moa_keywords = [
            'als inhibitor',
            'acetolactate synthase',
            'ppo inhibitor',
            'microtubule',
            'cell division',
            'acetyl coa',
            'accase',
            'amino acid',
            'photosynthesis',
            'contact desiccant',
        ]
        
        for keyword in moa_keywords:
            if keyword in notes_lower:
                return keyword
        
        return None
    
    def extract_unit(self, dose_str: Optional[str]) -> Optional[str]:
        """
        Extract unit from dose string.
        
        Args:
            dose_str: Dose string with unit
        
        Returns:
            Extracted unit or None
        """
        if not dose_str:
            return None
        
        unit_patterns = [
            r'kg/ha',
            r'g/ha',
            r'l/ha',
            r'ml/ha',
            r'lit/ha',
            r'ltr/ha',
        ]
        
        for pattern in unit_patterns:
            if re.search(pattern, dose_str, re.IGNORECASE):
                return pattern
        
        return None
    
    def parse_numeric_rate(self, dose_str: Optional[str]) -> Optional[float]:
        """
        Parse numeric rate from dose string.
        
        Args:
            dose_str: Dose string with numeric value
        
        Returns:
            Numeric rate or None
        """
        if not dose_str:
            return None
        
        # Extract first numeric value
        match = re.search(r'(\d+\.?\d*)', dose_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def normalize_record(self, record: Dict) -> HerbicideRecord:
        """
        Normalize a single record to HerbicideRecord schema.
        
        Args:
            record: Original record dictionary
        
        Returns:
            Normalized HerbicideRecord
        """
        notes = record.get('notes', '')
        
        return HerbicideRecord(
            record_id=record.get('record_id', ''),
            crop=self.normalize_crop(record.get('crop', '')),
            cropping_system=self.normalize_cropping_system(record.get('cropping_system', '')),
            weed_species=record.get('weed_species', ''),
            weed_category=self.normalize_weed_category(record.get('weed_species', '')),
            herbicide_common_name=record.get('herbicide_common_name', '').strip() or None,
            trade_names=record.get('trade_names', '').strip() or None,
            active_ingredient=None,  # Would need extraction from formulation field
            formulation=None,  # Would need extraction from formulation field
            application_stage=self.normalize_application_stage(record.get('application_stage', '')),
            dose_ai_per_ha=record.get('dose_ai_per_ha', '').strip() or None,
            dose_formulation_per_ha=record.get('dose_formulation_per_ha', '').strip() or None,
            application_rate_normalized=self.parse_numeric_rate(record.get('dose_ai_per_ha', '')),
            unit=self.extract_unit(record.get('dose_formulation_per_ha', '')),
            phi_waiting_period_days=self._parse_int(record.get('phi_waiting_period_days', '')),
            mode_of_action=self.extract_mode_of_action(notes),
            limitations=None,  # Would need more sophisticated extraction
            crop_safety_information=None,  # Would need more sophisticated extraction
            region=None,  # Would need extraction from source
            season=None,  # Would need extraction from notes
            source=record.get('source', ''),
            page=record.get('page', '').strip() or None,
            confidence=self.normalize_confidence(record.get('confidence', '')),
            conflict_flag=record.get('conflict_flag', '').strip().lower() in ['yes', 'true', '1'],
            notes=notes.strip() or None,
            normalized_at=datetime.now().isoformat(),
            validation_status='pending'
        )
    
    def _parse_int(self, value: str) -> Optional[int]:
        """Parse integer from string, return None if invalid."""
        try:
            return int(value.strip()) if value.strip() else None
        except (ValueError, AttributeError):
            return None
    
    def normalize_all(self) -> List[HerbicideRecord]:
        """
        Normalize all records in the dataset.
        
        Returns:
            List of normalized HerbicideRecord objects
        """
        print("Normalizing all records...")
        
        for record in self.records:
            normalized = self.normalize_record(record)
            self.normalized_records.append(normalized)
        
        print(f"Normalized {len(self.normalized_records)} records")
        return self.normalized_records
    
    def save_normalized_csv(self, output_path: Path) -> None:
        """
        Save normalized records to CSV file.
        
        Args:
            output_path: Path to save normalized CSV
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.normalized_records:
            print("No normalized records to save")
            return
        
        # Get field names from HerbicideRecord
        field_names = [
            'record_id', 'crop', 'cropping_system', 'weed_species', 'weed_category',
            'herbicide_common_name', 'trade_names', 'active_ingredient', 'formulation',
            'application_stage', 'dose_ai_per_ha', 'dose_formulation_per_ha',
            'application_rate_normalized', 'unit', 'phi_waiting_period_days',
            'mode_of_action', 'limitations', 'crop_safety_information',
            'region', 'season', 'source', 'page', 'confidence',
            'conflict_flag', 'notes', 'normalized_at', 'validation_status'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            
            for record in self.normalized_records:
                writer.writerow(record.__dict__)
        
        print(f"Normalized dataset saved to {output_path}")
    
    def save_normalized_json(self, output_path: Path) -> None:
        """
        Save normalized records to JSON file.
        
        Args:
            output_path: Path to save normalized JSON
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.normalized_records:
            print("No normalized records to save")
            return
        
        data = [record.__dict__ for record in self.normalized_records]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Normalized dataset saved to {output_path}")


if __name__ == '__main__':
    # Run normalization
    raw_path = Path('herbicide/raw/herbicide_dataset.csv')
    normalizer = HerbicideNormalizer(raw_path)
    
    # Normalize all records
    normalizer.normalize_all()
    
    # Save normalized datasets
    normalizer.save_normalized_csv(Path('herbicide/processed/herbicide_dataset_normalized.csv'))
    normalizer.save_normalized_json(Path('herbicide/processed/herbicide_dataset_normalized.json'))
    
    print("\nNormalization complete!")
