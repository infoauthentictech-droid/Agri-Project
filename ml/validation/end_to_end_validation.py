"""
End-to-End Validation Script

Validates the entire ML pipeline:
1. Dataset integrity
2. Model availability
3. Inference functionality
4. XAI capability
5. Herbicide recommendation safety
6. API endpoints
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


class PipelineValidator:
    """Validates the complete ML pipeline."""

    def __init__(self):
        self.results = {
            "timestamp": str(time.time()),
            "validations": {}
        }

    def validate_dataset_integrity(self) -> Dict:
        """Validate dataset structure and manifest."""
        print("\n" + "="*60)
        print("VALIDATING DATASET INTEGRITY")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        # Check for dataset manifest
        manifest_path = PROJECT_ROOT / "datasets" / "dataset_manifest.json"
        if manifest_path.exists():
            validation["checks"].append("Dataset manifest exists")
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                validation["checks"].append(f"Manifest has {len(manifest.get('dataset_types', {}))} dataset types")
                validation["checks"].append(f"Manifest timestamp: {manifest.get('metadata', {}).get('timestamp', 'N/A')}")
            except Exception as e:
                validation["errors"].append(f"Failed to read manifest: {e}")
        else:
            validation["errors"].append("Dataset manifest not found")

        # Check for hybrid dataset
        hybrid_path = PROJECT_ROOT / "datasets" / "hybrid"
        if hybrid_path.exists():
            validation["checks"].append("Hybrid dataset directory exists")
            for task in ["classification", "detection", "segmentation"]:
                task_path = hybrid_path / task
                if task_path.exists():
                    validation["checks"].append(f"Hybrid {task} directory exists")
                else:
                    validation["errors"].append(f"Hybrid {task} directory missing")
        else:
            validation["errors"].append("Hybrid dataset directory not found")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def validate_model_checkpoints(self) -> Dict:
        """Validate model checkpoint availability."""
        print("\n" + "="*60)
        print("VALIDATING MODEL CHECKPOINTS")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        models_to_check = [
            "cnn", "vgg16", "mobilenet", "efficientnet", "resnet50",
            "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"
        ]

        classification_dir = PROJECT_ROOT / "experiments" / "classification"
        for model_name in models_to_check:
            model_dir = classification_dir / model_name
            if model_dir.exists():
                best_pth = model_dir / "best_model.pth"
                best_pt = model_dir / "best_model.pt"
                if best_pth.exists() or best_pt.exists():
                    validation["checks"].append(f"{model_name} checkpoint exists")
                else:
                    validation["errors"].append(f"{model_name} checkpoint missing")
            else:
                validation["errors"].append(f"{model_name} directory missing")

        # Check YOLO models
        yolo_dir = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n"
        if yolo_dir.exists():
            validation["checks"].append("YOLO directory exists")
            best_pt = yolo_dir / "weights" / "best.pt"
            if best_pt.exists():
                validation["checks"].append("YOLO checkpoint exists")
            else:
                validation["errors"].append("YOLO checkpoint missing")
        else:
            validation["errors"].append("YOLO directory missing")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def validate_inference_capability(self) -> Dict:
        """Validate model inference capability."""
        print("\n" + "="*60)
        print("VALIDATING INFERENCE CAPABILITY")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        try:
            from ml.models.registry import build_model

            # Test building a simple model
            config = {
                "model": {
                    "name": "cnn",
                    "num_classes": 2,
                },
                "dataset": {
                    "image_size": 128,
                },
            }

            model = build_model(config)
            validation["checks"].append("Model registry working")
            validation["checks"].append("CNN model built successfully")

        except Exception as e:
            validation["errors"].append(f"Model inference test failed: {e}")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def validate_xai_capability(self) -> Dict:
        """Validate XAI/Grad-CAM capability."""
        print("\n" + "="*60)
        print("VALIDATING XAI CAPABILITY")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        try:
            from ml.xai.gradcam import GradCAMExplainer
            validation["checks"].append("GradCAM module importable")
        except Exception as e:
            validation["errors"].append(f"GradCAM import failed: {e}")

        try:
            from ml.xai.api import XAIEngine
            validation["checks"].append("XAI API importable")
        except Exception as e:
            validation["errors"].append(f"XAI API import failed: {e}")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def validate_herbicide_safety(self) -> Dict:
        """Validate herbicide recommendation safety module."""
        print("\n" + "="*60)
        print("VALIDATING HERBICIDE SAFETY MODULE")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        try:
            from backend.app.utils.recommendation_safety_fixed import HerbicideSafetyChecker, SafetyStatus
            validation["checks"].append("Herbicide safety module importable")

            # Test safety checker
            checker = HerbicideSafetyChecker()
            validation["checks"].append("Safety checker instantiated")

            # Test safety check
            test_input = {
                "crop": "groundnut",
                "weed_species": "unknown",
                "crop_growth_stage": "vegetative",
                "region": "asia"
            }
            test_recommendation = {
                "success": True,
                "active_ingredient": "glyphosate"
            }

            result = checker.perform_safety_checks(test_input, test_recommendation, 0.8)
            validation["checks"].append(f"Safety check executed: {result.status.value}")

        except Exception as e:
            validation["errors"].append(f"Herbicide safety validation failed: {e}")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def validate_api_endpoints(self) -> Dict:
        """Validate API endpoint structure."""
        print("\n" + "="*60)
        print("VALIDATING API ENDPOINTS")
        print("="*60)

        validation = {
            "status": "unknown",
            "checks": [],
            "errors": []
        }

        # Check for API files
        api_file = PROJECT_ROOT / "backend" / "app" / "api" / "endpoints.py"
        if api_file.exists():
            validation["checks"].append("API endpoints file exists")
        else:
            validation["errors"].append("API endpoints file missing")

        main_file = PROJECT_ROOT / "backend" / "app" / "main.py"
        if main_file.exists():
            validation["checks"].append("API main file exists")
        else:
            validation["errors"].append("API main file missing")

        validation["status"] = "passed" if not validation["errors"] else "failed"
        print(f"Status: {validation['status']}")
        print(f"Checks passed: {len(validation['checks'])}")
        print(f"Errors: {len(validation['errors'])}")

        return validation

    def run_all_validations(self) -> Dict:
        """Run all validations and return summary."""
        print("\n" + "="*60)
        print("RUNNING END-TO-END VALIDATION")
        print("="*60)

        self.results["validations"]["dataset_integrity"] = self.validate_dataset_integrity()
        self.results["validations"]["model_checkpoints"] = self.validate_model_checkpoints()
        self.results["validations"]["inference_capability"] = self.validate_inference_capability()
        self.results["validations"]["xai_capability"] = self.validate_xai_capability()
        self.results["validations"]["herbicide_safety"] = self.validate_herbicide_safety()
        self.results["validations"]["api_endpoints"] = self.validate_api_endpoints()

        # Calculate overall status
        total_validations = len(self.results["validations"])
        passed_validations = sum(
            1 for v in self.results["validations"].values()
            if v["status"] == "passed"
        )

        self.results["summary"] = {
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "overall_status": "passed" if passed_validations == total_validations else "failed"
        }

        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total validations: {total_validations}")
        print(f"Passed: {passed_validations}")
        print(f"Failed: {total_validations - passed_validations}")
        print(f"Overall status: {self.results['summary']['overall_status'].upper()}")
        print("="*60)

        return self.results

    def save_report(self, output_path: Optional[Path] = None) -> None:
        """Save validation report to file."""
        if output_path is None:
            output_path = PROJECT_ROOT / "experiments" / "validation" / "pipeline_validation_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\nValidation report saved to: {output_path}")


def main():
    """Main function to run end-to-end validation."""
    validator = PipelineValidator()
    results = validator.run_all_validations()
    validator.save_report()

    return 0 if results["summary"]["overall_status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
