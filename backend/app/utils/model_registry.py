"""Central Model Registry

This registry explicitly records each model's task, checkpoint path, architecture,
preprocessing, class mapping, version, and evaluation status.

This ensures consistent model selection and prevents model-task confusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class ModelTask(str, Enum):
    """Model task types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    PLANT_GATE = "plant_gate"


class ModelStatus(str, Enum):
    """Model evaluation and deployment status."""
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    # Identification
    name: str
    task: ModelTask
    architecture: str
    version: str

    # Paths
    checkpoint_path: str
    config_path: Optional[str] = None
    normalization_path: Optional[str] = None

    # Preprocessing
    image_size: int = 224
    normalize: bool = True
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    # Class mapping
    class_names: Optional[List[str]] = None
    class_mapping: Optional[Dict[int, str]] = None

    # Evaluation status
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    test_accuracy: Optional[float] = None
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    test_date: Optional[str] = None
    test_dataset_version: Optional[str] = None

    # Dataset information
    dataset_type: Optional[str] = None  # primary, primary_augmented, secondary, hybrid
    dataset_path: Optional[str] = None

    # Deployment
    is_default: bool = False
    deployment_date: Optional[str] = None

    # Metadata
    description: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Set default timestamps."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()

        # Set default normalization if not provided
        if self.normalize and self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.normalize and self.std is None:
            self.std = [0.229, 0.224, 0.225]


class ModelRegistry:
    """Central registry for all models in the project."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize the model registry.

        Args:
            registry_path: Path to registry JSON file. If None, uses default.
        """
        if registry_path is None:
            registry_path = PROJECT_ROOT / "backend" / "app" / "utils" / "model_registry.json"

        self.registry_path = registry_path
        self.models: Dict[str, ModelConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from JSON file."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                for name, model_data in data.items():
                    # Convert enums
                    model_data["task"] = ModelTask(model_data["task"])
                    model_data["status"] = ModelStatus(model_data["status"])
                    self.models[name] = ModelConfig(**model_data)
        else:
            # Initialize with default models
            self._initialize_default_models()

    def _save_registry(self) -> None:
        """Save model registry to JSON file."""
        data = {}
        for name, config in self.models.items():
            data[name] = asdict(config)
            # Convert enums to strings
            data[name]["task"] = config.task.value
            data[name]["status"] = config.status.value

        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _initialize_default_models(self) -> None:
        """Initialize registry with known models from the project."""
        # Classification models
        self.register_model(
            ModelConfig(
                name="mobilenet",
                task=ModelTask.CLASSIFICATION,
                architecture="MobileNetV2",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "classification" / "mobilenet" / "best_model.pth"),
                config_path=str(PROJECT_ROOT / "experiments" / "classification" / "mobilenet" / "results.json"),
                image_size=128,
                class_names=["groundnut", "weed"],
                status=ModelStatus.VALIDATED,
                test_accuracy=0.9659,
                test_precision=0.9659,
                test_recall=0.9659,
                test_f1=0.9659,
                test_date="2024-10-24",
                test_dataset_version="v1.0",
                is_default=True,
                description="MobileNetV2 classifier for groundnut vs weed",
            )
        )

        self.register_model(
            ModelConfig(
                name="efficientnet",
                task=ModelTask.CLASSIFICATION,
                architecture="EfficientNet-B0",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "classification" / "efficientnet" / "best_model.pth"),
                config_path=str(PROJECT_ROOT / "experiments" / "classification" / "efficientnet" / "results.json"),
                image_size=128,
                class_names=["groundnut", "weed"],
                status=ModelStatus.VALIDATED,
                test_accuracy=0.9801,
                test_precision=0.9659,
                test_recall=0.9659,
                test_f1=0.9659,
                test_date="2024-10-24",
                test_dataset_version="v1.0",
                description="EfficientNet-B0 classifier for groundnut vs weed",
            )
        )

        # Detection model
        self.register_model(
            ModelConfig(
                name="yolov8n",
                task=ModelTask.DETECTION,
                architecture="YOLOv8n",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"),
                config_path=str(PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"),
                image_size=640,
                class_names=["crop", "weed"],
                class_mapping={0: "crop", 1: "weed"},
                status=ModelStatus.VALIDATED,
                test_accuracy=None,  # Detection uses mAP
                test_precision=0.85,  # Placeholder - update with actual metrics
                test_recall=0.82,
                test_f1=0.83,
                test_date="2024-10-24",
                test_dataset_version="v1.0",
                is_default=True,
                description="YOLOv8n detection model for crop and weed",
            )
        )

        # Segmentation model (commented out until checkpoint is available)
        # self.register_model(
        #     ModelConfig(
        #         name="yolov8n-seg",
        #         task=ModelTask.SEGMENTATION,
        #         architecture="YOLOv8n-Seg",
        #         version="1.0",
        #         checkpoint_path=str(PROJECT_ROOT / "experiments" / "yoloseg" / "agri_yolov8n_seg" / "weights" / "best.pt"),
        #         config_path=str(PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_segmentation" / "data.yaml"),
        #         image_size=640,
        #         class_names=["background", "crop", "weed"],
        #         class_mapping={0: "background", 1: "crop", 2: "weed"},
        #         status=ModelStatus.VALIDATED,
        #         test_accuracy=None,
        #         test_precision=0.78,
        #         test_recall=0.67,
        #         test_f1=0.72,
        #         test_date="2024-10-24",
        #         test_dataset_version="v1.0",
        #         is_default=True,
        #         description="YOLOv8n-Seg segmentation model for pixel-level plant masking",
        #     )
        # )

        # Plant gate model
        self.register_model(
            ModelConfig(
                name="plant_gate",
                task=ModelTask.PLANT_GATE,
                architecture="RandomForest",
                version="2.0",  # Updated after fix
                checkpoint_path=str(PROJECT_ROOT / "backend" / "app" / "utils" / "plant_gate_model.joblib"),
                config_path=str(PROJECT_ROOT / "backend" / "app" / "utils" / "plant_gate_model.json"),
                image_size=256,
                class_names=["non_plant", "plant"],
                status=ModelStatus.VALIDATED,
                test_accuracy=0.95,
                test_precision=0.94,
                test_recall=0.96,
                test_f1=0.95,
                test_date="2024-10-26",  # Updated after retrain
                test_dataset_version="v1.0",
                is_default=True,
                description="Plant vs non-plant gate classifier (updated with proper train/val/test split)",
                notes="Retrained with proper train/val/test split to prevent test contamination",
            )
        )

        # Hybrid models (for improved accuracy over single modules)
        self.register_model(
            ModelConfig(
                name="resnet50_mobilenet",
                task=ModelTask.CLASSIFICATION,
                architecture="ResNet50+MobileNetV2 (Hybrid)",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_mobilenet" / "best_model.pth"),
                config_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_mobilenet" / "results.json"),
                image_size=224,
                class_names=["groundnut", "weed"],
                status=ModelStatus.EXPERIMENTAL,  # Needs training
                test_accuracy=None,
                test_precision=None,
                test_recall=None,
                test_f1=None,
                test_date=None,
                test_dataset_version="v1.0",
                is_default=False,
                description="Hybrid model combining ResNet50 and MobileNetV2 for improved accuracy",
                notes="Expected to provide higher accuracy than single modules by combining complementary feature extraction",
            )
        )

        self.register_model(
            ModelConfig(
                name="resnet50_efficientnet",
                task=ModelTask.CLASSIFICATION,
                architecture="ResNet50+EfficientNet-B0 (Hybrid)",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_efficientnet" / "best_model.pth"),
                config_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_efficientnet" / "results.json"),
                image_size=224,
                class_names=["groundnut", "weed"],
                status=ModelStatus.EXPERIMENTAL,  # Needs training
                test_accuracy=None,
                test_precision=None,
                test_recall=None,
                test_f1=None,
                test_date=None,
                test_dataset_version="v1.0",
                is_default=False,
                description="Hybrid model combining ResNet50 and EfficientNet-B0 for improved accuracy",
                notes="Expected to provide higher accuracy than single modules by combining residual learning with compound scaling",
            )
        )

        self.register_model(
            ModelConfig(
                name="resnet50_efficientnet_mobilenet",
                task=ModelTask.CLASSIFICATION,
                architecture="ResNet50+EfficientNet+MobileNet (Hybrid)",
                version="1.0",
                checkpoint_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_efficientnet_mobilenet" / "best_model.pth"),
                config_path=str(PROJECT_ROOT / "experiments" / "classification" / "resnet50_efficientnet_mobilenet" / "results.json"),
                image_size=224,
                class_names=["groundnut", "weed"],
                status=ModelStatus.EXPERIMENTAL,  # Needs training
                test_accuracy=None,
                test_precision=None,
                test_recall=None,
                test_f1=None,
                test_date=None,
                test_dataset_version="v1.0",
                is_default=False,
                description="Hybrid model combining ResNet50, EfficientNet-B0, and MobileNetV2 for maximum accuracy",
                notes="Three-way ensemble leveraging strengths of all architectures: ResNet50 (deep residual), EfficientNet (compound scaling), MobileNetV2 (lightweight)",
            )
        )

    def register_model(self, config: ModelConfig) -> None:
        """Register or update a model in the registry.

        Args:
            config: Model configuration
        """
        config.updated_at = datetime.utcnow().isoformat()
        self.models[config.name] = config
        self._save_registry()

    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get a model configuration by name.

        Args:
            name: Model name

        Returns:
            Model configuration or None if not found
        """
        return self.models.get(name)

    def get_models_by_task(self, task: ModelTask) -> List[ModelConfig]:
        """Get all models for a specific task.

        Args:
            task: Model task

        Returns:
            List of model configurations
        """
        return [m for m in self.models.values() if m.task == task]

    def get_default_model(self, task: ModelTask) -> Optional[ModelConfig]:
        """Get the default model for a task.

        Args:
            task: Model task

        Returns:
            Default model configuration or None
        """
        for model in self.get_models_by_task(task):
            if model.is_default:
                return model
        return None

    def list_models(self, task: Optional[ModelTask] = None) -> Dict[str, Dict[str, Any]]:
        """List all models in the registry.

        Args:
            task: Optional task filter

        Returns:
            Dictionary of model summaries
        """
        models = self.get_models_by_task(task) if task else self.models.values()
        return {
            name: {
                "task": config.task.value,
                "architecture": config.architecture,
                "version": config.version,
                "status": config.status.value,
                "is_default": config.is_default,
                "checkpoint_exists": Path(config.checkpoint_path).exists(),
            }
            for name, config in self.models.items()
            if task is None or config.task == task
        }

    def update_model_status(
        self,
        name: str,
        status: ModelStatus,
        test_accuracy: Optional[float] = None,
        test_precision: Optional[float] = None,
        test_recall: Optional[float] = None,
        test_f1: Optional[float] = None,
        test_dataset_version: Optional[str] = None,
    ) -> bool:
        """Update model evaluation status.

        Args:
            name: Model name
            status: New status
            test_accuracy: Test accuracy
            test_precision: Test precision
            test_recall: Test recall
            test_f1: Test F1
            test_dataset_version: Dataset version used for testing

        Returns:
            True if updated, False if model not found
        """
        if name not in self.models:
            return False

        config = self.models[name]
        config.status = status
        config.test_date = datetime.utcnow().isoformat()

        if test_accuracy is not None:
            config.test_accuracy = test_accuracy
        if test_precision is not None:
            config.test_precision = test_precision
        if test_recall is not None:
            config.test_recall = test_recall
        if test_f1 is not None:
            config.test_f1 = test_f1
        if test_dataset_version is not None:
            config.test_dataset_version = test_dataset_version

        config.updated_at = datetime.utcnow().isoformat()
        self._save_registry()
        return True

    def set_default_model(self, task: ModelTask, name: str) -> bool:
        """Set the default model for a task.

        Args:
            task: Model task
            name: Model name

        Returns:
            True if set, False if model not found
        """
        if name not in self.models or self.models[name].task != task:
            return False

        # Remove default from other models of same task
        for model in self.get_models_by_task(task):
            model.is_default = False

        self.models[name].is_default = True
        self._save_registry()
        return True

    def validate_model_paths(self) -> Dict[str, bool]:
        """Validate that all registered model paths exist.

        Returns:
            Dictionary mapping model names to path validity
        """
        results = {}
        for name, config in self.models.items():
            checkpoint_exists = Path(config.checkpoint_path).exists()
            config_exists = config.config_path is None or Path(config.config_path).exists()
            norm_exists = config.normalization_path is None or Path(config.normalization_path).exists()
            results[name] = checkpoint_exists and config_exists and norm_exists
        return results


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
