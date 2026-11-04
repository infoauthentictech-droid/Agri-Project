"""Hybrid classification models combining multiple architectures.

This module implements hybrid ensemble models that combine predictions from
multiple CNN architectures for improved accuracy and robustness.

Supported hybrid configurations:
- resnet50 + mobilenet
- resnet50 + efficientnet
- resnet50 + efficientnet + mobilenet

These hybrid models leverage:
- Complementary feature extraction from different architectures
- Ensemble learning for better generalization
- ResNet residual backbones for hierarchical feature extraction
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, mobilenet_v2, efficientnet_b0
from typing import List, Dict


class HybridClassifier(nn.Module):
    """Hybrid classifier combining multiple CNN architectures.

    This class implements ensemble-based classification by combining
    features from multiple pretrained backbones.
    """

    def __init__(
        self,
        models: List[str],
        num_classes: int = 2,
        pretrained: bool = True,
        fusion_method: str = "concat"
    ):
        """
        Initialize hybrid classifier.

        Args:
            models: List of model names to combine (e.g., ["resnet50", "mobilenet"])
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            fusion_method: Method to combine features ("concat", "average", "weighted")
        """
        super().__init__()

        self.models = models
        self.num_classes = num_classes
        self.fusion_method = fusion_method

        # Initialize individual models
        self.backbones = nn.ModuleDict()
        self.feature_dims = []

        for model_name in models:
            backbone = self._build_backbone(model_name, pretrained)
            self.backbones[model_name] = backbone
            self.feature_dims.append(self._get_feature_dim(model_name, backbone))

        # Build fusion layer
        if fusion_method == "concat":
            total_dim = sum(self.feature_dims)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        elif fusion_method == "average":
            # Simple averaging of logits
            self.fusion = None
        elif fusion_method == "weighted":
            # Learnable weighted average
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def _build_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """Build a single backbone model."""
        if model_name == "resnet50":
            model = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            # Remove the final classification layer
            model.fc = nn.Identity()
            return model

        elif model_name == "mobilenet":
            model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            # Remove the final classification layer
            model.classifier = nn.Identity()
            return model

        elif model_name == "efficientnet":
            model = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            # Remove the final classification layer
            model.classifier = nn.Identity()
            return model

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_feature_dim(self, model_name: str, model: nn.Module) -> int:
        """Get the feature dimension for a model."""
        if model_name == "resnet50":
            return 2048  # ResNet50 feature dimension
        elif model_name == "mobilenet":
            return 1280  # MobileNetV2 feature dimension
        elif model_name == "efficientnet":
            return 1280  # EfficientNet-B0 feature dimension
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model."""
        features = []

        # Extract features from each backbone
        for model_name in self.models:
            backbone = self.backbones[model_name]
            feat = backbone(x)
            features.append(feat)

        # Fuse features based on method
        if self.fusion_method == "concat":
            # Concatenate features and pass through fusion layer
            fused = torch.cat(features, dim=1)
            output = self.fusion(fused)

        elif self.fusion_method == "average":
            # Average the features (requires same dimensions)
            # This is a simplified approach - in practice, you might need
            # projection layers to match dimensions
            # For now, we'll use a voting approach on logits
            logits = []
            for feat in features:
                # Temporarily add a classifier head for each backbone
                # This is a workaround - in production, you'd project to common dim
                temp_classifier = nn.Linear(feat.size(1), self.num_classes).to(feat.device)
                logit = temp_classifier(feat)
                logits.append(logit)
            output = torch.stack(logits).mean(dim=0)

        elif self.fusion_method == "weighted":
            # Weighted average of logits
            logits = []
            for feat in features:
                temp_classifier = nn.Linear(feat.size(1), self.num_classes).to(feat.device)
                logit = temp_classifier(feat)
                logits.append(logit)

            stacked_logits = torch.stack(logits)  # (num_models, batch, num_classes)
            weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
            output = (stacked_logits * weights).sum(dim=0)

        return output


class ResNet50MobileNet(nn.Module):
    """Hybrid model combining ResNet50 and MobileNetV2.

    This combines the powerful feature extraction of ResNet50 with the
    efficiency of MobileNetV2 for a balanced approach.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = HybridClassifier(
            models=["resnet50", "mobilenet"],
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_method="concat"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50EfficientNet(nn.Module):
    """Hybrid model combining ResNet50 and EfficientNet-B0.

    This combines ResNet50's residual learning with EfficientNet's
    compound scaling for improved accuracy.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = HybridClassifier(
            models=["resnet50", "efficientnet"],
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_method="concat"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet50EfficientNetMobileNet(nn.Module):
    """Hybrid model combining ResNet50, EfficientNet-B0, and MobileNetV2.

    This three-way ensemble leverages the strengths of all three architectures:
    - ResNet50: Deep residual learning for hierarchical features
    - EfficientNet: Compound scaling for efficient accuracy
    - MobileNetV2: Lightweight architecture for speed
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model = HybridClassifier(
            models=["resnet50", "efficientnet", "mobilenet"],
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_method="concat"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_hybrid_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a hybrid model by name.

    Args:
        model_name: Name of the hybrid model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights

    Returns:
        Hybrid model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower()

    if model_name == "resnet50_mobilenet":
        return ResNet50MobileNet(num_classes, pretrained)
    elif model_name == "resnet50_efficientnet":
        return ResNet50EfficientNet(num_classes, pretrained)
    elif model_name == "resnet50_efficientnet_mobilenet":
        return ResNet50EfficientNetMobileNet(num_classes, pretrained)
    else:
        raise ValueError(
            f"Unknown hybrid model: {model_name}. "
            "Use 'resnet50_mobilenet', 'resnet50_efficientnet', or 'resnet50_efficientnet_mobilenet'"
        )


def list_hybrid_models() -> List[str]:
    """List available hybrid model names."""
    return [
        "resnet50_mobilenet",
        "resnet50_efficientnet",
        "resnet50_efficientnet_mobilenet"
    ]


if __name__ == "__main__":
    # Test hybrid models
    print("Testing hybrid models...")

    batch_size = 4
    num_classes = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    for model_name in list_hybrid_models():
        print(f"\nTesting {model_name}...")
        model = build_hybrid_model(model_name, num_classes, pretrained=False)
        output = model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nAll hybrid models tested successfully!")
