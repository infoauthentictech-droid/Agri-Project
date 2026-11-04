"""Model registry — uniform interface for the classifier benchmark.

Classification backbones (all ImageNet pretrained by default):
- ``cnn`` : original architecture (ml/models/cnn.py) - the single CNN model used
- ``vgg16``, ``resnet50``
- ``mobilenet`` (MobileNetV2), ``efficientnet`` (EfficientNet-B0)
- Hybrid models: ``resnet50_mobilenet``, ``resnet50_efficientnet``, ``resnet50_efficientnet_mobilenet``

Note: ``custom_cnn`` is deprecated and hidden from the API. Use ``cnn`` instead.

Every transfer model gets an identical, newly-initialized classification head
(``Dropout(0.5) -> Linear(num_classes)``) so head differences never explain
benchmark differences. Optional backbone freezing is config-driven.
"""
from __future__ import annotations

import torch
from torch import nn

from .cnn import CNN as _CNNTorch  # noqa: F401  (kept for API compat)

_WEIGHT_ENUMS = {
    "vgg16": "VGG16_Weights",
    "resnet50": "ResNet50_Weights",
    "mobilenet": "MobileNet_V2_Weights",
    "efficientnet": "EfficientNet_B0_Weights",
}

_WEIGHT_VARIANTS = {
    "vgg16": "IMAGENET1K_V1",
    "resnet50": "IMAGENET1K_V2",
    "mobilenet": "IMAGENET1K_V1",
    "efficientnet": "IMAGENET1K_V1",
}

_HEAD_KEY = {
    "vgg16": "classifier",
    "resnet50": "fc",
    "mobilenet": "classifier",
    "efficientnet": "classifier",
}


def _pretrained_weights(name: str):
    import torchvision.models as tv

    enum = getattr(tv, _WEIGHT_ENUMS[name])
    return getattr(enum, _WEIGHT_VARIANTS[name])


def _classification_head(in_features: int, num_classes: int) -> nn.Module:
    return nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))


def _freeze_backbone(model: nn.Module, head_modules: tuple) -> None:
    for name, p in model.named_parameters():
        if not name.startswith(head_modules):
            p.requires_grad = False
def _build_transfer_model(mcfg: dict, name: str, num_classes: int) -> nn.Module:
    use_pretrained = bool(mcfg.get("weights", "imagenet") != "none")
    weights = _pretrained_weights(name) if use_pretrained else None
    head_modules = ()
    import torchvision.models as tv

    if name == "vgg16":
        base = tv.vgg16(weights=weights)
        in_f = base.classifier[0].in_features
        base.classifier = _classification_head(in_f, num_classes)
        head_modules = ("classifier",)
    elif name == "resnet50":
        base = tv.resnet50(weights=weights)
        in_f = base.fc.in_features
        base.fc = _classification_head(in_f, num_classes)
        head_modules = ("fc",)
    elif name == "mobilenet":
        base = tv.mobilenet_v2(weights=weights)
        in_f = base.classifier[1].in_features
        base.classifier = _classification_head(in_f, num_classes)
        head_modules = ("classifier",)
    elif name == "efficientnet":
        base = tv.efficientnet_b0(weights=weights)
        in_f = base.classifier[1].in_features
        base.classifier = _classification_head(in_f, num_classes)
        head_modules = ("classifier",)
    else:  # pragma: no cover - guarded by the registry table
        raise ValueError(f"unknown transfer model '{name}'")

    if bool(mcfg.get("freeze_backbone", False)):
        _freeze_backbone(base, head_modules)
    return base


def build_model(cfg: dict) -> nn.Module:
    """Build a model from a config dict (must contain ``model.name``).

    Supports every name in ``_WEIGHT_ENUMS`` plus the original ``cnn``
    (accepted as ``custom_cnn`` too for backward compatibility only),
    and hybrid models.
    """
    mcfg = cfg["model"]
    name = str(mcfg["name"]).lower()
    num_classes = int(mcfg["num_classes"])

    if name in ("custom_cnn", "cnn"):
        from .cnn import CNN

        return CNN(
            num_classes=num_classes,
            input_channels=int(mcfg.get("input_channels", 3)),
            base_channels=tuple(int(c) for c in mcfg.get("base_channels", (32, 64, 128, 256))),
            dropout=float(mcfg.get("dropout", 0.25)),
        )

    # Check for hybrid models
    if name in ("resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"):
        from .hybrid import build_hybrid_model
        return build_hybrid_model(
            name,
            num_classes=num_classes,
            pretrained=bool(mcfg.get("weights", "imagenet") != "none")
        )

    if name not in _WEIGHT_ENUMS:
        raise ValueError(
            f"unknown model name '{name}' (use cnn, vgg16, resnet50, "
            "mobilenet, efficientnet, or hybrid models: resnet50_mobilenet, "
            "resnet50_efficientnet, resnet50_efficientnet_mobilenet)"
        )

    return _build_transfer_model(mcfg, name, num_classes)


def list_models() -> list:
    """Registered model names (for the backend / CLI).

    Note: custom_cnn is deprecated and not included in this list.
    """
    return [
        "cnn", "vgg16", "resnet50", "mobilenet", "efficientnet",
        "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"
    ]


def count_parameters(model: nn.Module) -> int:
    """Total learnable (requires_grad) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)