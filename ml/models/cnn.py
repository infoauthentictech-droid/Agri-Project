"""CNN — original convolutional network for groundnut-vs-weed classification.

(Formerly named ``CustomCNN``; renamed to ``CNN`` for a cleaner registry.)

Architecture rationale (dataset-driven, not a copied standard net):

- Input: 128x128 RGB field patches (the small plant patches extracted from the
  Peanut segmentation dataset; see ml/data/prepare_classification.py).
- Four convolutional blocks, each Conv3x3 -> BatchNorm -> ReLU -> MaxPool(2),
  doubling channels 32/64/128/256. The small 3x3 stacks are a good fit for
  the modest image size and give an effective receptive field that covers
  local leaf texture while pooling captures plant-level structure.
- Dropout before the global-average-pooling bottleneck to regularize.
- Global Average Pooling -> single linear head (2 classes). GAP avoids huge
  FC parameter counts and is robust to translation.

No pretrained weights are used anywhere; this is a fully original model.

Param/FLOP accounting helpers are included for the experiment ledger.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_ch, out_ch, dropout: float):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(dropout),
    )


class CNN(nn.Module):
    """Groundnut-vs-weed classifier (original architecture)."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 3,
        base_channels: tuple = (32, 64, 128, 256),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout
        blocks = []
        in_ch = input_channels
        for ch in base_channels:
            blocks.append(_conv_block(in_ch, ch, dropout))
            in_ch = ch
        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        # Deterministic, scale-aware initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# Backward-compatible alias so checkpoints / scripts that referenced the old
# class name keep working after the rename.
CustomCNN = CNN


def count_parameters(model: nn.Module) -> int:
    """Total learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    """Approximate in-memory size of the parameters (fp32)."""
    return count_parameters(model) * 4


def build_model(cfg: dict) -> CNN:
    mcfg = cfg["model"]
    return CNN(
        num_classes=int(mcfg["num_classes"]),
        input_channels=int(mcfg.get("input_channels", 3)),
        base_channels=tuple(int(c) for c in mcfg["base_channels"]),
        dropout=float(mcfg.get("dropout", 0.25)),
    )