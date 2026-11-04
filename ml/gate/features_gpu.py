"""GPU version of the plant-gate features, used when training the gate.

Produces the same feature vector as ``ml/gate/features.py``, which the backend keeps using at
inference so the API still works on machines without a GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import v2

from ml.data.gpu_loader import IMAGENET_MEAN, IMAGENET_STD, decode_file
from ml.gate.features import PLANT_SYNSETS, FEATURE_NAMES, ImageNetFeaturizer

CAP_SIZE = 256  # longest side, matching the size the gate has always been trained on


def hsv_fractions(rgb: torch.Tensor) -> tuple:
    """(green, soil, vivid) fractions of a CxHxW float image in [0, 1]."""
    r, g, b = rgb
    mx = rgb.max(dim=0).values
    diff = mx - rgb.min(dim=0).values
    sat = torch.where(mx > 1e-6, diff / mx.clamp(min=1e-6), torch.zeros_like(mx))
    safe = diff.clamp(min=1e-6)
    hue = torch.zeros_like(mx)
    mask = diff > 1e-6
    # Same precedence as the CPU version: later bands overwrite earlier ones.
    hue = torch.where(mask & (mx == r), 60.0 * torch.remainder((g - b) / safe, 6.0), hue)
    hue = torch.where(mask & (mx == g), 60.0 * ((b - r) / safe + 2.0), hue)
    hue = torch.where(mask & (mx == b), 60.0 * ((r - g) / safe + 4.0), hue)
    green = ((hue >= 35) & (hue <= 90) & (sat >= 0.25) & (mx >= 0.12)).float().mean()
    soil = ((hue >= 8) & (hue <= 25) & (sat >= 0.20) & (mx >= 0.20)).float().mean()
    vivid = (sat >= 0.60).float().mean()
    return green.item(), soil.item(), vivid.item()


def colorfulness(rgb255: torch.Tensor) -> float:
    r, g, b = rgb255
    rg = (r - g).abs()
    yb = (0.5 * (r + g) - b).abs()
    return (torch.sqrt(rg.std(unbiased=False) ** 2 + yb.std(unbiased=False) ** 2)
            + 0.3 * torch.sqrt(rg.mean() ** 2 + yb.mean() ** 2)).item()


def rms_gradient(gray: torch.Tensor) -> float:
    gy, gx = torch.gradient(gray)
    return torch.sqrt((gx * gx + gy * gy).mean()).item()


class GpuGateFeatures:
    """Turn an image file into the gate's feature vector, all on the GPU."""

    def __init__(self, imagenet: str = "resnet50", device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available. Feature extraction is GPU-only.")
        self.device = device
        self.model, _ = ImageNetFeaturizer(imagenet)._load()
        self.preprocess = v2.Compose([
            v2.Resize(256, antialias=True),
            v2.CenterCrop(224),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.synsets = torch.tensor(sorted(PLANT_SYNSETS), device=device)

    def extract(self, path: Path) -> np.ndarray:
        image, _ = decode_file(path, self.device)
        height, width = image.shape[1:]
        if max(height, width) > CAP_SIZE:
            scale = CAP_SIZE / max(height, width)
            image = v2.functional.resize(image, [max(1, int(height * scale)), max(1, int(width * scale))],
                                         antialias=True)
        if min(image.shape[1:]) < 8:
            raise ValueError("Image too small")
        rgb255 = image.float()
        rgb = rgb255 / 255.0
        gray = (rgb255[0] * 299 + rgb255[1] * 587 + rgb255[2] * 114 + 500).div(1000).floor()
        green, soil, vivid = hsv_fractions(rgb)
        with torch.inference_mode():
            probs = torch.softmax(self.model(self.preprocess(rgb).unsqueeze(0)), dim=1)[0]
        top5 = probs.topk(5).indices
        in_plants = torch.isin(top5, self.synsets)
        values = {
            "green_fraction": green,
            "soil_fraction": soil,
            "vivid_fraction": vivid,
            "detail_rms": rms_gradient(gray),
            "colorfulness": colorfulness(rgb255),
            "brightness": (gray.mean() / 255.0).item(),
            "imagenet_plant_score": probs[self.synsets].sum().item(),
            "imagenet_top1_conf": probs[top5[0]].item(),
            "imagenet_top1_is_plant": float(in_plants[0].item()),
            "imagenet_top5_has_plant": float(in_plants.any().item()),
            "det_best_conf": 0.0,
            "det_max_cover": 0.0,
        }
        return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
