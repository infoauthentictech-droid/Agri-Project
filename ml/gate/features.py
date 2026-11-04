"""Feature extraction for the plant-vs-other gate.

Every feature is cheap, deterministic and explainable:
- HSV colour statistics (green vegetation, soil/tan, vivid man-made colours)
- texture (RMS gradient detail)
- colorfulness and brightness
- ImageNet-pretrained semantics: total probability over plant/food-crop
  synsets, top-1 class confidence, and whether the top-1/top-5 classes are
  plant-related.

The ImageNet model is a torchvision checkpoint (cached locally - no download
at inference time) and is shared per-process.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: ImageNet-1k synsets that correspond to plants / food crops / natural fruit.
#: Derived from the official torchvision label list (no ambiguous synsets such
#: as "sea cucumber", "cabbage butterfly", "cornet").
PLANT_SYNSETS: set = {
    936,  # head cabbage
    937,  # broccoli
    938,  # cauliflower
    939,  # zucchini
    940,  # spaghetti squash
    941,  # acorn squash
    942,  # butternut squash
    943,  # cucumber
    944,  # artichoke
    945,  # bell pepper
    947,  # mushroom
    949,  # strawberry
    950,  # orange
    951,  # lemon
    952,  # fig
    953,  # pineapple
    954,  # banana
    955,  # jackfruit
    956,  # custard apple
    958,  # hay
    984,  # rapeseed
    985,  # daisy
    986,  # yellow lady's slipper (orchid)
    987,  # corn
    988,  # acorn
    989,  # hip (rose hip)
    990,  # buckeye
}


class ImageNetFeaturizer:
    """Lazily-loaded torchvision ImageNet model for plant-likeness signals."""

    def __init__(self, name: str = "resnet50") -> None:
        self.name = name
        self._model = None
        self._device = None
        self._preprocess = None

    def _load(self):
        if self._model is not None:
            return self._model, self._device
        import torch
        import torchvision.transforms as T
        from torchvision.models import (
            resnet50, mobilenet_v2,
            ResNet50_Weights, MobileNet_V2_Weights,
        )
        family = {
            "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
            "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
        }[self.name]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = family[0](weights=family[1]).to(device).eval()
        self._preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._model, self._device = model, device
        logger.info(
            "Plant-gate ImageNet featurizer '%s' loaded on %s", self.name, device
        )
        return model, device

    def signal(self, rgb: np.ndarray) -> Dict:
        """Return ImageNet-based plant signals for an RGB frame."""
        try:
            import torch
            model, device = self._load()
            img = Image.fromarray(rgb.astype(np.uint8))
            x = self._preprocess(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            top_indices = np.argsort(probs)[::-1]
            top1 = int(top_indices[0])
            top5 = [int(i) for i in top_indices[:5]]
            return {
                "imagenet_plant_score": float(probs[list(PLANT_SYNSETS)].sum()),
                "imagenet_top1_conf": float(probs[top1]),
                "imagenet_top1_is_plant": float(int(top1 in PLANT_SYNSETS)),
                "imagenet_top5_has_plant": float(any(i in PLANT_SYNSETS for i in top5)),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ImageNet featurizer unavailable (%s)", exc)
            return {
                "imagenet_plant_score": 0.0,
                "imagenet_top1_conf": 0.0,
                "imagenet_top1_is_plant": 0.0,
                "imagenet_top5_has_plant": 0.0,
            }
def hsv_fractions(rgb: np.ndarray) -> Tuple[float, float, float]:
    """(green_fraction, soil_fraction, vivid_fraction) via HSV bands."""
    r = rgb[..., 0].astype(np.float32) / 255.0
    g = rgb[..., 1].astype(np.float32) / 255.0
    b = rgb[..., 2].astype(np.float32) / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    sat = np.divide(diff, np.maximum(mx, 1e-6), out=np.zeros_like(diff),
                    where=mx > 1e-6)
    val = mx

    h = np.zeros_like(mx)
    mask = diff > 1e-6
    dr = np.divide(g - b, diff, out=np.zeros_like(h), where=mask)
    dg = np.divide(b - r, diff, out=np.zeros_like(h), where=mask)
    db = np.divide(r - g, diff, out=np.zeros_like(h), where=mask)
    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)
    h[rmask] = 60.0 * (dr[rmask] % 6.0)
    h[gmask] = 60.0 * (dg[gmask] + 2.0)
    h[bmask] = 60.0 * (db[bmask] + 4.0)

    green = ((h >= 35) & (h <= 90) & (sat >= 0.25) & (val >= 0.12)).mean()
    soil = ((h >= 8) & (h <= 25) & (sat >= 0.20) & (val >= 0.20)).mean()
    vivid = (sat >= 0.60).mean()
    return float(green), float(soil), float(vivid)


def rms_gradient(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(np.float32))
    return float(np.sqrt((gx * gx + gy * gy).mean()))


def colorfulness(rgb: np.ndarray) -> float:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


#: Feature name order used by the trained model.
FEATURE_NAMES: List[str] = [
    "green_fraction", "soil_fraction", "vivid_fraction", "detail_rms",
    "colorfulness", "brightness",
    "imagenet_plant_score", "imagenet_top1_conf", "imagenet_top1_is_plant",
    "imagenet_top5_has_plant",
    "det_best_conf", "det_max_cover",
]


def extract_features(
    rgb: np.ndarray,
    detections: Optional[List[dict]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    featurizer: Optional[ImageNetFeaturizer] = None,
) -> Dict:
    """Extract the fixed-order feature dict for one decoded image."""
    green, soil, vivid = hsv_fractions(rgb)
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    feats: Dict[str, float] = {
        "green_fraction": green,
        "soil_fraction": soil,
        "vivid_fraction": vivid,
        "detail_rms": rms_gradient(gray),
        "colorfulness": colorfulness(rgb),
        "brightness": float(gray.mean() / 255.0),
    }
    if featurizer is not None:
        feats.update(featurizer.signal(rgb))
    else:
        for k in ("imagenet_plant_score", "imagenet_top1_conf",
                  "imagenet_top1_is_plant", "imagenet_top5_has_plant"):
            feats[k] = 0.0

    # Detector-derived signals (best box confidence + frame coverage).
    best_conf, max_cover = 0.0, 0.0
    w, h = image_size or (1, 1)
    area = max(1.0, float(w) * float(h))
    for det in detections or []:
        conf = float(det.get("confidence", 0.0))
        best_conf = max(best_conf, conf)
        bb = det.get("bounding_box") or {}
        x1 = float(bb.get("x1", 0.0))
        y1 = float(bb.get("y1", 0.0))
        x2 = float(bb.get("x2", x1))
        y2 = float(bb.get("y2", y1))
        cover = max(0.0, x2 - x1) * max(0.0, y2 - y1) / area
        max_cover = max(max_cover, cover)
    feats["det_best_conf"] = best_conf
    feats["det_max_cover"] = max_cover
    return feats


def feature_vector(feats: Dict) -> np.ndarray:
    return np.array([feats.get(name, 0.0) for name in FEATURE_NAMES],
                    dtype=np.float32)