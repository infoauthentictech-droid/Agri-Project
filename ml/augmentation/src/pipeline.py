"""Config-driven, seed-reproducible augmentation pipeline.

The pipeline applies the transforms listed in ``apply_order`` with the
probabilities/parameter ranges declared in the config. All randomness comes
from a ``random.Random`` (op sampling) and a ``numpy`` PCG64 generator (op
parameters); both are seeded from a single integer, so a fixed seed fully
reproduces an augmentation run.

Geometric transforms are recorded in :attr:`geometry` so segmentation masks
and YOLO boxes can be synchronized exactly (``apply_with_mask`` /
``apply_with_boxes``).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import transforms as T


def load_config(path: Any) -> Dict[str, Any]:
    """Load a JSON augmentation config."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "apply_order" not in cfg:
        raise ValueError(f"config {path} missing 'apply_order'")
    return cfg


class AugmentationPipeline:
    """Apply a config-driven sequence of identity-safe augmentations."""

    GEOMETRIC = {"rotation", "scale", "random_crop", "h_flip", "v_flip"}
    PHOTOMETRIC = {"brightness", "contrast", "hue", "blur", "noise", "shadow", "occlusion", "jpeg", "mosaic", "mixup"}

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None) -> None:
        self.config = config
        self.name = str(config.get("name", "custom"))
        self.seed = seed
        self.geometry = T.GeometryState()
        self._set_rng(seed)

    def _set_rng(self, seed: Optional[int]) -> None:
        self.seed = seed
        if seed is None:
            self._prng = random.Random()
            self._rng = np.random.default_rng()
        else:
            self._prng = random.Random(int(seed))
            self._rng = np.random.default_rng(int(seed))

    def set_seed(self, seed: int) -> None:
        """Re-seed both generators (reproducibility)."""
        self._set_rng(seed)

    # ------------------------------------------------------------------
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Augment one RGB image; records geometry for mask/box sync."""
        img = np.asarray(image)
        if img.ndim != 3:
            raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
        h, w = img.shape[:2]
        self.geometry = T.GeometryState()
        for name in self.config.get("apply_order", []):
            tcfg = self.config.get("transforms", {}).get(name)
            if not tcfg:
                continue
            if not tcfg.get("enabled", True):
                continue
            if self._prng.random() >= float(tcfg.get("p", 1.0)):
                continue
            img = self._apply_named(name, tcfg, img, w, h)
        return img

    def apply_with_mask(self, image: np.ndarray, mask: np.ndarray) -> tuple:
        """Augment an image and apply the same geometry to its mask."""
        image = self.apply(image)
        return image, self.geometry.apply_to_mask(np.asarray(mask))

    def apply_with_boxes(self, image: np.ndarray, boxes: Sequence) -> tuple:
        """Augment an image and transform normalized YOLO boxes [x,y,w,h]."""
        image = self.apply(image)
        h, w = np.asarray(image).shape[:2]
        return image, self.geometry.apply_to_boxes(np.asarray(boxes), (w, h))

    # ------------------------------------------------------------------
    def _apply_named(self, name: str, cfg: Dict[str, Any], img: np.ndarray,
                     w: int, h: int) -> np.ndarray:
        p = cfg.get("params", {})

        if name == "brightness":
            return T.op_brightness(img, self._rng.uniform(*p["range"]))
        if name == "contrast":
            return T.op_contrast(img, self._rng.uniform(*p["range"]))
        if name == "hue":
            return T.op_hue(img, self._rng.uniform(*p["degrees"]))
        if name == "blur":
            return T.op_blur(img, self._rng.uniform(*p["sigma"]))
        if name == "noise":
            return T.op_noise(img, self._rng.uniform(*p["std"]), self._rng)
        if name == "shadow":
            return T.op_shadow(img, self._rng, p["intensity"],
                               p.get("size", (0.3, 0.6)))
        if name == "occlusion":
            return T.op_occlusion(img, self._rng, tuple(p["patches"]),
                                  tuple(p["size"]))
        if name == "jpeg":
            return T.op_jpeg(img, int(self._rng.uniform(*p["quality"])))
        if name == "mosaic":
            return T.op_mosaic(img, self._rng, int(p.get("n_images", 4)),
                              tuple(p.get("center_ratio", (0.4, 0.6))))
        if name == "mixup":
            return T.op_mixup(img, self._rng, tuple(p.get("alpha", (0.2, 0.4))))

        if name == "rotation":
            img, op = T.op_rotate(img, self._rng, tuple(p["degrees"]), (w, h))
            self.geometry.append(op)
            return img
        if name == "scale":
            img, op = T.op_scale(img, self._rng, tuple(p["factor"]), (w, h))
            self.geometry.append(op)
            return img
        if name == "random_crop":
            img, op = T.op_random_crop(img, self._rng, tuple(p["area"]),
                                       tuple(p.get("ratio", (0.9, 1.1))), (w, h))
            self.geometry.append(op)
            return img
        if name == "h_flip":
            img = T.op_flip_h(img)
            self.geometry.append({"op": "flip_h"})
            return img
        if name == "v_flip":
            img = T.op_flip_v(img)
            self.geometry.append({"op": "flip_v"})
            return img

        raise ValueError(f"unknown transform '{name}' in config")

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Compact human-readable summary of enabled transforms."""
        parts = []
        for name in self.config.get("apply_order", []):
            tcfg = self.config.get("transforms", {}).get(name, {})
            if tcfg.get("enabled", True):
                parts.append(f"{name}(p={tcfg.get('p', 1.0)})")
        return f"{self.name}: " + (", ".join(parts) if parts else "no transforms")
