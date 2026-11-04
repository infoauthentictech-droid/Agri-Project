"""Segmentation mask validation (Peanut-dataset RGB masks).

Semantics (from the dataset README, cited in the audit): Green = crop
(groundnut), Red = weed, Black = background. Masks are RGB PNGs; this module
validates image/mask correspondence and dimensions, derives which classes are
present, and flags empty masks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class MaskResult:
    image: str
    mask: str
    valid: bool = True
    size_match: bool = True
    image_size: Optional[tuple] = None
    mask_size: Optional[tuple] = None
    mode: Optional[str] = None
    mode_error: Optional[str] = None
    colors: List = field(default_factory=list)
    class_presence: List[str] = field(default_factory=list)
    empty: bool = False
    errors: List[str] = field(default_factory=list)


def classify_color(r: int, g: int, b: int) -> str:
    """Map an RGB pixel to background / crop / weed / other."""
    if r < 40 and g < 40 and b < 40:
        return "background"
    if g > 90 and g > r + 30 and g > b + 30:
        return "crop"
    if r > 90 and r > g + 30 and r > b + 30:
        return "weed"
    return "other"


def inspect_mask(image_path: Path, mask_path: Path) -> MaskResult:
    """Validate one image/mask pair and derive the classes present."""
    res = MaskResult(image=str(image_path), mask=str(mask_path))
    try:
        with Image.open(image_path) as im:
            res.image_size = im.size
        with Image.open(mask_path) as mk:
            res.mask_size = mk.size
            res.mode = mk.mode
            arr = np.asarray(mk.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        res.valid = False
        res.errors.append(f"cannot open mask: {type(exc).__name__}: {exc}")
        return res

    if res.image_size != res.mask_size:
        res.size_match = False
        res.valid = False
        res.errors.append(
            f"size mismatch image={res.image_size} mask={res.mask_size}"
        )
    if res.mode not in ("RGB", "P", "L"):
        res.mode_error = res.mode
        res.valid = False
        res.errors.append(f"unexpected mask mode {res.mode}")

    small = arr[::8, ::8].reshape(-1, 3)
    present: set = set()
    for px in small:
        present.add(classify_color(int(px[0]), int(px[1]), int(px[2])))
    present.discard("other")
    res.colors = sorted(present)
    res.class_presence = res.colors
    res.empty = res.colors == ["background"]
    if res.empty:
        res.errors.append("mask contains no foreground (only background)")
    return res
