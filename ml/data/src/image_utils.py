"""Image loading, integrity validation and (optional) transformation utilities.

Raw files are only ever *read*; any transformed copy is written to a separate
processed directory so the original data stays untouched.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass
class ImageInfo:
    """Result of validating one image file (read-only inspection)."""

    path: str
    valid: bool
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    size_bytes: int = 0
    error: Optional[str] = None


def inspect_image(path: Path) -> ImageInfo:
    """Open, decode and verify an image without modifying it.

    ``verify()`` catches truncated/corrupt files; dimensions and mode are read
    from the validated image.
    """
    path = Path(path)
    size_bytes = path.stat().st_size if path.exists() else 0
    info = ImageInfo(path=str(path), valid=False, size_bytes=size_bytes)
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            info.width, info.height = im.size
            info.mode = im.mode
        info.valid = True
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError) as exc:  # noqa: PERF203
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def resize_image(
    src: Path,
    dst: Path,
    target: Tuple[int, int],
    keep_aspect: bool = True,
    resample: int = Image.Resampling.LANCZOS,
) -> ImageInfo:
    """Resize *and write a new file*; the source is never touched.

    ``keep_aspect=True`` uses ``thumbnail`` (aspect-preserving, never upscales
    beyond the target); the result is saved as JPEG quality 95.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if keep_aspect:
            im.thumbnail((int(target[0]), int(target[1])), resample)
        else:
            im = im.resize((int(target[0]), int(target[1])), resample)
        im.save(dst, format="JPEG", quality=95)
    return inspect_image(dst)


def copy_image(src: Path, dst: Path, target: Optional[Tuple[int, int]] = None) -> ImageInfo:
    """Copy an image into the processed tree; optionally resize in the copy.

    Plain copies (``target is None``) skip re-verification because the source
    was already validated by the pipeline; resized copies are verified after
    writing.
    """
    dst = Path(dst)
    if target is None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return ImageInfo(
            path=str(dst),
            valid=True,
            size_bytes=dst.stat().st_size,
        )
    return resize_image(src, dst, target)


def resize_mask(src: Path, dst: Path, target: Tuple[int, int]) -> None:
    """Resize a class-index / label mask with NEAREST sampling (lossless ids)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.resize((int(target[0]), int(target[1])), Image.Resampling.NEAREST)
        im.save(dst, format="PNG")


def normalize_image(image: Image.Image) -> np.ndarray:
    """Normalize an RGB image to float32 in [0, 1] (H, W, 3).

    This is a load-time transform for the training pipeline; normalized tensors
    are never written back to disk so raw values remain reproducible.
    """
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def parse_size(text: str) -> Tuple[int, int]:
    """Parse a 'WxH' string (e.g. '512x512') into a (width, height) tuple."""
    w, h = text.lower().split("x")
    return int(w), int(h)
