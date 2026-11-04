"""Augmentation transforms for agricultural field imagery.

Design principle: **preserve plant identity.** Every transform stays within
physically plausible field-capture variation: small rotation (camera tilt),
horizontal-only flips (vertical flips are unrealistic for row-crop photos),
modest scale/crop factors, daylight-range photometric changes, and realistic
shadow / occlusion / blur / JPEG capture artifacts.

All operations are numpy-uint8 RGB (H, W, 3). Geometric operations return a
recorded op-dict so the identical geometry can be replayed on segmentation
masks or YOLO boxes (see :class:`GeometryState`).
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST


def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(img).astype(np.uint8))


def _from_pil(pil: Image.Image) -> np.ndarray:
    return np.asarray(pil.convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Photometric transforms (images only; masks/boxes unaffected)
# ---------------------------------------------------------------------------

def op_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative brightness (ImageEnhance). factor ~ 0.7-1.3."""
    return _from_pil(ImageEnhance.Brightness(_to_pil(img)).enhance(float(factor)))


def op_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Contrast adjustment (ImageEnhance). factor ~ 0.7-1.3."""
    return _from_pil(ImageEnhance.Contrast(_to_pil(img)).enhance(float(factor)))


def op_hue(img: np.ndarray, delta_deg: float) -> np.ndarray:
    """Small hue rotation (degrees) simulating sun-angle / plant tint shift."""
    hsv = _to_pil(img).convert("HSV")
    h, s, v = hsv.split()
    d = int(round(float(delta_deg) / 360.0 * 255.0)) % 256
    h = h.point(lambda x: (x + d) % 256)
    return _from_pil(Image.merge("HSV", (h, s, v)))


def op_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur (hand-shake / misfocus). sigma ~ 0.5-1.5 px."""
    return _from_pil(_to_pil(img).filter(ImageFilter.GaussianBlur(float(sigma))))


def op_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian sensor noise on float [0,1], clipped."""
    arr = img.astype(np.float32) / 255.0
    arr = arr + rng.normal(0.0, float(std), size=arr.shape)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def op_shadow(img: np.ndarray, rng: np.random.Generator,
              intensity_range: Tuple[float, float],
              size_range: Tuple[float, float] = (0.3, 0.6)) -> np.ndarray:
    """Soft-edged elliptical shadow (cloud / neighbouring-plant shade)."""
    h, w = img.shape[:2]
    cx = rng.uniform(0.2, 0.8)
    cy = rng.uniform(0.2, 0.8)
    rx = w * rng.uniform(*size_range) / 2.0
    ry = h * rng.uniform(*size_range) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    d = ((xx - cx * w) / rx) ** 2 + ((yy - cy * h) / ry) ** 2
    alpha = np.clip(1.0 - d, 0.0, 1.0)[..., None]
    intensity = rng.uniform(*intensity_range)
    dark = img.astype(np.float32) * (1.0 - alpha * intensity)
    return np.clip(dark, 0.0, 255.0).astype(np.uint8)


def op_occlusion(img: np.ndarray, rng: np.random.Generator,
                 n_patches: Tuple[int, int],
                 size_range: Tuple[float, float]) -> np.ndarray:
    """Simulate leaf-on-leaf occlusion: blurred patches of the scene are
    displaced over the plant (a natural 'blocked view', never artificial
    shapes that could imply a new object)."""
    out = img.copy()
    h, w = out.shape[:2]
    n = int(rng.integers(int(n_patches[0]), int(n_patches[1]) + 1))
    for _ in range(n):
        pw = max(1, int(w * rng.uniform(*size_range)))
        ph = max(1, int(h * rng.uniform(*size_range)))
        px = int(rng.integers(0, max(1, w - pw)))
        py = int(rng.integers(0, max(1, h - ph)))
        patch = out[py:py + ph, px:px + pw].copy()
        sigma = rng.uniform(2.0, 6.0)
        patch = _from_pil(_to_pil(patch).filter(ImageFilter.GaussianBlur(sigma)))
        dx = int(rng.integers(-int(pw * 0.2), int(pw * 0.2) + 1))
        dy = int(rng.integers(-int(ph * 0.2), int(ph * 0.2) + 1))
        sx = min(max(0, px + dx), max(0, w - pw))
        sy = min(max(0, py + dy), max(0, h - ph))
        out[sy:sy + ph, sx:sx + pw] = patch
    return out


def op_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Re-compress through JPEG to simulate app/upload artifacts."""
    buf = io.BytesIO()
    _to_pil(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return _from_pil(Image.open(buf))


def op_mosaic(img: np.ndarray, rng: np.random.Generator,
              n_images: int = 4, center_ratio: Tuple[float, float] = (0.4, 0.6)) -> np.ndarray:
    """Mosaic augmentation: combines 4 images into one mosaic grid.

    This forces the model to learn overlapping leaf edges and occlusion patterns.
    For this implementation, we simulate mosaic by dividing the image into a grid
    and applying variations to each quadrant.
    """
    h, w = img.shape[:2]
    half_h, half_w = h // 2, w // 2

    # Create mosaic by dividing image into quadrants with slight variations
    mosaic = img.copy()

    # Top-left quadrant - apply slight brightness variation
    tl = mosaic[:half_h, :half_w]
    tl_brightness = rng.uniform(0.9, 1.1)
    mosaic[:half_h, :half_w] = np.clip(tl * tl_brightness, 0, 255).astype(np.uint8)

    # Top-right quadrant - apply slight contrast variation
    tr = mosaic[:half_h, half_w:]
    tr_contrast = rng.uniform(0.9, 1.1)
    tr_mean = tr.mean(axis=(0, 1), keepdims=True)
    mosaic[:half_h, half_w:] = np.clip((tr - tr_mean) * tr_contrast + tr_mean, 0, 255).astype(np.uint8)

    # Bottom-left quadrant - apply slight hue variation
    bl = mosaic[half_h:, :half_w]
    bl = _to_pil(bl).convert("HSV")
    b_h, b_s, b_v = bl.split()
    d = int(rng.uniform(-10, 10))
    b_h = b_h.point(lambda x: (x + d) % 256)
    mosaic[half_h:, :half_w] = _from_pil(Image.merge("HSV", (b_h, b_s, b_v)))

    # Bottom-right quadrant - slight saturation variation
    br = mosaic[half_h:, half_w:]
    br = _to_pil(br).convert("HSV")
    b_h, b_s, b_v = br.split()
    sat_factor = rng.uniform(0.9, 1.1)
    b_s = b_s.point(lambda x: min(255, int(x * sat_factor)))
    mosaic[half_h:, half_w:] = _from_pil(Image.merge("HSV", (b_h, b_s, b_v)))

    return mosaic


def op_mixup(img: np.ndarray, rng: np.random.Generator,
             alpha: Tuple[float, float] = (0.2, 0.4)) -> np.ndarray:
    """Mixup augmentation: blends image with a slightly transformed version.

    This helps the model learn varied light exposures and smooth transitions.
    Uses lambda sampled from Beta(alpha, alpha) distribution.
    """
    alpha_val = rng.uniform(alpha[0], alpha[1])
    lam = rng.beta(alpha_val, alpha_val)

    # Create a transformed version of the image
    img_transformed = img.copy()

    # Apply random transformations to create the second image
    brightness_factor = rng.uniform(0.8, 1.2)
    img_transformed = np.clip(img_transformed * brightness_factor, 0, 255).astype(np.uint8)

    # Mix the two images
    mixed = (lam * img.astype(np.float32) + (1 - lam) * img_transformed.astype(np.float32))
    return np.clip(mixed, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometric transforms (images AND masks/boxes — recorded for replay)
# ---------------------------------------------------------------------------

def op_random_crop(img: np.ndarray, rng: np.random.Generator,
                   area_range: Tuple[float, float], ratio_range: Tuple[float, float],
                   target: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
    """Random-resized crop back to ``target`` (framing/zoom variation)."""
    h, w = img.shape[:2]
    th, tw = int(target[1]), int(target[0])
    area = h * w * rng.uniform(*area_range)
    ratio = rng.uniform(*ratio_range)
    crop_h = int(round(np.sqrt(area * ratio)))
    crop_w = int(round(np.sqrt(area / ratio)))
    crop_w = min(max(1, crop_w), w)
    crop_h = min(max(1, crop_h), h)
    x0 = int(rng.integers(0, w - crop_w + 1))
    y0 = int(rng.integers(0, h - crop_h + 1))
    crop = img[y0:y0 + crop_h, x0:x0 + crop_w]
    resized = _from_pil(_to_pil(crop).resize((tw, th), BILINEAR))
    op = {"op": "crop", "box": (x0, y0, crop_w, crop_h), "target": (tw, th)}
    return resized, op



def op_scale(img: np.ndarray, rng: np.random.Generator,
             factor_range: Tuple[float, float],
             target: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
    """Zoom in/out (resize by factor, then center-crop or edge-pad to target)."""
    h, w = img.shape[:2]
    th, tw = int(target[1]), int(target[0])
    f = float(rng.uniform(*factor_range))
    nw = max(1, int(round(w * f)))
    nh = max(1, int(round(h * f)))
    resized = _from_pil(_to_pil(img).resize((nw, nh), BILINEAR))
    if nh >= th and nw >= tw:  # zoom-in -> center crop
        x0 = (nw - tw) // 2
        y0 = (nh - th) // 2
        out = resized[y0:y0 + th, x0:x0 + tw]
        op = {"op": "scale", "factor": f, "crop_origin": (x0, y0),
              "orig": (w, h), "target": (tw, th)}
    else:  # zoom-out -> edge-pad to target
        px0 = max(0, (tw - nw) // 2)
        px1 = max(0, (tw - nw) - px0)
        py0 = max(0, (th - nh) // 2)
        py1 = max(0, (th - nh) - py0)
        padded = np.pad(resized, ((py0, py1), (px0, px1), (0, 0)), mode="edge")
        out = padded[:th, :tw]
        op = {"op": "scale", "factor": f, "pad": (px0, py0),
              "orig": (w, h), "target": (tw, th)}
    return out, op


def op_rotate(img: np.ndarray, rng: np.random.Generator,
              angle_range: Tuple[float, float],
              target: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
    """Small rotation about the image center (expand + center-crop)."""
    h, w = img.shape[:2]
    th, tw = int(target[1]), int(target[0])
    angle = float(rng.uniform(*angle_range))
    pil = _to_pil(img).rotate(angle, expand=True, resample=BILINEAR)
    nw, nh = pil.size
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    out = _from_pil(pil.crop((x0, y0, x0 + tw, y0 + th)))
    op = {"op": "rotate", "angle": angle, "orig": (w, h), "crop_origin": (x0, y0)}
    return out, op


def op_flip_h(img: np.ndarray) -> np.ndarray:
    """Horizontal (left-right) mirror — safe for in-field row-crop photos."""
    return img[:, ::-1].copy()


def op_flip_v(img: np.ndarray) -> np.ndarray:
    """Vertical flip — physically unrealistic for field photos (sky/soil)."""
    return img[::-1, :].copy()



# ---------------------------------------------------------------------------
# GeometryState — record & replay of geometry on masks and YOLO boxes
# ---------------------------------------------------------------------------

class GeometryState:
    """Ordered list of geometric ops applied to an image.

    The same ops can be replayed onto a segmentation mask (exact, NEAREST
    interpolation) or YOLO boxes (N x [x_center, y_center, w, h], normalized)
    so that image/label geometry stays synchronized during training.
    """

    def __init__(self) -> None:
        self.ops: list = []

    def append(self, op: dict) -> None:
        self.ops.append(op)

    @property
    def empty(self) -> bool:
        return len(self.ops) == 0

    def apply_to_mask(self, mask: np.ndarray) -> np.ndarray:
        """Replay the geometry on a (H, W) or (H, W, C) mask/array."""
        m = np.asarray(mask)
        for op in self.ops:
            if op["op"] == "crop":
                x0, y0, cw, ch = op["box"]
                tw, th = op["target"]
                m = _resize_nearest(m[y0:y0 + ch, x0:x0 + cw], (tw, th))
            elif op["op"] == "scale":
                f = op["factor"]
                h, w = m.shape[:2]
                nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
                m = _resize_nearest(m, (nw, nh))
                tw, th = op["target"]
                if "crop_origin" in op:
                    x0, y0 = op["crop_origin"]
                    m = m[y0:y0 + th, x0:x0 + tw]
                else:
                    px0, py0 = op["pad"]
                    tw, th = op["target"]
                    nw, nh = m.shape[1], m.shape[0]
                    px1 = max(0, (tw - nw) - px0)
                    py1 = max(0, (th - nh) - py0)
                    padcfg = ((py0, py1), (px0, px1)) + ((0, 0),) * (m.ndim - 2)
                    m = np.pad(m, padcfg, mode="edge")
                    m = m[:th, :tw]
            elif op["op"] == "rotate":
                tw, th = op["orig"]
                m = _rotate_nearest(m, op["angle"], (tw, th))
            elif op["op"] == "flip_h":
                m = m[:, ::-1].copy()
            elif op["op"] == "flip_v":
                m = m[::-1, :].copy()
        return m

    def apply_to_boxes(self, boxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Replay geometry on normalized YOLO boxes N x [x, y, w, h]."""
        b = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
        w, h = float(size[0]), float(size[1])
        for op in self.ops:
            if op["op"] == "crop":
                x0, y0, cw, ch = op["box"]
                b[:, 0] = (b[:, 0] * w - x0) / cw
                b[:, 1] = (b[:, 1] * h - y0) / ch
                b[:, 2] = b[:, 2] * w / cw
                b[:, 3] = b[:, 3] * h / ch
                w, h = float(cw), float(ch)
            elif op["op"] == "scale":
                f = op["factor"]
                tw, th = op["target"]
                if "crop_origin" in op:
                    x0, y0 = op["crop_origin"]
                    b[:, 0] = (b[:, 0] * w * f - x0) / tw
                    b[:, 1] = (b[:, 1] * h * f - y0) / th
                else:
                    px, py = op["pad"]
                    b[:, 0] = (b[:, 0] * w * f + px) / tw
                    b[:, 1] = (b[:, 1] * h * f + py) / th
                b[:, 2] = b[:, 2] * w * f / tw
                b[:, 3] = b[:, 3] * h * f / th
                w, h = float(tw), float(th)
            elif op["op"] == "rotate":
                b = _rotate_boxes(b, op["angle"], w, h)
            elif op["op"] == "flip_h":
                b[:, 0] = 1.0 - b[:, 0]
            elif op["op"] == "flip_v":
                b[:, 1] = 1.0 - b[:, 1]
        b[:, 0] = np.clip(b[:, 0], 0.0, 1.0)
        b[:, 1] = np.clip(b[:, 1], 0.0, 1.0)
        return b


def _resize_nearest(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((int(size[0]), int(size[1])), NEAREST))


def _rotate_nearest(arr: np.ndarray, angle: float, target: Tuple[int, int]) -> np.ndarray:
    tw, th = int(target[0]), int(target[1])
    pil = Image.fromarray(arr).rotate(angle, expand=True, resample=NEAREST)
    nw, nh = pil.size
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return np.asarray(pil.crop((x0, y0, x0 + tw, y0 + th)))


def _rotate_boxes(b: np.ndarray, angle: float, w: float, h: float) -> np.ndarray:
    """Axis-aligned boxes after rotating corners about the image center."""
    a = np.radians(angle)
    c, s = np.cos(a), np.sin(a)
    cx, cy = w / 2.0, h / 2.0
    x, y = b[:, 0] * w, b[:, 1] * h
    bw, bh = b[:, 2] * w, b[:, 3] * h
    corners = np.stack(
        [
            np.stack([x - bw / 2, y - bh / 2], axis=1),
            np.stack([x + bw / 2, y - bh / 2], axis=1),
            np.stack([x + bw / 2, y + bh / 2], axis=1),
            np.stack([x - bw / 2, y + bh / 2], axis=1),
        ],
        axis=1,
    )
    px = corners[..., 0] - cx
    py = corners[..., 1] - cy
    rx = c * px - s * py + cx
    ry = s * px + c * py + cy
    minx, maxx = rx.min(axis=1), rx.max(axis=1)
    miny, maxy = ry.min(axis=1), ry.max(axis=1)
    out = np.stack(
        [((minx + maxx) / 2.0) / w, ((miny + maxy) / 2.0) / h,
         (maxx - minx) / w, (maxy - miny) / h],
        axis=1,
    )
    return out