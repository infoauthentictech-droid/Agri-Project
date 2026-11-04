"""Backend-side density estimation from detection results.

This is deliberately a *transparent, uncalibrated* area-based estimate and is
never presented as a scientifically validated weed-coverage measurement:

- Coverage is computed only over boxes whose ``class_id`` is in the configured
  weed-class set (``weed_class_ids``), never over every detection.
- Overlapping boxes are merged (union area) so the same plant counted twice by
  the detector does not inflate the percentage.
- The category thresholds (low < 5%, medium 5-15%, high > 15%) mirror the
  science-tool config in ``ml/density/default_config.json`` and remain
  **uncalibrated** until a field study provides proper calibration data.
- The class labels themselves are still ``class_0``/``class_1`` placeholders
  until the detector label mapping is confirmed, so even the "weed" class id it
  is computed for is a configuration assumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Defaults kept in sync with ml/density/default_config.json.
DEFAULT_WEED_CLASS_IDS = [1]
DEFAULT_LOW_THRESHOLD = 5.0
DEFAULT_MEDIUM_THRESHOLD = 15.0


def _load_density_config() -> Dict:
    try:
        p = Path(__file__).parent.parent.parent.parent / "ml/density/default_config.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_DENSITY_CONFIG = _load_density_config()


def _overlap(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> bool:
    """True when two axis-aligned boxes overlap (IoU > 0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    return ix * iy > 1e-6


def _merge_union_area(boxes: List[Tuple[float, float, float, float]]) -> float:
    """Sum of the union area of the boxes (overlapping boxes merged once)."""
    boxes = [tuple(b) for b in boxes]
    merged: List[Tuple[float, float, float, float]] = []
    for box in boxes:
        if not any(_overlap(box, m) for m in merged):
            merged.append(box)
            continue
        # Merge into the first overlapping region and re-check (small lists).
        merged = [_merge_pair(m, box) if _overlap(m, box) else m for m in merged]
    area = 0.0
    for (x1, y1, x2, y2) in merged:
        area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area


def _merge_pair(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def compute_density_estimate(
    detections: List[dict],
    image_width: float,
    image_height: float,
    weed_class_ids: Optional[List[int]] = None,
) -> Optional[Dict]:
    """Compute an honest (uncalibrated) area-based density estimate.

    Returns None when there are no boxes for the configured weed classes (so
    the API can omit the field), otherwise a JSON-safe dict matching
    ``DensityEstimate``.
    """
    ids = weed_class_ids or _DENSITY_CONFIG.get(
        "weed_class_id", DEFAULT_WEED_CLASS_IDS[0]
    )
    if isinstance(ids, int):
        ids = [ids]
    ids = [int(i) for i in ids]

    boxes: List[Tuple[float, float, float, float]] = []
    for det in detections:
        if int(det.get("class_id", -1)) not in ids:
            continue
        bb = det.get("bounding_box") or {}
        x1 = float(bb.get("x1", 0.0))
        y1 = float(bb.get("y1", 0.0))
        x2 = float(bb.get("x2", 0.0))
        y2 = float(bb.get("y2", 0.0))
        boxes.append((x1, y1, max(x2, x1), max(y2, y1)))

    if not boxes:
        return None

    area = max(1.0, float(image_width) * float(image_height))
    union_area = _merge_union_area(boxes)
    coverage = (union_area / area) * 100.0
    coverage = max(0.0, min(100.0, coverage))

    low_t = float(_DENSITY_CONFIG.get("low_threshold", DEFAULT_LOW_THRESHOLD))
    med_t = float(_DENSITY_CONFIG.get("medium_threshold", DEFAULT_MEDIUM_THRESHOLD))
    if coverage < low_t:
        category = "low"
    elif coverage < med_t:
        category = "medium"
    else:
        category = "high"

    ids_label = ", ".join(str(i) for i in ids)
    return {
        "weed_class_ids": ids,
        "weed_count": len(boxes),
        "coverage_percent": round(coverage, 2),
        "density_category": category,
        "calibrated": False,
        "note": (
            "Uncalibrated area-based estimate (union of boxes for class "
            f"{ids_label}). Detector label mapping is unconfirmed and the "
            "5%/15% thresholds are not field-calibrated."
        ),
    }