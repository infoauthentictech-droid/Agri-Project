"""Production plant-vs-other image gate.

Trained by ``ml/gate/train_plant_gate.py`` (RandomForest on HSV + texture +
ImageNet-pretrained features). This wrapper is what ``/analyze`` uses to decide
whether an uploaded photo plausibly contains a crop field *before* any
detection/classification result is shown.

Decision (two-tier):
- probability < 0.40  -> reject ("not a crop-field photo")
- probability >= 0.65 -> accept
- 0.40-0.65 (borderline) -> accept only when a strong detector box (>=0.45)
  coincides with vegetation/soil colour, or the image has high texture; this
  guarantees real field photos with moderately confident boxes are never lost.

Fallback: if the trained model is missing, the transparent heuristic in
``app/utils/domain.py`` is used instead so the API never 500s.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# The gate's features live in the top-level ``ml`` package.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
#: Gate trained on the same (hybrid) dataset as the served detector. Override
#: with GROUNDNUT_PLANT_GATE_MODEL to serve a different training run.
_MODEL_PATH = Path(os.environ.get(
    "GROUNDNUT_PLANT_GATE_MODEL",
    _PROJECT_ROOT / "experiments/by_dataset/hybrid/plant_gate/plant_gate_model.joblib",
))

#: ml/gate/features_gpu.py extracts training features with the longest side
#: capped at 256 px; texture/colour features depend on resolution, so serving
#: must apply the same cap.
_CAP_SIZE = 256


def _cap_resolution(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= _CAP_SIZE:
        return rgb
    scale = _CAP_SIZE / max(h, w)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))


class PlantImageGate:
    """Reusable gate singleton (model loaded lazily once per process)."""

    def __init__(self) -> None:
        self._clf = None
        self._threshold = 0.5
        self._feature_names: List[str] = []
        self._featurizer = None

    def _load(self) -> bool:
        if self._clf is not None:
            return True
        if not _MODEL_PATH.exists():
            logger.warning("Plant gate model missing at %s - using heuristic "
                           "fallback.", _MODEL_PATH)
            return False
        try:
            import joblib
            from ml.gate.features import ImageNetFeaturizer
            payload = joblib.load(_MODEL_PATH)
            self._clf = payload["model"]
            self._threshold = float(payload.get("threshold", 0.5))
            self._feature_names = list(payload.get("feature_names", []))
            self._featurizer = ImageNetFeaturizer("resnet50")
            logger.info("Plant gate model loaded (threshold=%.4f, metrics=%s)",
                        self._threshold, payload.get("metrics"))
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load plant gate model: %s", exc)
            return False

    def preload(self) -> None:
        """Eagerly load the gate model + ImageNet featurizer.

        Called during backend startup so the first ``/analyze`` request does not
        pay the one-time resnet50 checkpoint load (which can take tens of
        seconds and blow past the client's request timeout). Loading is cached
        and idempotent; failures degrade to the heuristic fallback.
        """
        self._load()

    def assess(
        self,
        rgb: np.ndarray,
        detections: Optional[List[dict]] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Decide whether an RGB frame plausibly contains a crop field."""
        loaded = self._load()
        if loaded:
            try:
                from ml.gate.features import extract_features
                feats = extract_features(_cap_resolution(rgb), detections,
                                         image_size, self._featurizer)
                vec = np.array(
                    [feats.get(name, 0.0) for name in self._feature_names],
                    dtype=np.float32,
                ).reshape(1, -1)
                proba = float(self._clf.predict_proba(vec)[0, 1])
                best_conf = max((float(d.get("confidence", 0.0))
                                 for d in detections or []), default=0.0)
                plant_frac = feats.get("green_fraction", 0.0)
                soil_frac = feats.get("soil_fraction", 0.0)
                detail = feats.get("detail_rms", 0.0)

                # Two-tier decision: firmly reject clearly non-plant images
                # (proba < 0.40), firmly accept clearly plant images
                # (proba >= 0.65), and resolve the ambiguous middle
                # (0.40-0.65) with detector + colour evidence so real field
                # photos with moderately confident boxes are never lost.
                is_plant = proba >= 0.65 or (
                    0.40 <= proba < 0.65
                    and ((best_conf >= 0.45 and
                          (plant_frac >= 0.04 or soil_frac >= 0.05))
                         or detail >= 25.0)
                )
                reason = ""
                if not is_plant:
                    reason = _explain_reject(proba, plant_frac, soil_frac, detail)
                return {
                    "is_plant": is_plant,
                    "probability": round(proba, 4),
                    "threshold": round(self._threshold, 4),
                    "decision_bands": {"low": 0.40, "high": 0.65},
                    "model": "plant_gate_rf",
                    "reasons": [reason] if reason else [],
                    "green_fraction": round(plant_frac, 4),
                    "soil_fraction": round(soil_frac, 4),
                    "detail_rms": round(detail, 3),
                }
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Plant gate inference failed (%s); falling back "
                               "to heuristic.", exc)

        # Heuristic fallback (greener/saturated cues), never a hard crash.
        # Normalised to the same keys as the model path so callers can rely on
        # ``is_plant`` / ``probability`` / ``threshold`` either way.
        from app.utils.domain import assess_agricultural_content
        h, w = rgb.shape[:2]
        heuristic = assess_agricultural_content(rgb, detections or [],
                                                image_size or (w, h))
        is_plant = bool(heuristic["is_agricultural"])
        return {
            **heuristic,
            "is_plant": is_plant,
            "probability": 1.0 if is_plant else 0.0,
            "threshold": 0.5,
            "model": "heuristic",
        }


def _explain_reject(proba: float, green: float, soil: float,
                    detail: float) -> str:
    if green < 0.015 and soil < 0.03:
        return (
            "This image has almost no vegetation or soil colour - it does not "
            "appear to be a crop-field photo. Please retake / upload a clear "
            "photo of the groundnut field or weeds."
        )
    if detail < 3.0:
        return (
            "This image is too flat/featureless to be a crop-field photo. "
            "Please retake / upload a clear photo of the groundnut field."
        )
    return (
        "This image does not look like a crop-field photo (only weak "
        f"plant content detected, confidence {proba:.2f}). Please retake / "
        "upload a clear photo of the groundnut field or weeds."
    )