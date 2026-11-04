"""ML model service — single-model YOLO object detection.

The application deliberately uses exactly ONE detection model — the locally
trained ``yolov8n`` checkpoint that scored the best on validation
(``agri_yolov8n`` run, best mAP50-95 ≈ 0.545). Keeping a single variant ensures
the app, density engine, and Grad-CAM explainer always operate on the same detector.

The weights are resolved relative to the project root, so the service works
regardless of the current working directory. The model is loaded lazily and
cached; inference feeds numpy arrays straight to Ultralytics (no temp files).

The default variant keeps the legacy model path as a fallback so existing
deployments remain bit-compatible if the best checkpoint is ever moved:
``runs/detect/experiments/yolo/agri_yolov8n/weights/best.pt``.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO  # noqa: E402

logger = logging.getLogger(__name__)

# Default dataset config that defines the authoritative class mapping for the
# detector. Names are read from the YAML so the backend can never silently
# diverge from the dataset used to train the model.
DEFAULT_DATA_YAML = "datasets/hybrid/detection/data.yaml"

#: Detection model: name -> candidate weight paths (relative to project root).
#: The sole model is the locally trained ``yolov8n`` checkpoint that scored the
#: best on validation (the ``agri_yolov8n`` 60-epoch run, best mAP50-95 ≈ 0.545).
#: The legacy ``agri_detection`` path is kept as a second candidate on purpose:
#: both point at the SAME trained detector, so removing/renaming folders never
#: breaks an existing deployment.
DETECTION_DEFS: Dict[str, Dict[str, object]] = {
    # Best trained detector (single model used app-wide).
    "yolov8n": {
        "weights": [
            "experiments/yolo/agri_yolov8n/weights/best.pt",
            "runs/detect/experiments/yolo/agri_yolov8n/weights/best.pt",
        ],
        "family": "yolov8",
    },
}

DEFAULT_DETECTION_MODEL = "yolov8n"

#: Allowed detection-variant names (tight allow-list). Requests carrying any
#: other ``model_name`` are rejected before any filesystem path is built, so a
#: user-provided string can never traverse directories.
ALLOWED_DETECTION_MODELS = frozenset(DETECTION_DEFS.keys())

#: Process-wide YOLO instance cache keyed by resolved weights path. Both
#: ModelService and XAIService load their checkpoints through :func:`load_yolo`
#: so the heavy .pt file is loaded into memory exactly once per process.
_YOLO_LOADER_CACHE: Dict[str, "YOLO"] = {}


def load_yolo(path: Path) -> "YOLO":
    """Load a YOLO checkpoint once per process and return the shared instance."""
    key = str(Path(path).resolve())
    cached = _YOLO_LOADER_CACHE.get(key)
    if cached is None:
        cached = YOLO(key)
        _YOLO_LOADER_CACHE[key] = cached
    return cached


def load_class_names(data_yaml: str) -> Dict[int, str]:
    """Read class-name mapping from the Ultralytics data config.

    If the file is missing or unreadable we return placeholder names so the
    model output remains honest (``class_0`` / ``class_1``) instead of
    inventing crop/weed semantics that have not been confirmed.
    """
    p = Path(data_yaml)
    if not p.exists():
        logger.warning("Class-name data config not found at %s; using placeholders", data_yaml)
        return {0: "class_0", 1: "class_1"}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        names = cfg.get("names") or {}
        return {int(k): str(v) for k, v in names.items()}
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Failed to parse class names from %s: %s", data_yaml, e)
        return {0: "class_0", 1: "class_1"}


def _decode_rgb(image_data: str) -> "Image.Image":
    image_bytes = base64.b64decode(image_data)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _boxes_to_list(result, class_names: Dict[int, str]) -> list:
    """Convert an Ultralytics result into the wire format list."""
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        detections.append({
            "class_id": cls_id,
            "class_name": class_names.get(cls_id, f"class_{cls_id}"),
            "confidence": round(conf, 6),
            "bounding_box": {
                "x1": round(float(xyxy[0]), 2),
                "y1": round(float(xyxy[1]), 2),
                "x2": round(float(xyxy[2]), 2),
                "y2": round(float(xyxy[3]), 2),
            },
        })
    return detections
class ModelService:
    """Service for YOLO model inference (multiple detection variants)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_yaml: Optional[str] = None,
        default_model: str = DEFAULT_DETECTION_MODEL,
    ) -> None:
        """Initialize the detection model service.

        Args:
            model_path: (legacy) explicit weights path; overrides the default
                variant's weights when provided.
            data_yaml: dataset config holding the class-name mapping.
            default_model: variant used when callers omit ``model_name``.

        The constructor is intentionally cheap — no YOLO checkpoint is touched
        until :meth:`load_model` is called (e.g. by the FastAPI lifespan pre-warm
        or by the first inference request).
        """
        self.default_model = default_model
        self._models: Dict[str, YOLO] = {}
        # name -> list of candidate weight paths (first existing one wins).
        self._candidate_paths: Dict[str, List[Path]] = {
            name: [project_root / p for p in defn["weights"]]
            for name, defn in DETECTION_DEFS.items()
        }
        # Legacy explicit path support (XAIService / tests pass one).
        if model_path:
            self._candidate_paths[DEFAULT_DETECTION_MODEL] = [Path(model_path)]
        self.class_names = load_class_names(
            data_yaml or str(project_root / DEFAULT_DATA_YAML)
        )

    def _resolve_weights(self, name: str) -> Optional[Path]:
        """Raise on unknown variants, else first existing candidate weight path."""
        if name not in ALLOWED_DETECTION_MODELS:
            raise ValueError(
                f"unknown detection model '{name}'; allowed: "
                + ", ".join(sorted(ALLOWED_DETECTION_MODELS))
            )
        paths = self._candidate_paths[name]
        for p in paths:
            if p.exists():
                return p
        return paths[0]

    # -- registry ----------------------------------------------------------
    @property
    def available_models(self) -> List[str]:
        """Detection variants whose weight files actually exist."""
        return [n for n, ps in self._candidate_paths.items()
                if any(p.exists() for p in ps)]

    def model_status(self) -> Dict[str, dict]:
        """Per-variant availability + load status (for the /models endpoint).

        Never exposes the server-side filesystem path of the weights.
        """
        return {
            name: {
                "available": any(p.exists() for p in ps),
                "loaded": name in self._models,
                "family": DETECTION_DEFS[name].get("family", "yolo"),
            }
            for name, ps in self._candidate_paths.items()
        }

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Load a detection variant (cached).

        Returns:
            True if the model is loaded (or was already loaded), False if the
            model_name is unknown, the weights are missing, or loading failed.
        """
        name = model_name or self.default_model
        if name in self._models:
            return True
        try:
            path = self._resolve_weights(name)
        except ValueError as e:
            logger.warning("%s", e)
            return False
        try:
            if path.exists():
                model = load_yolo(path)
                embedded = getattr(model, "names", None)
                if embedded and all(isinstance(v, str) for v in embedded.values()):
                    self.class_names = {int(k): str(v) for k, v in embedded.items()}
                self._models[name] = model
                logger.info("Detection model '%s' loaded from %s names=%s",
                            name, path, self.class_names)
                return True
            logger.warning("Model weights not found at %s", path)
            return False
        except Exception as e:
            logger.error("Failed to load detection model '%s': %s", name, e)
            return False

    def is_loaded(self, model_name: Optional[str] = None) -> bool:
        """Check if a detection model is loaded."""
        name = model_name or self.default_model
        return name in self._models
# -- inference ---------------------------------------------------------
    def predict(
        self,
        image_data: str,
        confidence_threshold: float = 0.5,
        model_name: Optional[str] = None,
    ) -> Tuple[bool, List[Dict], float, Optional[str]]:
        """Run object-detection ``predict`` on an image (legacy alias)."""
        return self.detect(image_data, confidence_threshold,
                           iou_threshold=0.45, model_name=model_name or "yolov8n")

    def detect(
        self,
        image_data: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        model_name: Optional[str] = None,
    ) -> Tuple[bool, List[Dict], float, Optional[str]]:
        """Run object detection on an image.

        Args:
            image_data: Base64 encoded image.
            confidence_threshold: Minimum confidence.
            iou_threshold: NMS IoU threshold.
            model_name: Detection variant to use (defaults to the service default).

        Returns:
            Tuple of (success, detections, inference_time_ms, error_message).
        """
        name = model_name or self.default_model
        if name not in self._models:
            if not self.load_model(name):
                return False, [], 0.0, f"Detection model '{name}' not loaded"

        try:
            image = _decode_rgb(image_data)
            start_time = time.time()
            results = self._models[name](
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            inference_time = (time.time() - start_time) * 1000

            detections = []
            for result in results:
                detections.extend(_boxes_to_list(result, self.class_names))

            # Check for low confidence
            if detections:
                avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
                if avg_confidence < 0.3:
                    logger.warning("Low confidence detection: %.2f", avg_confidence)

            return True, detections, float(inference_time), None
        except Exception as e:
            logger.error("Detection error (%s): %s", name, e)
            return False, [], 0.0, f"Detection failed: {e}"