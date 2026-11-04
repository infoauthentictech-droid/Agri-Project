"""
XAI service for model explanations.
"""

import base64
import io
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO  # noqa
from ml.xai.api import XAIEngine
from app.services.model_service import (
    DEFAULT_DATA_YAML,
    DEFAULT_DETECTION_MODEL,
    DETECTION_DEFS,
    load_class_names,
)

logger = logging.getLogger(__name__)


def _default_detector_weights() -> Path:
    candidates = [project_root / p for p in DETECTION_DEFS[DEFAULT_DETECTION_MODEL]["weights"]]
    return next((p for p in candidates if p.exists()), candidates[0])


class XAIService:
    """Service for XAI explanations."""
    
    def __init__(self, model_path: Optional[str] = None, output_dir: Optional[Path] = None):
        """
        Initialize XAI service.
        
        Args:
            model_path: Path to trained YOLO model
            output_dir: Directory to save XAI outputs
        """
        self.engine = None
        # Resolve paths relative to the project root so the service works
        # regardless of the current working directory at launch time.
        # Uses the same detector checkpoint as ModelService (first existing
        # candidate path), so Grad-CAM always explains the served model.
        self.model_path = model_path or str(_default_detector_weights())
        self.output_dir = output_dir or (project_root / 'experiments/xai')
        # Read the authoritative mapping from the dataset config (no guesses).
        self.class_names = load_class_names(str(project_root / DEFAULT_DATA_YAML))

    def load_engine(self) -> bool:
        """
        Load XAI engine lazily with its OWN YOLO instance. Grad-CAM registers
        forward/backward hooks on the model, and a hooked model can no longer
        be deep-copied by Ultralytics, so sharing ModelService's cached
        instance breaks every detection request.
        """
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                model = YOLO(str(model_path))
                # Ultralytics deep-copies the model when it first builds its
                # predictor; once Grad-CAM's hooks are attached that copy
                # fails ("cannot pickle '_thread.lock'"). Build the predictor
                # now, before XAIEngine registers the hooks.
                model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                self.engine = XAIEngine(model, self.class_names, self.output_dir)
                logger.info(f"XAI engine loaded with model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load XAI engine: {str(e)}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self.engine is not None

    def _cleanup_request_files(self, saved_files: dict) -> None:
        """Delete files produced for this request and any stale XAI images.

        XAI images are embedded into the JSON response as base64, so keeping
        them on disk for every request would grow disk usage without bound.
        The opportunistic sweep is restricted to IMAGE files so non-image
        research artifacts (e.g. scenario summary JSONs) are never touched.
        """
        image_exts = {".png", ".jpg", ".jpeg", ".svg"}
        try:
            for value in (saved_files or {}).values():
                path = Path(str(value))
                if path.is_file() and path.suffix.lower() in image_exts:
                    path.unlink(missing_ok=True)
        except OSError:
            pass
        # Opportunistic sweep of stale image artifacts (older than an hour,
        # e.g. from a crashed worker). Non-image files are preserved.
        try:
            if self.output_dir.is_dir():
                cutoff = time.time() - 3600
                for p in self.output_dir.iterdir():
                    try:
                        if (p.is_file()
                                and p.suffix.lower() in image_exts
                                and p.stat().st_mtime < cutoff):
                            p.unlink(missing_ok=True)
                    except OSError:
                        continue
        except OSError:
            pass
    
    def explain_image(
        self,
        image_data: str,
        include_visualization: bool = True
    ) -> Dict:
        """
        Generate explanation for image.
        
        Args:
            image_data: Base64 encoded image
            include_visualization: Whether to include visualization data
        
        Returns:
            Dictionary with explanation result
        """
        if not self.is_loaded():
            return {
                'success': False,
                'error': 'XAI engine not loaded'
            }
        
        try:
            # Decode and save image temporarily
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = Path(tmp.name)
            try:
                image.save(temp_path)

                # Generate explanation
                save_prefix = f"api_{temp_path.stem}"
                result = self.engine.explain_image(str(temp_path), save_prefix, include_visualization)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            # Embed a compact base64-encoded overlay + heatmap of the first
            # detected object so the mobile client can render the Grad-CAM
            # visualization directly (server-side files are not reachable).
            visualizations = {}
            try:
                for key in ("overlay_0", "heatmap_0"):
                    path = Path(result.get("saved_files", {}).get(key, ""))
                    if path.exists() and path.stat().st_size <= 4 * 1024 * 1024:
                        try:
                            visualizations[key] = base64.b64encode(
                                path.read_bytes()).decode("ascii")
                        except OSError:
                            continue
            finally:
                # Cleanup: never grow disk by retaining per-request artifacts.
                self._cleanup_request_files(result.get("saved_files", {}))

            # Never disclose absolute server paths. The visualizations are
            # embedded inline, so clients only need the basenames (or nothing).
            sanitized_files = {
                k: Path(str(v)).name for k, v in (result.get("saved_files") or {}).items()
            }

            return {
                'success': result['success'],
                'predictions': result['predictions'],
                'explanations': result['explanations'],
                'saved_files': sanitized_files,
                'visualizations': visualizations,
            }

        except Exception as e:
            logger.error(f"XAI explanation error: {str(e)}")
            return {
                'success': False,
                'error': "Explanation failed. Please try a different image.",
            }
