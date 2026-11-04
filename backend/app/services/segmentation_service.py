"""
Semantic & Instance Segmentation service for pixel-level plant/weed masking.

Uses YOLOv8-Seg / YOLO11-Seg to generate pixel-precise segmentation masks
(background/crop/weed). This combines semantic and instance segmentation
into a single, high-speed engine.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Default segmentation model path (YOLO-Seg)
DEFAULT_MODEL_PATH = "experiments/yoloseg/agri_yolov8n_seg/weights/best.pt"

CLASS_NAMES = ["background", "crop", "weed"]


class SegmentationService:
    """Service for YOLO-Seg semantic and instance segmentation inference."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the segmentation service.

        Args:
            model_path: Path to trained YOLO-Seg model weights.
        """
        self._model = None
        self._device = "cuda" if self._is_cuda_available() else "cpu"
        self.model_path = model_path or str(project_root / DEFAULT_MODEL_PATH)
        self.class_names = CLASS_NAMES

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def load_model(self) -> bool:
        """Load the YOLO-Seg segmentation model.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._model is not None:
            return True

        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(
                    "Segmentation model not found at %s",
                    self.model_path,
                )
                return False

            from ultralytics import YOLO

            self._model = YOLO(str(model_file))
            self._model.to(self._device)

            logger.info(
                "YOLO-Seg model loaded from %s (device=%s)",
                self.model_path,
                self._device,
            )
            return True

        except Exception as e:
            logger.error("Failed to load segmentation model: %s", e)
            self._model = None
            return False

    def is_loaded(self) -> bool:
        """Check if the segmentation model is loaded."""
        return self._model is not None

    def segment(
        self,
        image_data: str,
        target_size: int = 224,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> Tuple[bool, Dict, float, Optional[str]]:
        """Run semantic and instance segmentation on an image.

        Args:
            image_data: Base64-encoded image.
            target_size: YOLO inference image size. Expected range: 64–1024.
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU threshold for NMS.

        Returns:
            Tuple of (success, result_dict, inference_time_ms, error_message).
            result_dict contains:
                - mask: base64-encoded PNG RGB visualization mask
                - class_mask: base64-encoded PNG class-index mask
                - classes: list of detected class names
                - coverage: percentage coverage per class
                - class_names: all available class names
                - instances: list of instance metadata
        """
        # Keep service-level validation in place in case this method is called
        # from somewhere other than the validated FastAPI request schema.
        if not isinstance(target_size, int) or not 64 <= target_size <= 1024:
            return (
                False,
                {},
                0.0,
                "target_size must be an integer between 64 and 1024",
            )

        if self._model is None:
            if not self.load_model():
                return False, {}, 0.0, "Segmentation model not loaded"

        try:
            # Decode image
            try:
                image_bytes = base64.b64decode(image_data, validate=True)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                return False, {}, 0.0, f"Invalid image data: {e}"

            image_array = np.asarray(image)
            h, w = image_array.shape[:2]

            # Run inference. `imgsz` is the Ultralytics inference-size argument.
            start_time = time.time()
            results = self._model(
                image_array,
                imgsz=target_size,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self._device,
                verbose=False,
            )
            inference_time = (time.time() - start_time) * 1000

            if not results or len(results) == 0:
                return True, self._empty_result(h, w), float(inference_time), None

            result = results[0]

            # Get masks and detection metadata if available.
            if result.boxes is None or len(result.boxes) == 0:
                return True, self._empty_result(h, w), float(inference_time), None

            boxes = result.boxes.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            else:
                masks = None

            # Semantic mask: 0=background, 1=crop, 2=weed.
            semantic_mask = np.zeros((h, w), dtype=np.uint8)

            # RGB mask: background=black, crop=green, weed=red.
            color_map = {
                0: [0, 0, 0],
                1: [0, 255, 0],
                2: [255, 0, 0],
            }
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

            instances = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box[:4]
                conf = float(confidences[i])

                # Assumes the trained YOLO model uses:
                # class 0 = crop, class 1 = weed.
                # Add one to reserve class ID 0 for background.
                cls_id = int(classes[i]) + 1

                # Prevent an unexpected model class from crashing inference.
                if cls_id not in (1, 2):
                    logger.warning(
                        "Ignoring unexpected segmentation class ID: %s",
                        classes[i],
                    )
                    continue

                if masks is not None and i < len(masks):
                    instance_mask = masks[i]

                    # Resize the mask to the original image dimensions.
                    instance_mask_resized = cv2.resize(
                        instance_mask,
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    instance_mask_binary = (
                        instance_mask_resized > 0.5
                    ).astype(np.uint8)

                    # Preserve existing behavior: later instances overwrite
                    # pixels where instance masks overlap.
                    semantic_mask[instance_mask_binary > 0] = cls_id
                    rgb_mask[instance_mask_binary > 0] = color_map[cls_id]
                else:
                    # Fallback when the loaded model produces boxes but no masks.
                    x1i = max(0, min(w, int(x1)))
                    y1i = max(0, min(h, int(y1)))
                    x2i = max(0, min(w, int(x2)))
                    y2i = max(0, min(h, int(y2)))

                    if x2i > x1i and y2i > y1i:
                        semantic_mask[y1i:y2i, x1i:x2i] = cls_id
                        rgb_mask[y1i:y2i, x1i:x2i] = color_map[cls_id]

                instances.append(
                    {
                        "class_id": cls_id,
                        "class_name": self.class_names[cls_id],
                        "confidence": conf,
                        "bbox": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ],
                    }
                )

            # Calculate coverage from the final semantic mask so percentages
            # reflect actual assigned pixels and do not double-count overlaps.
            total_pixels = h * w
            if total_pixels == 0:
                return False, {}, 0.0, "Decoded image has invalid dimensions"

            background_pixels = int(np.count_nonzero(semantic_mask == 0))
            crop_pixels = int(np.count_nonzero(semantic_mask == 1))
            weed_pixels = int(np.count_nonzero(semantic_mask == 2))

            coverage = {
                "background": round(background_pixels / total_pixels * 100, 2),
                "crop": round(crop_pixels / total_pixels * 100, 2),
                "weed": round(weed_pixels / total_pixels * 100, 2),
            }

            detected_classes = [
                name
                for name in ("crop", "weed")
                if coverage[name] > 0
            ]

            # Encode RGB visualization mask as PNG and then base64.
            rgb_mask_pil = Image.fromarray(rgb_mask)
            rgb_buffer = io.BytesIO()
            rgb_mask_pil.save(rgb_buffer, format="PNG")
            rgb_mask_base64 = base64.b64encode(
                rgb_buffer.getvalue()
            ).decode("ascii")

            # Encode class-index mask as PNG and then base64.
            class_mask_pil = Image.fromarray(semantic_mask)
            class_buffer = io.BytesIO()
            class_mask_pil.save(class_buffer, format="PNG")
            class_mask_base64 = base64.b64encode(
                class_buffer.getvalue()
            ).decode("ascii")

            result_dict = {
                "mask": rgb_mask_base64,
                "class_mask": class_mask_base64,
                "classes": detected_classes,
                "coverage": coverage,
                "class_names": self.class_names,
                "instances": instances,
            }

            return True, result_dict, float(inference_time), None

        except Exception:
            logger.exception("Segmentation inference failed")
            return False, {}, 0.0, "Segmentation failed. Check server logs."

    def _empty_result(self, height: int, width: int) -> Dict:
        """Build a valid empty segmentation response for an image."""
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

        rgb_buffer = io.BytesIO()
        Image.fromarray(rgb_mask).save(rgb_buffer, format="PNG")
        rgb_mask_base64 = base64.b64encode(
            rgb_buffer.getvalue()
        ).decode("ascii")

        class_buffer = io.BytesIO()
        Image.fromarray(empty_mask).save(class_buffer, format="PNG")
        class_mask_base64 = base64.b64encode(
            class_buffer.getvalue()
        ).decode("ascii")

        return {
            "mask": rgb_mask_base64,
            "class_mask": class_mask_base64,
            "classes": [],
            "coverage": {
                "background": 100.0,
                "crop": 0.0,
                "weed": 0.0,
            },
            "class_names": self.class_names,
            "instances": [],
        }