"""Centralized Preprocessing for All Models

Ensures consistent preprocessing across training, evaluation, and serving pipelines.
Prevents silent mismatches in resize, normalization, or color-channel conventions.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

# ImageNet normalization constants (standard for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ColorSpace(str, Enum):
    """Color space types."""
    RGB = "RGB"
    GRAY = "GRAY"
    RGBA = "RGBA"


class ResizeMethod(str, Enum):
    """Resize methods."""
    RESIZE = "resize"  # Simple resize (may distort)
    LETTERBOX = "letterbox"  # Add padding to maintain aspect ratio
    CENTER_CROP = "center_crop"  # Crop from center
    PAD = "pad"  # Pad to exact size


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""

    target_size: Tuple[int, int] = (224, 224)
    resize_method: ResizeMethod = ResizeMethod.LETTERBOX
    color_space: ColorSpace = ColorSpace.RGB
    normalize: bool = True
    mean: Optional[list] = None
    std: Optional[list] = None
    maintain_aspect_ratio: bool = True
    interpolation: str = "BILINEAR"

    def __post_init__(self):
        """Set default normalization values."""
        if self.normalize and self.mean is None:
            self.mean = IMAGENET_MEAN
        if self.normalize and self.std is None:
            self.std = IMAGENET_STD


class ImagePreprocessor:
    """Centralized image preprocessor."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize preprocessor.

        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()

    def decode_base64(self, base64_data: str) -> Image.Image:
        """Decode base64 image.

        Args:
            base64_data: Base64-encoded image

        Returns:
            PIL Image
        """
        image_bytes = base64.b64decode(base64_data, validate=True)
        image = Image.open(io.BytesIO(image_bytes))
        return image

    def ensure_color_space(self, image: Image.Image) -> Image.Image:
        """Ensure image is in target color space.

        Args:
            image: PIL Image

        Returns:
            PIL Image in target color space
        """
        if self.config.color_space == ColorSpace.RGB:
            if image.mode not in ["RGB", "RGBA"]:
                image = image.convert("RGB")
            elif image.mode == "RGBA":
                # Convert RGBA to RGB by compositing on white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
        elif self.config.color_space == ColorSpace.GRAY:
            image = image.convert("L")
        elif self.config.color_space == ColorSpace.RGBA:
            if image.mode != "RGBA":
                image = image.convert("RGBA")

        return image

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image according to configured method.

        Args:
            image: PIL Image

        Returns:
            Resized PIL Image
        """
        target_w, target_h = self.config.target_size
        original_w, original_h = image.size

        if self.config.resize_method == ResizeMethod.RESIZE:
            return image.resize((target_w, target_h), getattr(Image, self.config.interpolation))

        elif self.config.resize_method == ResizeMethod.LETTERBOX:
            # Calculate scaling to fit within target while maintaining aspect ratio
            scale = min(target_w / original_w, target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            # Resize
            resized = image.resize((new_w, new_h), getattr(Image, self.config.interpolation))

            # Create new image with target size and paste resized image centered
            new_image = Image.new("RGB", (target_w, target_h), (128, 128, 128))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            new_image.paste(resized, (paste_x, paste_y))

            return new_image

        elif self.config.resize_method == ResizeMethod.CENTER_CROP:
            # Calculate crop to maintain aspect ratio
            aspect_ratio = original_w / original_h
            target_aspect = target_w / target_h

            if aspect_ratio > target_aspect:
                # Image is wider, crop sides
                new_h = original_h
                new_w = int(original_h * target_aspect)
                left = (original_w - new_w) // 2
                top = 0
            else:
                # Image is taller, crop top/bottom
                new_w = original_w
                new_h = int(original_w / target_aspect)
                left = 0
                top = (original_h - new_h) // 2

            cropped = image.crop((left, top, left + new_w, top + new_h))
            return cropped.resize((target_w, target_h), getattr(Image, self.config.interpolation))

        elif self.config.resize_method == ResizeMethod.PAD:
            # Resize maintaining aspect ratio, then pad
            scale = min(target_w / original_w, target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)

            resized = image.resize((new_w, new_h), getattr(Image, self.config.interpolation))

            # Pad to target size
            new_image = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            new_image.paste(resized, (paste_x, paste_y))

            return new_image

        return image

    def normalize(self, image_array: np.ndarray) -> np.ndarray:
        """Normalize image array.

        Args:
            image_array: Image array (H, W, C) in range [0, 255]

        Returns:
            Normalized array (H, W, C) in range approximately [-2, 2]
        """
        if not self.config.normalize:
            return image_array

        # Convert to float and scale to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0

        # Normalize with mean and std
        mean = np.array(self.config.mean, dtype=np.float32)
        std = np.array(self.config.std, dtype=np.float32)

        normalized = (image_array - mean[None, None, :]) / std[None, None, :]

        return normalized

    def preprocess(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Complete preprocessing pipeline.

        Args:
            image: Input image (base64 string, PIL Image, or numpy array)

        Returns:
            Preprocessed numpy array
        """
        # Decode if base64
        if isinstance(image, str):
            image = self.decode_base64(image)

        # Ensure color space
        image = self.ensure_color_space(image)

        # Resize
        image = self.resize_image(image)

        # Convert to numpy array
        image_array = np.asarray(image, dtype=np.float32)

        # Normalize
        image_array = self.normalize(image_array)

        return image_array

    def preprocess_for_model(self, model_name: str, image: Union[str, Image.Image]) -> np.ndarray:
        """Preprocess image for a specific model.

        Args:
            model_name: Name of model (mobilenet, efficientnet, yolov8n, etc.)
            image: Input image

        Returns:
            Preprocessed array
        """
        # Get model-specific config from registry
        from backend.app.utils.model_registry import get_registry

        registry = get_registry()
        model_config = registry.get_model(model_name)

        if model_config:
            # Use model-specific config
            config = PreprocessingConfig(
                target_size=(model_config.image_size, model_config.image_size),
                normalize=model_config.normalize,
                mean=model_config.mean,
                std=model_config.std,
            )
            self.config = config

        return self.preprocess(image)


# Predefined configs for common models
CLASSIFICATION_CONFIG = PreprocessingConfig(
    target_size=(128, 128),
    resize_method=ResizeMethod.LETTERBOX,
    color_space=ColorSpace.RGB,
    normalize=True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
)

DETECTION_CONFIG = PreprocessingConfig(
    target_size=(640, 640),
    resize_method=ResizeMethod.LETTERBOX,
    color_space=ColorSpace.RGB,
    normalize=False,  # YOLO handles normalization internally
)

SEGMENTATION_CONFIG = PreprocessingConfig(
    target_size=(640, 640),
    resize_method=ResizeMethod.LETTERBOX,
    color_space=ColorSpace.RGB,
    normalize=False,  # YOLO handles normalization internally
)

PLANT_GATE_CONFIG = PreprocessingConfig(
    target_size=(256, 256),
    resize_method=ResizeMethod.LETTERBOX,
    color_space=ColorSpace.RGB,
    normalize=True,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
)


def get_preprocessor_for_task(task: str) -> ImagePreprocessor:
    """Get preprocessor for a specific task.

    Args:
        task: Task name (classification, detection, segmentation, plant_gate)

    Returns:
        ImagePreprocessor instance
    """
    configs = {
        "classification": CLASSIFICATION_CONFIG,
        "detection": DETECTION_CONFIG,
        "segmentation": SEGMENTATION_CONFIG,
        "plant_gate": PLANT_GATE_CONFIG,
    }

    config = configs.get(task, CLASSIFICATION_CONFIG)
    return ImagePreprocessor(config)
