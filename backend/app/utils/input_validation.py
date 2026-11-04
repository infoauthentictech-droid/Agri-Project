"""Enhanced Input Validation and Security

Provides comprehensive input validation for the backend to handle:
- Malformed base64 data
- Oversized images
- Invalid image dimensions
- Decompression bombs
- Unsupported formats
- Truncated files
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class ValidationError(str, Enum):
    """Types of validation errors."""
    INVALID_BASE64 = "invalid_base64"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_DIMENSIONS = "invalid_dimensions"
    UNSUPPORTED_FORMAT = "unsupported_format"
    TRUNCATED_FILE = "truncated_file"
    DECOMPRESSION_BOMB = "decompression_bomb"
    INVALID_COLOR_SPACE = "invalid_color_space"
    TOO_SMALL = "too_small"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    error_type: Optional[ValidationError] = None
    error_message: str = ""
    image_size: Optional[Tuple[int, int]] = None
    file_size: int = 0
    format: Optional[str] = None


class ImageValidator:
    """Comprehensive image input validator."""

    # Maximum file size in bytes (10 MB for base64 decoded)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Maximum image dimensions
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096

    # Minimum image dimensions
    MIN_WIDTH = 32
    MIN_HEIGHT = 32

    # Supported formats
    SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}

    # Maximum number of pixels (to prevent decompression bombs)
    MAX_PIXELS = 89478485  # ~90 MP (PIL default)

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        max_width: int = MAX_WIDTH,
        max_height: int = MAX_HEIGHT,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
    ):
        """Initialize image validator.

        Args:
            max_file_size: Maximum file size in bytes
            max_width: Maximum image width
            max_height: Maximum image height
            min_width: Minimum image width
            min_height: Minimum image height
        """
        self.max_file_size = max_file_size
        self.max_width = max_width
        self.max_height = max_height
        self.min_width = min_width
        self.min_height = min_height

    def validate_base64_image(self, base64_data: str) -> ValidationResult:
        """Validate a base64-encoded image.

        Args:
            base64_data: Base64-encoded image data

        Returns:
            ValidationResult
        """
        # Check if base64 data is empty
        if not base64_data:
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.INVALID_BASE64,
                error_message="Empty base64 data",
            )

        try:
            # Decode base64
            image_bytes = base64.b64decode(base64_data, validate=True)
        except Exception as e:
            logger.warning(f"Invalid base64 data: {e}")
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.INVALID_BASE64,
                error_message=f"Invalid base64 encoding: {str(e)}",
            )

        # Check file size
        if len(image_bytes) > self.max_file_size:
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.IMAGE_TOO_LARGE,
                error_message=f"Image too large: {len(image_bytes)} bytes (max: {self.max_file_size})",
                file_size=len(image_bytes),
            )

        return self.validate_image_bytes(image_bytes)

    def validate_image_bytes(self, image_bytes: bytes) -> ValidationResult:
        """Validate raw image bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            ValidationResult
        """
        try:
            # Use BytesIO to avoid disk writes
            with io.BytesIO(image_bytes) as buffer:
                # PIL.Image.open will detect decompression bombs
                img = Image.open(buffer)

                # Check format
                if img.format not in self.SUPPORTED_FORMATS:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationError.UNSUPPORTED_FORMAT,
                        error_message=f"Unsupported format: {img.format}. Supported: {self.SUPPORTED_FORMATS}",
                        format=img.format,
                    )

                # Check dimensions
                width, height = img.size

                if width > self.max_width or height > self.max_height:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationError.INVALID_DIMENSIONS,
                        error_message=f"Image too large: {width}x{height} (max: {self.max_width}x{self.max_height})",
                        image_size=(width, height),
                        format=img.format,
                    )

                if width < self.min_width or height < self.min_height:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationError.TOO_SMALL,
                        error_message=f"Image too small: {width}x{height} (min: {self.min_width}x{self.min_height})",
                        image_size=(width, height),
                        format=img.format,
                    )

                # Verify image can be loaded (catches truncated files)
                img.verify()

                # Reload after verify (verify puts image in a closed state)
                buffer.seek(0)
                img = Image.open(buffer)

                # Check color space
                if img.mode not in ["RGB", "RGBA", "L"]:
                    return ValidationResult(
                        is_valid=False,
                        error_type=ValidationError.INVALID_COLOR_SPACE,
                        error_message=f"Unsupported color space: {img.mode}. Supported: RGB, RGBA, L",
                        image_size=(width, height),
                        format=img.format,
                    )

                # Convert to RGB if needed
                if img.mode != "RGB":
                    img = img.convert("RGB")

                return ValidationResult(
                    is_valid=True,
                    image_size=(width, height),
                    file_size=len(image_bytes),
                    format=img.format,
                )

        except Image.DecompressionBombError as e:
            logger.warning(f"Decompression bomb detected: {e}")
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.DECOMPRESSION_BOMB,
                error_message="Decompression bomb detected - image may be malicious",
            )
        except (IOError, OSError) as e:
            logger.warning(f"Truncated or corrupted image: {e}")
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.TRUNCATED_FILE,
                error_message=f"Truncated or corrupted image: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_type=ValidationError.INVALID_BASE64,
                error_message=f"Validation failed: {str(e)}",
            )

    def validate_and_load(self, base64_data: str) -> Tuple[bool, Optional[Image.Image], Optional[str]]:
        """Validate and load image in one call.

        Args:
            base64_data: Base64-encoded image data

        Returns:
            Tuple of (success, PIL_image, error_message)
        """
        result = self.validate_base64_image(base64_data)

        if not result.is_valid:
            return False, None, result.error_message

        try:
            image_bytes = base64.b64decode(base64_data, validate=True)
            with io.BytesIO(image_bytes) as buffer:
                img = Image.open(buffer)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return True, img, None
        except Exception as e:
            return False, None, f"Failed to load image: {str(e)}"


class RequestValidator:
    """Validator for API requests."""

    def __init__(self, image_validator: Optional[ImageValidator] = None):
        """Initialize request validator.

        Args:
            image_validator: Image validator instance
        """
        self.image_validator = image_validator or ImageValidator()

    def validate_segmentation_request(
        self, image_data: str, target_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate segmentation request.

        Args:
            image_data: Base64-encoded image
            target_size: Target size for segmentation

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate image
        result = self.image_validator.validate_base64_image(image_data)
        if not result.is_valid:
            return False, result.error_message

        # Validate target_size if provided
        if target_size is not None:
            if not isinstance(target_size, int):
                return False, "target_size must be an integer"
            if target_size < 64 or target_size > 1024:
                return False, "target_size must be between 64 and 1024"

        return True, None

    def validate_detection_request(
        self, image_data: str, confidence_threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate detection request.

        Args:
            image_data: Base64-encoded image
            confidence_threshold: Confidence threshold

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate image
        result = self.image_validator.validate_base64_image(image_data)
        if not result.is_valid:
            return False, result.error_message

        # Validate confidence_threshold if provided
        if confidence_threshold is not None:
            if not isinstance(confidence_threshold, (int, float)):
                return False, "confidence_threshold must be a number"
            if confidence_threshold < 0.0 or confidence_threshold > 1.0:
                return False, "confidence_threshold must be between 0.0 and 1.0"

        return True, None

    def validate_classification_request(
        self, image_data: str, top_k: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate classification request.

        Args:
            image_data: Base64-encoded image
            top_k: Number of top classes to return

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate image
        result = self.image_validator.validate_base64_image(image_data)
        if not result.is_valid:
            return False, result.error_message

        # Validate top_k if provided
        if top_k is not None:
            if not isinstance(top_k, int):
                return False, "top_k must be an integer"
            if top_k < 1 or top_k > 5:
                return False, "top_k must be between 1 and 5"

        return True, None


# Global validator instances
_image_validator: Optional[ImageValidator] = None
_request_validator: Optional[RequestValidator] = None


def get_image_validator() -> ImageValidator:
    """Get the global image validator.

    Returns:
        ImageValidator instance
    """
    global _image_validator
    if _image_validator is None:
        _image_validator = ImageValidator()
    return _image_validator


def get_request_validator() -> RequestValidator:
    """Get the global request validator.

    Returns:
        RequestValidator instance
    """
    global _request_validator
    if _request_validator is None:
        _request_validator = RequestValidator()
    return _request_validator
