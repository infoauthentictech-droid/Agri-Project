"""
File validation utilities.
"""

import base64
import io
from typing import Tuple, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4096
ALLOWED_FORMATS = ['JPEG', 'PNG', 'JPG']

#: Base64 inflates binary by ~4/3. A 10 MB image encodes to ~13.4 MB of ASCII;
#: allowing a little headroom guards decode work without accepting junk.
MAX_BASE64_CHARS = int(MAX_IMAGE_SIZE_MB * (1024 * 1024 * 4 / 3) * 1.05)


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_image_file(image_data: str) -> Tuple[bool, Optional[str]]:
    """
    Validate base64 encoded image file.
    
    Args:
        image_data: Base64 encoded image string
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Reject absurdly large base64 inputs before decoding (DoS guard).
        if len(image_data) > MAX_BASE64_CHARS:
            return False, f"Image size exceeds maximum of {MAX_IMAGE_SIZE_MB}MB"

        # Decode base64
        image_bytes = base64.b64decode(image_data, validate=True)
        
        # Check file size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return False, f"Image size exceeds maximum of {MAX_IMAGE_SIZE_MB}MB"
        
        # Open image to validate format
        image = Image.open(io.BytesIO(image_bytes))
        
        # Validate format
        if image.format not in ALLOWED_FORMATS:
            return False, f"Image format {image.format} not allowed. Allowed: {ALLOWED_FORMATS}"
        
        # Validate dimensions
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image dimensions exceed maximum of {MAX_IMAGE_DIMENSION}px"
        
        return True, None
        
    except base64.binascii.Error:
        return False, "Invalid base64 encoding"
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False, f"Image validation failed: {str(e)}"


def validate_image_size(image_data: str, max_size_mb: int = MAX_IMAGE_SIZE_MB) -> Tuple[bool, Optional[str]]:
    """
    Validate image file size.
    
    Args:
        image_data: Base64 encoded image string
        max_size_mb: Maximum size in MB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        image_bytes = base64.b64decode(image_data)
        size_mb = len(image_bytes) / (1024 * 1024)
        
        if size_mb > max_size_mb:
            return False, f"Image size {size_mb:.2f}MB exceeds maximum of {max_size_mb}MB"
        
        return True, None
        
    except base64.binascii.Error:
        return False, "Invalid base64 encoding"


def validate_image_format(image_data: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate image format.
    
    Args:
        image_data: Base64 encoded image string
    
    Returns:
        Tuple of (is_valid, error_message, format)
    """
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.format not in ALLOWED_FORMATS:
            return False, f"Format {image.format} not allowed", image.format
        
        return True, None, image.format
        
    except base64.binascii.Error:
        return False, "Invalid base64 encoding", None
    except Exception as e:
        return False, f"Format validation failed: {str(e)}", None
