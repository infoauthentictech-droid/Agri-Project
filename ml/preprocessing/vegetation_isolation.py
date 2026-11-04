"""
Vegetation isolation using Gaussian Filtering + ExG Thresholding.

This module implements preprocessing for agricultural images:
1. Gaussian Filtering: Scrubs sensor noise from images
2. Excess Green (ExG) Thresholding: Isolates vegetation from bare dirt

These preprocessing steps are essential for:
- Improving segmentation accuracy
- Reducing false positives from soil and debris
- Enhancing plant feature extraction
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2
from PIL import Image


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """Apply Gaussian filter to remove sensor noise.

    Gaussian filtering effectively scrubs sensor noise while preserving
    important edges and features in agricultural imagery.

    Args:
        image: Input RGB image (H, W, 3) as numpy array
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian kernel

    Returns:
        Filtered image as numpy array (H, W, 3)
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Apply Gaussian blur to each channel
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered


def calculate_exg(image: np.ndarray) -> np.ndarray:
    """Calculate Excess Green (ExG) vegetation index.

    ExG = 2*G - R - B

    This index effectively separates green vegetation from soil and other
    non-vegetative elements in RGB imagery.

    Args:
        image: Input RGB image (H, W, 3) as numpy array

    Returns:
        ExG index map as numpy array (H, W) with float values
    """
    # Extract color channels
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # Calculate ExG: 2*G - R - B
    exg = 2.0 * g - r - b

    return exg


def apply_exg_thresholding(
    image: np.ndarray,
    threshold: Optional[float] = None,
    otsu: bool = True
) -> Tuple[np.ndarray, float]:
    """Apply ExG thresholding to isolate vegetation.

    This function calculates the ExG index and applies thresholding to
    create a binary mask separating vegetation from background (soil, dirt).

    Args:
        image: Input RGB image (H, W, 3) as numpy array
        threshold: Manual threshold value (if None, uses Otsu's method)
        otsu: Whether to use Otsu's method for automatic threshold selection

    Returns:
        Tuple of (binary_mask, threshold_used)
        - binary_mask: Binary mask (H, W) where 1 = vegetation, 0 = background
        - threshold_used: The threshold value that was applied
    """
    # Calculate ExG index
    exg = calculate_exg(image)

    # Normalize ExG to 0-255 range for thresholding
    exg_normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Determine threshold
    if threshold is None and otsu:
        # Use Otsu's method for automatic threshold selection
        threshold_used, _ = cv2.threshold(
            exg_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold_used = float(threshold_used)
    elif threshold is not None:
        threshold_used = threshold
    else:
        # Default threshold if not specified
        threshold_used = 50.0

    # Apply thresholding
    _, binary_mask = cv2.threshold(
        exg_normalized,
        int(threshold_used),
        255,
        cv2.THRESH_BINARY
    )

    # Convert to binary (0 or 1)
    binary_mask = (binary_mask > 0).astype(np.uint8)

    return binary_mask, threshold_used


def preprocess_image(
    image: np.ndarray,
    gaussian_kernel: int = 5,
    gaussian_sigma: float = 1.0,
    exg_threshold: Optional[float] = None,
    use_otsu: bool = True
) -> dict:
    """Complete preprocessing pipeline: Gaussian filtering + ExG thresholding.

    This combines both preprocessing steps into a single pipeline for
    efficient vegetation isolation from agricultural imagery.

    Args:
        image: Input RGB image (H, W, 3) as numpy array
        gaussian_kernel: Kernel size for Gaussian filter
        gaussian_sigma: Sigma for Gaussian filter
        exg_threshold: Manual ExG threshold (None for automatic)
        use_otsu: Whether to use Otsu's method for threshold selection

    Returns:
        Dictionary containing:
        - 'filtered': Gaussian-filtered image
        - 'exg': ExG index map
        - 'vegetation_mask': Binary vegetation mask
        - 'threshold': Threshold value used
    """
    # Step 1: Apply Gaussian filtering to remove sensor noise
    filtered = apply_gaussian_filter(image, gaussian_kernel, gaussian_sigma)

    # Step 2: Calculate ExG index
    exg = calculate_exg(filtered)

    # Step 3: Apply ExG thresholding to isolate vegetation
    vegetation_mask, threshold = apply_exg_thresholding(
        filtered, exg_threshold, use_otsu
    )

    return {
        "filtered": filtered,
        "exg": exg,
        "vegetation_mask": vegetation_mask,
        "threshold": threshold
    }


def enhance_segmentation_with_preprocessing(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    gaussian_kernel: int = 5,
    gaussian_sigma: float = 1.0
) -> np.ndarray:
    """Enhance segmentation mask using vegetation preprocessing.

    This uses the vegetation mask from ExG thresholding to refine
    segmentation results, removing false positives from soil and debris.

    Args:
        image: Original RGB image
        segmentation_mask: Initial segmentation mask
        gaussian_kernel: Kernel size for Gaussian filter
        gaussian_sigma: Sigma for Gaussian filter

    Returns:
        Refined segmentation mask
    """
    # Get vegetation mask
    result = preprocess_image(image, gaussian_kernel, gaussian_sigma)
    vegetation_mask = result["vegetation_mask"]

    # Combine with segmentation mask (logical AND)
    refined_mask = np.logical_and(segmentation_mask > 0, vegetation_mask > 0)
    refined_mask = refined_mask.astype(np.uint8)

    return refined_mask


def main():
    """Demonstration of preprocessing pipeline."""
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python vegetation_isolation.py <image_path>")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 1

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    result = preprocess_image(image)

    print(f"Preprocessing complete.")
    print(f"Threshold used: {result['threshold']:.2f}")
    print(f"Vegetation coverage: {result['vegetation_mask'].mean() * 100:.2f}%")

    # Save results
    output_dir = image_path.parent / "preprocessing_output"
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(
        str(output_dir / f"{image_path.stem}_filtered.jpg"),
        cv2.cvtColor(result["filtered"], cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        str(output_dir / f"{image_path.stem}_vegetation_mask.png"),
        result["vegetation_mask"] * 255
    )

    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
