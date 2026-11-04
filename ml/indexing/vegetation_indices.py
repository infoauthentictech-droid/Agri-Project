"""
Vegetation indices for agricultural image analysis.

This module implements vegetation indices commonly used in remote sensing
and agricultural monitoring:

1. NDVI (Normalized Difference Vegetation Index):
   - Used with multispectral imagery (NIR and Red bands)
   - Formula: (NIR - Red) / (NIR + Red)
   - Range: -1 to 1, where higher values indicate healthier vegetation

2. ExG (Excess Green Index):
   - Used with standard RGB cameras
   - Formula: 2*G - R - B
   - Effective for separating green vegetation from soil

These indices are essential for:
- Assessing plant health and biomass
- Monitoring crop growth stages
- Detecting stress or disease
- Supporting precision agriculture decisions
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2


def calculate_ndvi(
    nir_band: np.ndarray,
    red_band: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index (NDVI).

    NDVI is the most widely used vegetation index for multispectral imagery.
    It measures the difference between near-infrared (which vegetation reflects
    strongly) and red light (which vegetation absorbs).

    Formula: NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir_band: Near-infrared band as 2D numpy array (H, W)
        red_band: Red band as 2D numpy array (H, W)
        epsilon: Small value to prevent division by zero

    Returns:
        NDVI index map as numpy array (H, W) with values in range [-1, 1]

    Raises:
        ValueError: If input arrays have different shapes
    """
    if nir_band.shape != red_band.shape:
        raise ValueError("NIR and Red bands must have the same shape")

    # Convert to float for calculation
    nir = nir_band.astype(np.float32)
    red = red_band.astype(np.float32)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red + epsilon)

    return ndvi


def calculate_exg(image: np.ndarray) -> np.ndarray:
    """Calculate Excess Green (ExG) vegetation index.

    ExG is designed for RGB imagery and effectively separates green vegetation
    from soil and other non-vegetative elements.

    Formula: ExG = 2*G - R - B

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


def calculate_exgr(image: np.ndarray) -> np.ndarray:
    """Calculate Excess Green minus Excess Red (ExGR) index.

    ExGR is an improved version of ExG that better distinguishes vegetation
    from soil and other background elements.

    Formula: ExGR = ExG - 1.4 * (R - B)

    Args:
        image: Input RGB image (H, W, 3) as numpy array

    Returns:
        ExGR index map as numpy array (H, W) with float values
    """
    # Extract color channels
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)

    # Calculate ExG
    exg = 2.0 * g - r - b

    # Calculate ExGR
    exgr = exg - 1.4 * (r - b)

    return exgr


def calculate_vegetation_index(
    image: Optional[np.ndarray] = None,
    nir_band: Optional[np.ndarray] = None,
    red_band: Optional[np.ndarray] = None,
    index_type: str = "exg"
) -> np.ndarray:
    """Calculate vegetation index based on available data.

    This function automatically selects the appropriate vegetation index
    based on the input data type and user preference.

    Args:
        image: RGB image (H, W, 3) for ExG/ExGR calculation
        nir_band: Near-infrared band (H, W) for NDVI calculation
        red_band: Red band (H, W) for NDVI calculation
        index_type: Type of index to calculate ("ndvi", "exg", "exgr")

    Returns:
        Vegetation index map as numpy array

    Raises:
        ValueError: If required inputs are not provided or index_type is invalid
    """
    index_type = index_type.lower()

    if index_type == "ndvi":
        if nir_band is None or red_band is None:
            raise ValueError("NDVI requires both nir_band and red_band")
        return calculate_ndvi(nir_band, red_band)

    elif index_type == "exg":
        if image is None:
            raise ValueError("ExG requires RGB image")
        return calculate_exg(image)

    elif index_type == "exgr":
        if image is None:
            raise ValueError("ExGR requires RGB image")
        return calculate_exgr(image)

    else:
        raise ValueError(f"Unknown index type: {index_type}. Use 'ndvi', 'exg', or 'exgr'")


def classify_vegetation_health(
    vegetation_index: np.ndarray,
    index_type: str = "ndvi"
) -> Tuple[np.ndarray, dict]:
    """Classify vegetation health based on index values.

    This function interprets vegetation index values to classify vegetation
    into health categories.

    Args:
        vegetation_index: Vegetation index map (H, W)
        index_type: Type of index ("ndvi" or "exg")

    Returns:
        Tuple of (classification_map, statistics)
        - classification_map: Integer array with class labels
        - statistics: Dictionary with health category statistics
    """
    if index_type.lower() == "ndvi":
        # NDVI thresholds for health classification
        # Reference: standard NDVI interpretation guidelines
        thresholds = {
            "unhealthy": (-1.0, 0.2),
            "sparse": (0.2, 0.4),
            "moderate": (0.4, 0.6),
            "healthy": (0.6, 0.8),
            "very_healthy": (0.8, 1.0)
        }
    else:
        # ExG thresholds (empirically determined for RGB imagery)
        thresholds = {
            "unhealthy": (-255.0, 0.0),
            "sparse": (0.0, 30.0),
            "moderate": (30.0, 60.0),
            "healthy": (60.0, 100.0),
            "very_healthy": (100.0, 255.0)
        }

    # Initialize classification map
    classification = np.zeros_like(vegetation_index, dtype=np.int32)

    # Apply thresholds
    class_labels = list(thresholds.keys())
    for i, (category, (min_val, max_val)) in enumerate(thresholds.items()):
        mask = (vegetation_index >= min_val) & (vegetation_index < max_val)
        classification[mask] = i

    # Calculate statistics
    stats = {}
    total_pixels = classification.size
    for i, category in enumerate(class_labels):
        count = np.sum(classification == i)
        stats[category] = {
            "pixel_count": int(count),
            "percentage": float(count / total_pixels * 100)
        }

    return classification, stats


def calculate_density_map(
    vegetation_index: np.ndarray,
    index_type: str = "ndvi",
    threshold: Optional[float] = None
) -> np.ndarray:
    """Calculate vegetation density map.

    This function creates a density map showing the proportion of vegetation
    in different regions of the image.

    Args:
        vegetation_index: Vegetation index map (H, W)
        index_type: Type of index ("ndvi" or "exg")
        threshold: Threshold for vegetation detection (None for automatic)

    Returns:
        Density map as numpy array (H, W) with values in range [0, 1]
    """
    if threshold is None:
        # Use automatic threshold based on index type
        if index_type.lower() == "ndvi":
            threshold = 0.3  # Common NDVI threshold for vegetation
        else:
            threshold = 40.0  # Common ExG threshold for vegetation

    # Create binary vegetation mask
    vegetation_mask = (vegetation_index > threshold).astype(np.float32)

    # Apply smoothing to create density map
    density_map = cv2.GaussianBlur(vegetation_mask, (15, 15), 0)

    return density_map


def get_index_statistics(vegetation_index: np.ndarray) -> dict:
    """Calculate statistics for a vegetation index map.

    Args:
        vegetation_index: Vegetation index map (H, W)

    Returns:
        Dictionary with statistical measures
    """
    stats = {
        "mean": float(np.mean(vegetation_index)),
        "std": float(np.std(vegetation_index)),
        "min": float(np.min(vegetation_index)),
        "max": float(np.max(vegetation_index)),
        "median": float(np.median(vegetation_index)),
        "percentile_25": float(np.percentile(vegetation_index, 25)),
        "percentile_75": float(np.percentile(vegetation_index, 75)),
    }

    return stats


def main():
    """Demonstration of vegetation indexing."""
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python vegetation_indices.py <image_path> [--ndvi <nir> <red>]")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    # Load RGB image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 1

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate ExG index
    exg = calculate_exg(image)
    exg_stats = get_index_statistics(exg)

    print("ExG Index Statistics:")
    for key, value in exg_stats.items():
        print(f"  {key}: {value:.4f}")

    # Classify vegetation health
    classification, health_stats = classify_vegetation_health(exg, "exg")

    print("\nVegetation Health Classification:")
    for category, stats in health_stats.items():
        print(f"  {category}: {stats['percentage']:.2f}%")

    # Calculate density map
    density = calculate_density_map(exg, "exg")
    print(f"\nAverage vegetation density: {np.mean(density) * 100:.2f}%")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
