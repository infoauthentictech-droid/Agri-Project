"""Vegetation indexing module for agricultural image analysis.

This module implements vegetation indices for vegetation detection:
- NDVI (Normalized Difference Vegetation Index) for multispectral imagery
- ExG (Excess Green) for standard RGB cameras

These indices are used to:
- Quantify vegetation health and density
- Separate vegetation from soil and background
- Support decision-making in agricultural applications
"""

from .vegetation_indices import calculate_ndvi, calculate_exg, calculate_vegetation_index

__all__ = ["calculate_ndvi", "calculate_exg", "calculate_vegetation_index"]
