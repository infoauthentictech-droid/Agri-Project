"""Preprocessing module for agricultural image analysis.

This module implements Gaussian Filtering + ExG Thresholding for:
- Scrubbing sensor noise with Gaussian filtering
- Isolating vegetation from bare dirt using Excess Green (ExG) thresholding
"""

from .vegetation_isolation import apply_gaussian_filter, calculate_exg, apply_exg_thresholding

__all__ = ["apply_gaussian_filter", "calculate_exg", "apply_exg_thresholding"]
