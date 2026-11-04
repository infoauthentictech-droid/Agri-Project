"""Synthesis module for CycleGAN-based image generation.

This module implements CycleGAN for converting normal crop images to
weed-infested scenarios without needing paired training data.
"""

from .cyclegan import CycleGANModel, CycleGANTrainer

__all__ = ["CycleGANModel", "CycleGANTrainer"]
