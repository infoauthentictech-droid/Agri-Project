"""Model improvement experiments.

This package contains scripts for the recommended experiments:

1. experiment_1_threshold_sweep.py - Confidence threshold sweep
2. experiment_2_missed_weeds.py - Missed weeds analysis
3. experiment_3_data_augmentation.py - Data augmentation guidance
4. experiment_4_resolution_comparison.py - Image resolution comparison
5. experiment_5_efficientnet_validation.py - EfficientNet validation
"""

from .experiment_1_threshold_sweep import main as exp1_main
from .experiment_2_missed_weeds import main as exp2_main
from .experiment_3_data_augmentation import main as exp3_main
from .experiment_4_resolution_comparison import main as exp4_main
from .experiment_5_efficientnet_validation import main as exp5_main

__all__ = [
    "exp1_main",
    "exp2_main",
    "exp3_main",
    "exp4_main",
    "exp5_main",
]
