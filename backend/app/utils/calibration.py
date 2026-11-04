"""Model Confidence Calibration

Implements confidence calibration to ensure model confidence scores are
well-calibrated probabilities. Uncalibrated confidence scores can be misleading
and should not be treated as certainty.

This module provides temperature scaling and Platt scaling for calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class CalibrationResult:
    """Result of calibration evaluation."""
    is_calibrated: bool
    brier_score: float
    expected_calibration_error: float
    reliability_diagram: Dict
    calibration_method: str
    calibration_params: Optional[Dict] = None


class ConfidenceCalibrator:
    """Calibrator for model confidence scores."""

    def __init__(self, method: str = "temperature"):
        """Initialize calibrator.

        Args:
            method: Calibration method ('temperature', 'platt', 'isotonic')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit calibrator on validation data.

        Args:
            logits: Model logits (n_samples, n_classes)
            labels: True labels (n_samples,)
        """
        if self.method == "temperature":
            # Temperature scaling (softmax with temperature parameter)
            # Simplified implementation
            self._fit_temperature_scaling(logits, labels)
        elif self.method == "platt":
            # Platt scaling (logistic regression)
            self.calibrator = CalibratedClassifierCV(
                method="sigmoid", cv="prefit"
            )
            self.calibrator.fit(logits, labels)
        elif self.method == "isotonic":
            # Isotonic regression
            self.calibrator = CalibratedClassifierCV(
                method="isotonic", cv="prefit"
            )
            self.calibrator.fit(logits, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

    def _fit_temperature_scaling(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature scaling.

        Simplified temperature scaling using validation data.
        """
        # For binary classification, use logistic regression on logits
        if logits.shape[1] == 2:
            from sklearn.linear_model import LogisticRegression

            self.calibrator = LogisticRegression()
            self.calibrator.fit(logits, labels)
        else:
            # For multi-class, use sklearn's CalibratedClassifierCV
            self.calibrator = CalibratedClassifierCV(
                method="sigmoid", cv="prefit"
            )
            self.calibrator.fit(logits, labels)

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate confidence scores.

        Args:
            logits: Model logits (n_samples, n_classes)

        Returns:
            Calibrated probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            # If not fitted, return softmax (no calibration)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        if self.method == "temperature" and hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(logits)
        elif hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(logits)
        else:
            # Fallback to softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def evaluate_calibration(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> CalibrationResult:
        """Evaluate calibration quality.

        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
            n_bins: Number of bins for reliability diagram

        Returns:
            CalibrationResult
        """
        # Brier score (lower is better)
        if probs.shape[1] == 2:
            brier = brier_score_loss(labels, probs[:, 1])
        else:
            # Multi-class Brier score
            brier = brier_score_loss(labels, probs)

        # Expected Calibration Error (ECE)
        ece = self._compute_ece(probs, labels, n_bins)

        # Reliability diagram data
        reliability_diagram = self._compute_reliability_diagram(probs, labels, n_bins)

        # Determine if calibrated (ECE < 0.1 is acceptable)
        is_calibrated = ece < 0.1

        calibration_params = None
        if self.is_fitted and hasattr(self.calibrator, "coef_"):
            calibration_params = {"coefficients": self.calibrator.coef_.tolist()}

        return CalibrationResult(
            is_calibrated=is_calibrated,
            brier_score=float(brier),
            expected_calibration_error=float(ece),
            reliability_diagram=reliability_diagram,
            calibration_method=self.method,
            calibration_params=calibration_params,
        )

    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
        """Compute Expected Calibration Error.

        Args:
            probs: Predicted probabilities
            labels: True labels
            n_bins: Number of bins

        Returns:
            ECE score
        """
        if probs.shape[1] == 2:
            confidences = probs[:, 1]
        else:
            # Use max confidence for multi-class
            confidences = np.max(probs, axis=1)

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                bin_confidences = confidences[in_bin]
                bin_accuracies = (labels[in_bin] == np.argmax(probs[in_bin], axis=1)).astype(float)
                avg_confidence = bin_confidences.mean()
                avg_accuracy = bin_accuracies.mean()
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin

        return ece

    def _compute_reliability_diagram(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int
    ) -> Dict:
        """Compute reliability diagram data.

        Args:
            probs: Predicted probabilities
            labels: True labels
            n_bins: Number of bins

        Returns:
            Dictionary with reliability diagram data
        """
        if probs.shape[1] == 2:
            confidences = probs[:, 1]
        else:
            confidences = np.max(probs, axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_counts.append(in_bin.sum())

            if in_bin.sum() > 0:
                bin_confidences.append(confidences[in_bin].mean())
                bin_accuracies.append(
                    (labels[in_bin] == np.argmax(probs[in_bin], axis=1)).astype(float).mean()
                )
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)

        return {
            "bin_boundaries": bin_boundaries.tolist(),
            "bin_confidences": bin_confidences,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }

    def save(self, path: Path) -> None:
        """Save calibrator to disk.

        Args:
            path: Path to save calibrator
        """
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.method,
                "calibrator": self.calibrator,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load calibrator from disk.

        Args:
            path: Path to load calibrator from
        """
        import joblib

        data = joblib.load(path)
        self.method = data["method"]
        self.calibrator = data["calibrator"]
        self.is_fitted = data["is_fitted"]


def create_calibrator_for_model(model_name: str) -> Optional[ConfidenceCalibrator]:
    """Create or load calibrator for a specific model.

    Args:
        model_name: Name of model

    Returns:
        ConfidenceCalibrator or None if not found
    """
    calibrator_path = (
        PROJECT_ROOT / "experiments" / "classification" / model_name / "calibrator.joblib"
    )

    if calibrator_path.exists():
        calibrator = ConfidenceCalibrator()
        calibrator.load(calibrator_path)
        return calibrator

    return None


def calibrate_model_from_validation(
    model_name: str,
    validation_logits: np.ndarray,
    validation_labels: np.ndarray,
    method: str = "temperature",
) -> ConfidenceCalibrator:
    """Calibrate a model using validation data.

    Args:
        model_name: Name of model
        validation_logits: Validation logits
        validation_labels: Validation labels
        method: Calibration method

    Returns:
        Fitted ConfidenceCalibrator
    """
    calibrator = ConfidenceCalibrator(method=method)
    calibrator.fit(validation_logits, validation_labels)

    # Save calibrator
    calibrator_path = (
        PROJECT_ROOT / "experiments" / "classification" / model_name / "calibrator.joblib"
    )
    calibrator.save(calibrator_path)

    return calibrator
