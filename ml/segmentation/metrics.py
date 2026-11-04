"""Segmentation metrics (IoU, Dice, pixel accuracy, precision, recall)."""
from __future__ import annotations

import numpy as np
import torch


def batched_cm(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Confusion matrix from (N,H,W) int tensors -> (C,C) tensor."""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    cm = torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray) -> dict:
    """Per-class and macro metrics from a (C,C) confusion matrix."""
    cm = cm.astype(np.float64)
    diag = np.diag(cm)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    eps = 1e-7
    iou = diag / (row + col - diag + eps)
    dice = 2 * diag / (row + col + eps)
    prec = diag / (col + eps)
    rec = diag / (row + eps)
    pa = float(diag.sum() / (cm.sum() + eps))
    return {
        "mean_iou": float(iou.mean()),
        "mean_dice": float(dice.mean()),
        "pixel_accuracy": pa,
        "mean_precision": float(prec.mean()),
        "mean_recall": float(rec.mean()),
        "per_class": {
            "iou": iou.tolist(),
            "dice": dice.tolist(),
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "support_rows": row.astype(int).tolist(),
        },
        "confusion_matrix": cm.astype(int).tolist(),
    }
