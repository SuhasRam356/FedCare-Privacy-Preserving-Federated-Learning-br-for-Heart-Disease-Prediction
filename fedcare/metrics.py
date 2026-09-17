"""
fedcare.metrics - Comprehensive clinical and federated performance metrics.

Provides standard classification metrics and federated fairness accounting:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Per-Hospital Performance Breakdown
- Fairness Accounting:
    * Worst-Hospital Accuracy & AUC (min-max fairness)
    * Inter-Hospital Variance / Standard Deviation
    * Performance Equity Gap (Best client vs Worst client)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("fedcare.metrics")


def compute_classification_metrics(
    y_true: np.ndarray | Sequence[int],
    y_pred_probs: np.ndarray | Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute standard clinical classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_pred_probs : np.ndarray
        Predicted probabilities for the positive class (disease = 1).
    threshold : float, optional
        Decision threshold for binary classification (default: 0.5).

    Returns
    -------
    dict[str, float]
        Dictionary containing accuracy, precision, recall, f1, and auc.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_probs = np.asarray(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC handles potential edge case of homogeneous labels in evaluation split
    try:
        auc = float(roc_auc_score(y_true, y_pred_probs))
    except ValueError:
        auc = 0.5  # Fallback if only single class is represented

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
    }


def compute_fairness_metrics(
    client_metrics: dict[str | int, dict[str, float]],
    metric_key: str = "auc",
) -> dict[str, float]:
    """
    Compute federated fairness metrics across participating hospitals.

    Fairness in healthcare FL ensures the global model does not sacrifice
    outlying hospitals (e.g. rural clinics with extreme class imbalance).

    Parameters
    ----------
    client_metrics : dict[str | int, dict[str, float]]
        Mapping of hospital_id -> metrics dict (e.g. {1: {"accuracy": 0.84, "auc": 0.86}, ...}).
    metric_key : str, optional
        Target metric to evaluate fairness on (default: 'auc').

    Returns
    -------
    dict[str, float]
        Dictionary with mean, std_dev, worst (min), best (max), and equity gap.
    """
    values = [
        m[metric_key]
        for m in client_metrics.values()
        if metric_key in m
    ]

    if not values:
        return {
            f"{metric_key}_mean": 0.0,
            f"{metric_key}_std": 0.0,
            f"{metric_key}_worst": 0.0,
            f"{metric_key}_best": 0.0,
            f"{metric_key}_equity_gap": 0.0,
        }

    arr = np.array(values, dtype=float)
    worst_val = float(np.min(arr))
    best_val = float(np.max(arr))

    return {
        f"{metric_key}_mean": round(float(np.mean(arr)), 4),
        f"{metric_key}_std": round(float(np.std(arr)), 4),
        f"{metric_key}_worst": round(worst_val, 4),
        f"{metric_key}_best": round(best_val, 4),
        f"{metric_key}_equity_gap": round(best_val - worst_val, 4),
    }
