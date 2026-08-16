"""
backend/app/benchmarking/metrics.py
====================================

Pure statistical error metric computations for scientific model validation.

Supports:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Bias (Mean Error: predicted - ground truth)
- R² (Coefficient of Determination)
- Ensemble Interval Coverage (fraction of ground truth inside ensemble bounds)
"""

from typing import Optional, Tuple
import numpy as np

from backend.app.benchmarking.schemas import MetricSet


def compute_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> Optional[float]:
    """Compute Mean Absolute Error (MAE)."""
    if len(predicted) == 0 or len(ground_truth) == 0 or len(predicted) != len(ground_truth):
        return None
    return float(np.mean(np.abs(predicted - ground_truth)))


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> Optional[float]:
    """Compute Root Mean Squared Error (RMSE)."""
    if len(predicted) == 0 or len(ground_truth) == 0 or len(predicted) != len(ground_truth):
        return None
    return float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))


def compute_bias(predicted: np.ndarray, ground_truth: np.ndarray) -> Optional[float]:
    """Compute Bias / Mean Error (predicted - ground_truth)."""
    if len(predicted) == 0 or len(ground_truth) == 0 or len(predicted) != len(ground_truth):
        return None
    return float(np.mean(predicted - ground_truth))


def compute_r2(predicted: np.ndarray, ground_truth: np.ndarray) -> Optional[float]:
    """Compute Coefficient of Determination (R²).

    Returns None if fewer than 2 points exist or if ground truth has zero variance.
    """
    if len(predicted) < 2 or len(ground_truth) < 2 or len(predicted) != len(ground_truth):
        return None

    ss_res = np.sum((ground_truth - predicted) ** 2)
    gt_mean = np.mean(ground_truth)
    ss_tot = np.sum((ground_truth - gt_mean) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else None

    r2_val = 1.0 - (ss_res / ss_tot)
    return float(r2_val)


def compute_ensemble_coverage(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    ground_truth: np.ndarray,
) -> Optional[float]:
    """Compute empirical coverage rate (fraction of ground truth values within ensemble bounds)."""
    if (
        len(lower_bounds) == 0
        or len(upper_bounds) == 0
        or len(ground_truth) == 0
        or not (len(lower_bounds) == len(upper_bounds) == len(ground_truth))
    ):
        return None

    inside = (ground_truth >= lower_bounds) & (ground_truth <= upper_bounds)
    return float(np.mean(inside))


def compute_all_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    ensemble_lower: Optional[np.ndarray] = None,
    ensemble_upper: Optional[np.ndarray] = None,
) -> MetricSet:
    """Compute complete metric set for a series of aligned prediction vs ground truth values."""
    if len(predicted) == 0 or len(ground_truth) == 0 or len(predicted) != len(ground_truth):
        return MetricSet(
            mae=None,
            rmse=None,
            bias=None,
            r2=None,
            ensemble_interval_coverage=None,
            sample_size=0,
        )

    mae = compute_mae(predicted, ground_truth)
    rmse = compute_rmse(predicted, ground_truth)
    bias = compute_bias(predicted, ground_truth)
    r2 = compute_r2(predicted, ground_truth)

    coverage = None
    if ensemble_lower is not None and ensemble_upper is not None:
        coverage = compute_ensemble_coverage(ensemble_lower, ensemble_upper, ground_truth)

    return MetricSet(
        mae=mae,
        rmse=rmse,
        bias=bias,
        r2=r2,
        ensemble_interval_coverage=coverage,
        sample_size=len(predicted),
    )
