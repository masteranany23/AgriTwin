"""
Evaluation metrics for model performance.
"""
import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error
)


logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.
    
    Args:
        y_true: Ground truth values (N,).
        y_pred: Predicted values (N,).
        
    Returns:
        Dictionary with computed metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    metrics = {}
    
    try:
        # RMSE
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # R² score
        metrics["r2"] = float(r2_score(y_true, y_pred))
        
        # MAE
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        
        # MAPE (%)
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = np.nan
        
        # Relative RMSE (%)
        mean_true = y_true.mean()
        if mean_true != 0:
            metrics["rmse_percent"] = float(metrics["rmse"] / mean_true * 100)
        else:
            metrics["rmse_percent"] = np.nan
        
        # Bias
        metrics["bias"] = float((y_pred - y_true).mean())
        
        # Log metrics
        logger.info("Metrics computed:")
        logger.info(f"  RMSE: {metrics['rmse']:.2f} ({metrics['rmse_percent']:.1f}%)")
        logger.info(f"  R²: {metrics['r2']:.3f}")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.1f}%")
        logger.info(f"  Bias: {metrics['bias']:.2f}")
        
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise
    
    return metrics
