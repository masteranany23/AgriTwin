"""
Population Stability Index (PSI) drift detection.
"""
import logging
from typing import Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def calculate_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: List[str],
    bins: int = 10
) -> Dict[str, float]:
    """
    Calculate Population Stability Index for drift detection.
    
    PSI measures distribution shift between reference and current data.
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change (model retraining recommended)
    
    Args:
        reference: Reference dataset (training data).
        current: Current dataset (production data).
        feature_cols: Features to monitor.
        bins: Number of bins for discretization.
        
    Returns:
        Dictionary mapping feature name to PSI score.
    """
    psi_scores = {}
    
    for col in feature_cols:
        if col not in reference.columns or col not in current.columns:
            logger.warning(f"Feature {col} not found in data, skipping")
            continue
        
        ref_vals = reference[col].dropna().values
        curr_vals = current[col].dropna().values
        
        if len(ref_vals) == 0 or len(curr_vals) == 0:
            logger.warning(f"Empty values for {col}, skipping")
            continue
        
        try:
            # Create bins based on reference quantiles
            _, bin_edges = np.histogram(ref_vals, bins=bins)
            
            # Ensure bins cover current data range
            bin_edges[0] = min(bin_edges[0], curr_vals.min())
            bin_edges[-1] = max(bin_edges[-1], curr_vals.max())
            
            # Calculate distributions
            ref_counts, _ = np.histogram(ref_vals, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_vals, bins=bin_edges)
            
            # Convert to proportions
            ref_prop = ref_counts / len(ref_vals)
            curr_prop = curr_counts / len(curr_vals)
            
            # Avoid division by zero
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            psi_scores[col] = float(psi)
            
        except Exception as e:
            logger.error(f"Error calculating PSI for {col}: {e}")
            psi_scores[col] = np.nan
    
    return psi_scores


def check_drift(
    psi_scores: Dict[str, float],
    threshold: float = 0.15
) -> Dict[str, any]:
    """
    Check for drift based on PSI scores.
    
    Args:
        psi_scores: PSI scores per feature.
        threshold: PSI threshold for drift alert.
        
    Returns:
        Dictionary with drift analysis.
    """
    drifted_features = []
    stable_features = []
    
    for feature, psi in psi_scores.items():
        if np.isnan(psi):
            continue
        
        if psi > threshold:
            drifted_features.append({
                "feature": feature,
                "psi": psi,
                "severity": "high" if psi > 0.25 else "moderate"
            })
        else:
            stable_features.append(feature)
    
    drift_detected = len(drifted_features) > 0
    
    result = {
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "stable_features": stable_features,
        "max_psi": max(psi_scores.values()) if psi_scores else 0.0,
        "threshold": threshold
    }
    
    if drift_detected:
        logger.warning(f"Drift detected in {len(drifted_features)} features")
        for item in drifted_features:
            logger.warning(f"  {item['feature']}: PSI={item['psi']:.3f} ({item['severity']})")
    else:
        logger.info("No significant drift detected")
    
    return result
