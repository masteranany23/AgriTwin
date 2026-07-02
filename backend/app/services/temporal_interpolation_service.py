"""
Temporal Interpolation Service
================================

Fills gaps between sparse satellite observations using multiple interpolation methods.

Monsoon Cloud-Gap Trigger
- Detects gaps > threshold (default 10 days)
- Returns None for large gaps instead of interpolating
- Signals EnKF to hold open-loop during monsoon periods

"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from datetime import date, timedelta
from typing import List, Optional, Dict, Tuple
import logging

from backend.app.api.schemas.interpolation import (
    InterpolationRequest,
    InterpolationResponse
)

logger = logging.getLogger(__name__)


class TemporalInterpolationService:
    """
    Temporal interpolation service for satellite observation gap-filling.
    
    Supports three methods:
    1. Linear: Fast, simple interpolation
    2. Cubic Spline: Smooth curves (recommended for LAI)
    3. Savitzky-Golay: Noise-reducing filter
    
    Implements monsoon cloud-gap detection to avoid unrealistic interpolation
    across extended cloud cover periods.
    """
    
    def __init__(self):
        """Initialize the interpolation service."""
        self.min_observations = 2
        self.max_lai_value = 8.0
        self.min_lai_value = 0.0
    
    def interpolate(self, request: InterpolationRequest) -> InterpolationResponse:
        """
        Fill temporal gaps in satellite observations.
        
        Process:
        1. Validate input observations
        2. Detect large gaps (monsoon cloud cover)
        3. Perform interpolation using selected method
        4. Apply gap masking (set None for large gaps)
        5. Generate quality flags for each date
        
        Args:
            request: InterpolationRequest containing observation dates/values
                     and target dates to interpolate
        
        Returns:
            InterpolationResponse with interpolated values and quality flags
        """
        logger.info(
            f"Starting interpolation: {len(request.observation_dates)} observations, "
            f"{len(request.target_dates)} target dates, method={request.method}"
        )
        
        # Step 1: Validate observations
        if len(request.observation_dates) < self.min_observations:
            logger.warning(f"Insufficient observations: {len(request.observation_dates)} < {self.min_observations}")
            return self._create_empty_response(
                request.target_dates,
                f"Need at least {self.min_observations} observations for interpolation"
            )
        
        if len(request.observation_dates) != len(request.observation_values):
            logger.error("Mismatch between observation dates and values")
            return self._create_empty_response(
                request.target_dates,
                "Observation dates and values must have same length"
            )
        
        # Step 2: Detect large gaps
        gap_info = self._detect_large_gaps(
            request.observation_dates,
            request.max_allowed_gap_days
        )
        
        if gap_info["gaps"]:
            logger.warning(
                f"Detected {len(gap_info['gaps'])} large gaps "
                f"(>{request.max_allowed_gap_days} days): {gap_info['gaps']}"
            )
        
        # Step 3: Convert dates to numeric days for interpolation
        date_arrays = self._prepare_date_arrays(
            request.observation_dates,
            request.target_dates
        )
        
        # Step 4: Perform interpolation
        try:
            interpolated_values = self._perform_interpolation(
                date_arrays["obs_days"],
                request.observation_values,
                date_arrays["target_days"],
                request.method
            )
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
            return self._create_empty_response(
                request.target_dates,
                f"Interpolation error: {str(e)}"
            )
        
        # Step 5: Apply constraints (clip to realistic range)
        interpolated_values = self._apply_constraints(interpolated_values)
        
        # Step 6: Apply gap masking (RESEARCH STEP 3 - Monsoon trigger)
        masked_values, quality_flags = self._apply_gap_masking(
            interpolated_values,
            request.target_dates,
            gap_info["gaps"],
            request.max_allowed_gap_days
        )
        
        logger.info(
            f"Interpolation complete: {len(masked_values)} values, "
            f"{sum(1 for v in masked_values if v is None)} gaps masked"
        )
        
        return InterpolationResponse(
            interpolated_dates=request.target_dates,
            interpolated_values=masked_values,
            quality_flags=quality_flags,
            method_used=request.method,
            message=f"Interpolated {len(masked_values)} dates. {len(gap_info['gaps'])} large gaps detected."
        )
    
    def _detect_large_gaps(
        self,
        dates: List[date],
        max_gap_days: int
    ) -> Dict[str, any]:
        """
        Monsoon Cloud-Gap Trigger.
        
        Detects gaps between consecutive observations that exceed the threshold.
        These gaps typically occur during monsoon season due to persistent
        cloud cover blocking satellite observations.
        
        Args:
            dates: List of observation dates (must be sorted)
            max_gap_days: Maximum allowed gap before triggering open-loop
        
        Returns:
            Dictionary with gap information:
            - gaps: List of gap periods
            - total_gap_days: Total days in gaps
        """
        sorted_dates = sorted(dates)
        gaps = []
        total_gap_days = 0
        
        for i in range(len(sorted_dates) - 1):
            gap_days = (sorted_dates[i + 1] - sorted_dates[i]).days
            
            if gap_days > max_gap_days:
                gap_info = {
                    "start_date": sorted_dates[i],
                    "end_date": sorted_dates[i + 1],
                    "gap_days": gap_days,
                    "action": "HOLD_OPEN_LOOP",
                    "reason": f"Cloud gap exceeds {max_gap_days} days (monsoon period)"
                }
                gaps.append(gap_info)
                total_gap_days += gap_days
        
        return {
            "gaps": gaps,
            "total_gap_days": total_gap_days,
            "max_single_gap": max(gaps, key=lambda g: g["gap_days"])["gap_days"] if gaps else 0
        }
    
    def _prepare_date_arrays(
        self,
        obs_dates: List[date],
        target_dates: List[date]
    ) -> Dict[str, np.ndarray]:
        """
        Convert dates to numeric arrays for interpolation.
        
        Uses days since first observation as the numeric representation.
        """
        reference_date = min(obs_dates)
        
        obs_days = np.array([
            (d - reference_date).days for d in obs_dates
        ])
        
        target_days = np.array([
            (d - reference_date).days for d in target_dates
        ])
        
        return {
            "obs_days": obs_days,
            "target_days": target_days,
            "reference_date": reference_date
        }
    
    def _perform_interpolation(
        self,
        obs_days: np.ndarray,
        obs_values: List[float],
        target_days: np.ndarray,
        method: str
    ) -> np.ndarray:
        """
        Perform interpolation using the specified method.
        
        Methods:
        - linear: Simple linear interpolation (fastest)
        - cubic_spline: Smooth cubic spline (recommended for LAI)
        - savgol: Savitzky-Golay filter (noise reduction)
        """
        obs_values_array = np.array(obs_values)
        
        if method == "linear":
            return self._linear_interpolation(obs_days, obs_values_array, target_days)
        
        elif method == "cubic_spline":
            return self._cubic_spline_interpolation(obs_days, obs_values_array, target_days)
        
        elif method == "savgol":
            return self._savgol_interpolation(obs_days, obs_values_array, target_days)
        
        else:
            logger.warning(f"Unknown method '{method}', falling back to linear")
            return self._linear_interpolation(obs_days, obs_values_array, target_days)
    
    def _linear_interpolation(
        self,
        obs_days: np.ndarray,
        obs_values: np.ndarray,
        target_days: np.ndarray
    ) -> np.ndarray:
        """Simple linear interpolation between observations."""
        return np.interp(target_days, obs_days, obs_values)
    
    def _cubic_spline_interpolation(
        self,
        obs_days: np.ndarray,
        obs_values: np.ndarray,
        target_days: np.ndarray
    ) -> np.ndarray:
        """
        Cubic spline interpolation with natural boundary conditions.
        
        Produces smooth curves suitable for biological variables like LAI.
        """
        cs = CubicSpline(obs_days, obs_values, bc_type='natural')
        return cs(target_days)
    
    def _savgol_interpolation(
        self,
        obs_days: np.ndarray,
        obs_values: np.ndarray,
        target_days: np.ndarray
    ) -> np.ndarray:
        """
        Savitzky-Golay filter for noise reduction.
        
        First performs linear interpolation, then applies smoothing filter.
        """
        # First do linear interpolation
        linear_values = np.interp(target_days, obs_days, obs_values)
        
        # Determine appropriate window size (must be odd)
        n_points = len(linear_values)
        window_length = min(7, n_points if n_points % 2 == 1 else n_points - 1)
        
        if window_length < 5:
            window_length = 5
        
        # Ensure window length is valid
        if window_length > n_points:
            logger.warning("Too few points for Savitzky-Golay, using linear interpolation")
            return linear_values
        
        # Apply Savitzky-Golay filter
        return savgol_filter(linear_values, window_length=window_length, polyorder=3)
    
    def _apply_constraints(self, values: np.ndarray) -> np.ndarray:
        """
        Apply physical constraints to interpolated values.
        
        For LAI: clip to realistic range [0, 8]
        For SM: clip to [0, 1]
        """
        return np.clip(values, self.min_lai_value, self.max_lai_value)
    
    def _apply_gap_masking(
        self,
        values: np.ndarray,
        target_dates: List[date],
        gaps: List[Dict],
        max_gap_days: int
    ) -> Tuple[List[Optional[float]], List[Dict]]:
        """
        Apply monsoon gap masking.
        
        Sets values to None for dates falling within large gaps.
        This signals downstream processes (EnKF) to hold open-loop
        instead of assimilating unrealistic interpolated data.
        """
        masked_values = values.tolist()
        quality_flags = []
        
        for i, target_date in enumerate(target_dates):
            in_gap = self._is_date_in_gap(target_date, gaps)
            
            if in_gap:
                # Mask value as None - EnKF will skip assimilation
                masked_values[i] = None
                quality_flags.append({
                    "date": target_date.isoformat(),
                    "action": "HOLD_OPEN_LOOP",
                    "status": "gap_masked",
                    "reason": f"Date falls in cloud gap > {max_gap_days} days"
                })
            else:
                # Keep interpolated value
                quality_flags.append({
                    "date": target_date.isoformat(),
                    "type": "interpolated",
                    "status": "valid",
                    "value": float(masked_values[i])
                })
        
        return masked_values, quality_flags
    
    def _is_date_in_gap(self, check_date: date, gaps: List[Dict]) -> bool:
        """Check if a date falls within any detected gap period."""
        for gap in gaps:
            if gap["start_date"] < check_date < gap["end_date"]:
                return True
        return False
    
    def _create_empty_response(
        self,
        target_dates: List[date],
        message: str
    ) -> InterpolationResponse:
        """Create empty response for error cases."""
        return InterpolationResponse(
            interpolated_dates=target_dates,
            interpolated_values=[None] * len(target_dates),
            quality_flags=[
                {"date": d.isoformat(), "status": "error", "reason": message}
                for d in target_dates
            ],
            method_used="none",
            message=message
        )
