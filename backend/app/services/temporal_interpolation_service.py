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
from typing import List, Optional, Dict, Tuple, Any
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
        2. Detect large gaps BEFORE interpolation/smoothing
        3. Partition observations into valid allowed segments
        4. Perform interpolation strictly within allowed segments
        5. Apply Savitzky-Golay smoothing post-interpolation if requested
        6. Apply physical range constraints
        7. Generate quality flags and response
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

        # Sort observations by date
        sorted_pairs = sorted(zip(request.observation_dates, request.observation_values), key=lambda x: x[0])
        obs_dates = [p[0] for p in sorted_pairs]
        obs_values = [p[1] for p in sorted_pairs]
        
        # Step 2: Detect large gaps BEFORE interpolation/smoothing
        gap_info = self._detect_large_gaps(obs_dates, request.max_allowed_gap_days)
        gaps = gap_info["gaps"]
        
        if gaps:
            logger.warning(
                f"Detected {len(gaps)} large gaps "
                f"(>{request.max_allowed_gap_days} days): {gaps}"
            )
        
        # Step 3: Segment observations into allowed sub-sequences
        segments = self._segment_observations(obs_dates, obs_values, request.max_allowed_gap_days)

        # Base interpolation method (Savitzky-Golay acts as post-smoothing over cubic/linear)
        base_method = "linear" if request.method == "savgol" else request.method

        # Step 4: Interpolate target dates segment-by-segment (detecting gaps BEFORE interpolation)
        raw_interpolated, quality_flags = self._interpolate_by_segments(
            segments,
            gaps,
            request.target_dates,
            base_method,
            request.max_allowed_gap_days
        )

        # Step 5: Post-interpolation Savitzky-Golay smoothing if requested
        if request.method == "savgol":
            raw_interpolated = self._apply_savgol_post_smoothing(raw_interpolated)

        # Step 6: Apply physical range constraints to non-None interpolated values
        final_values = []
        for i, val in enumerate(raw_interpolated):
            if val is not None:
                clipped = float(np.clip(val, self.min_lai_value, self.max_lai_value))
                final_values.append(clipped)
                # Update quality flag value if valid
                if quality_flags[i].get("status") == "valid":
                    quality_flags[i]["value"] = clipped
            else:
                final_values.append(None)
        
        logger.info(
            f"Interpolation complete: {len(final_values)} values, "
            f"{sum(1 for v in final_values if v is None)} gaps masked"
        )
        
        return InterpolationResponse(
            interpolated_dates=request.target_dates,
            interpolated_values=final_values,
            quality_flags=quality_flags,
            method_used=request.method,
            message=f"Interpolated {len(final_values)} dates. {len(gaps)} large gaps detected."
        )

    def _segment_observations(
        self,
        obs_dates: List[date],
        obs_values: List[float],
        max_gap_days: int
    ) -> List[Dict[str, Any]]:
        """
        Partition observations into contiguous segments where every gap between consecutive
        observations is <= max_gap_days.
        """
        segments = []
        current_dates = [obs_dates[0]]
        current_values = [obs_values[0]]

        for i in range(len(obs_dates) - 1):
            gap_days = (obs_dates[i + 1] - obs_dates[i]).days
            if gap_days <= max_gap_days:
                current_dates.append(obs_dates[i + 1])
                current_values.append(obs_values[i + 1])
            else:
                segments.append({
                    "dates": current_dates,
                    "values": current_values,
                    "start_date": current_dates[0],
                    "end_date": current_dates[-1]
                })
                current_dates = [obs_dates[i + 1]]
                current_values = [obs_values[i + 1]]

        segments.append({
            "dates": current_dates,
            "values": current_values,
            "start_date": current_dates[0],
            "end_date": current_dates[-1]
        })

        return segments

    def _interpolate_by_segments(
        self,
        segments: List[Dict[str, Any]],
        gaps: List[Dict],
        target_dates: List[date],
        method: str,
        max_gap_days: int
    ) -> Tuple[List[Optional[float]], List[Dict]]:
        """
        Interpolate target dates strictly within allowed segments. Target dates in large
        gaps are assigned None with HOLD_OPEN_LOOP.
        """
        interpolated_values: List[Optional[float]] = []
        quality_flags: List[Dict] = []

        for target_date in target_dates:
            # Check if target date falls in a forbidden gap
            if self._is_date_in_gap(target_date, gaps):
                interpolated_values.append(None)
                quality_flags.append({
                    "date": target_date.isoformat(),
                    "action": "HOLD_OPEN_LOOP",
                    "status": "gap_masked",
                    "reason": f"Date falls in cloud gap > {max_gap_days} days"
                })
                continue

            # Find matching segment for this target date
            matching_seg = None
            for seg in segments:
                # If date is within or near segment
                if (seg["start_date"] <= target_date <= seg["end_date"]) or \
                   (target_date < seg["start_date"] and (seg["start_date"] - target_date).days <= max_gap_days and seg == segments[0]) or \
                   (target_date > seg["end_date"] and (target_date - seg["end_date"]).days <= max_gap_days and seg == segments[-1]):
                    matching_seg = seg
                    break

            if matching_seg is None:
                # Target date is out of range / beyond max gap from any segment
                interpolated_values.append(None)
                quality_flags.append({
                    "date": target_date.isoformat(),
                    "action": "HOLD_OPEN_LOOP",
                    "status": "gap_masked",
                    "reason": f"Date exceeds {max_gap_days} days from nearest observation segment"
                })
                continue

            # Perform interpolation within the matched segment
            seg_dates = matching_seg["dates"]
            seg_values = matching_seg["values"]

            if len(seg_dates) == 1:
                val = float(seg_values[0])
            else:
                ref_date = seg_dates[0]
                obs_days = np.array([(d - ref_date).days for d in seg_dates])
                target_day = np.array([(target_date - ref_date).days])
                
                if method == "cubic_spline" and len(seg_dates) >= 3:
                    cs = CubicSpline(obs_days, seg_values, bc_type='natural')
                    val = float(cs(target_day)[0])
                else:
                    # Linear interpolation
                    val = float(np.interp(target_day, obs_days, seg_values)[0])

            interpolated_values.append(val)
            quality_flags.append({
                "date": target_date.isoformat(),
                "type": "interpolated",
                "status": "valid",
                "value": val
            })

        return interpolated_values, quality_flags

    def _apply_savgol_post_smoothing(self, values: List[Optional[float]]) -> List[Optional[float]]:
        """
        Apply Savitzky-Golay filter as a post-interpolation smoothing step over contiguous
        runs of valid (non-None) target values.
        """
        smoothed = list(values)
        n = len(values)
        i = 0

        while i < n:
            if values[i] is None:
                i += 1
                continue

            # Find contiguous run of non-None values
            start_run = i
            while i < n and values[i] is not None:
                i += 1
            end_run = i

            run_vals = np.array(values[start_run:end_run], dtype=float)
            run_len = len(run_vals)

            if run_len >= 5:
                window_length = min(7, run_len if run_len % 2 == 1 else run_len - 1)
                if window_length >= 5:
                    smoothed_run = savgol_filter(run_vals, window_length=window_length, polyorder=3)
                    for idx, val in enumerate(smoothed_run):
                        smoothed[start_run + idx] = float(val)

        return smoothed

    def _detect_large_gaps(
        self,
        dates: List[date],
        max_gap_days: int
    ) -> Dict[str, Any]:
        """
        Monsoon Cloud-Gap Trigger.
        
        Detects gaps between consecutive observations that exceed the threshold.
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

