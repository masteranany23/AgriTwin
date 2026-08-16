"""
tests/test_temporal_interpolation.py
=====================================

Regression tests for TemporalInterpolationService verifying:
- Small gaps (<= max_allowed_gap_days) are fully interpolated.
- Threshold gaps (= max_allowed_gap_days) are fully interpolated.
- Large gaps (> max_allowed_gap_days) are detected BEFORE interpolation/smoothing.
- Savitzky-Golay is treated as post-interpolation smoothing without bleeding across gaps.
"""

from datetime import date, timedelta
import pytest

from backend.app.api.schemas.interpolation import InterpolationRequest
from backend.app.services.temporal_interpolation_service import TemporalInterpolationService


@pytest.fixture
def service():
    return TemporalInterpolationService()


def test_small_gap_interpolation(service):
    """Verify small gaps (e.g. 5 days <= 10 max_gap) are fully interpolated."""
    obs_dates = [date(2024, 7, 1), date(2024, 7, 6)]
    obs_values = [2.0, 3.0]
    target_dates = [date(2024, 7, 1) + timedelta(days=i) for i in range(6)]

    req = InterpolationRequest(
        observation_dates=obs_dates,
        observation_values=obs_values,
        target_dates=target_dates,
        method="linear",
        max_allowed_gap_days=10
    )

    resp = service.interpolate(req)

    assert len(resp.interpolated_values) == 6
    assert None not in resp.interpolated_values
    # Linear midpoint at July 3.5 or July 3/4
    assert resp.interpolated_values[2] == pytest.approx(2.4, abs=0.01)
    assert all(q["status"] == "valid" for q in resp.quality_flags)


def test_threshold_gap_interpolation(service):
    """Verify gap equal to max_allowed_gap_days (10 days) is allowed and interpolated."""
    obs_dates = [date(2024, 7, 1), date(2024, 7, 11)]  # gap = 10 days
    obs_values = [1.0, 3.0]
    target_dates = [date(2024, 7, 1) + timedelta(days=i) for i in range(11)]

    req = InterpolationRequest(
        observation_dates=obs_dates,
        observation_values=obs_values,
        target_dates=target_dates,
        method="linear",
        max_allowed_gap_days=10
    )

    resp = service.interpolate(req)

    assert None not in resp.interpolated_values
    assert resp.interpolated_values[5] == pytest.approx(2.0, abs=0.01)


def test_large_gap_detected_before_interpolation(service):
    """
    Verify large gap (> 10 days) is detected BEFORE interpolation.
    Observations across the gap must not contaminate each other's segment interpolation.
    """
    # Segment 1: July 1 (1.0), July 5 (2.0) -> slope = 0.25/day
    # Gap: 20 days between July 5 and July 25 (> 10 days threshold)
    # Segment 2: July 25 (5.0), July 29 (6.0) -> slope = 0.25/day
    obs_dates = [date(2024, 7, 1), date(2024, 7, 5), date(2024, 7, 25), date(2024, 7, 29)]
    obs_values = [1.0, 2.0, 5.0, 6.0]

    target_dates = [date(2024, 7, 1) + timedelta(days=i) for i in range(29)]

    req = InterpolationRequest(
        observation_dates=obs_dates,
        observation_values=obs_values,
        target_dates=target_dates,
        method="linear",
        max_allowed_gap_days=10
    )

    resp = service.interpolate(req)

    # July 1 (idx 0): 1.0, July 3 (idx 2): 1.5, July 5 (idx 4): 2.0
    assert resp.interpolated_values[0] == pytest.approx(1.0)
    assert resp.interpolated_values[2] == pytest.approx(1.5)
    assert resp.interpolated_values[4] == pytest.approx(2.0)

    # Large gap targets (July 6..24, idx 5..23) must be None with HOLD_OPEN_LOOP
    for idx in range(5, 24):
        assert resp.interpolated_values[idx] is None
        assert resp.quality_flags[idx]["action"] == "HOLD_OPEN_LOOP"

    # Segment 2 targets (July 25..29, idx 24..28)
    assert resp.interpolated_values[24] == pytest.approx(5.0)
    assert resp.interpolated_values[26] == pytest.approx(5.5)
    assert resp.interpolated_values[28] == pytest.approx(6.0)


def test_savgol_post_smoothing_does_not_bleed_across_large_gap(service):
    """
    Verify Savitzky-Golay method acts as post-interpolation smoothing
    without bleeding smoothed values across large gap boundaries.
    """
    obs_dates = [
        date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 4), date(2024, 7, 5),
        # 20-day large gap
        date(2024, 7, 25), date(2024, 7, 26), date(2024, 7, 27), date(2024, 7, 28), date(2024, 7, 29)
    ]
    obs_values = [1.0, 1.2, 1.4, 1.6, 1.8, 4.0, 4.2, 4.4, 4.6, 4.8]
    target_dates = [date(2024, 7, 1) + timedelta(days=i) for i in range(29)]

    req = InterpolationRequest(
        observation_dates=obs_dates,
        observation_values=obs_values,
        target_dates=target_dates,
        method="savgol",
        max_allowed_gap_days=10
    )

    resp = service.interpolate(req)

    # Dates in large gap must remain None
    for idx in range(5, 24):
        assert resp.interpolated_values[idx] is None
        assert resp.quality_flags[idx]["action"] == "HOLD_OPEN_LOOP"

    # Valid segment values exist and are post-smoothed
    for idx in range(0, 5):
        assert resp.interpolated_values[idx] is not None
    for idx in range(24, 29):
        assert resp.interpolated_values[idx] is not None
