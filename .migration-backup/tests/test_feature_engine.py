"""
tests/test_feature_engine.py
=============================

Unit tests for AgriTwin FeatureEngine:
- Growth rates (ΔLAI/Δt, ΔTAGP/Δt)
- Cumulative water stress indicators
- Thermal stress indicators
- Assimilation innovation statistics & ensemble spread
- Observation counts, quality, and age
- Strict temporal leakage prevention (data > as_of_date ignored)
"""

import datetime
import pytest
from unittest.mock import MagicMock

from backend.app.features.feature_engine import FeatureEngine
from backend.app.features.schemas import FeatureVector


def test_feature_engine_growth_rates():
    """Verify ΔLAI/Δt and ΔTAGP/Δt calculations over 1d and 7d windows."""
    engine = FeatureEngine()
    start_date = datetime.date(2026, 7, 1)

    daily_outputs = []
    for i in range(10):
        d = start_date + datetime.timedelta(days=i)
        daily_outputs.append({
            "date": d,
            "lai": 1.0 + 0.1 * i,       # LAI grows by 0.1 per day
            "tagp": 100.0 + 50.0 * i,   # TAGP grows by 50.0 per day
            "rftra": 1.0,
            "sm": 0.25,
            "dvs": 0.1 * i,
        })

    as_of = datetime.date(2026, 7, 10)
    fv = engine.compute_features(as_of_date=as_of, daily_outputs=daily_outputs)

    assert fv.as_of_date == as_of
    assert fv.current_lai == pytest.approx(1.9)
    assert fv.current_tagp == pytest.approx(550.0)

    # 1-day LAI rate: (1.9 - 1.8) / 1 = 0.1
    assert fv.growth_rates.delta_lai_1d == pytest.approx(0.1)
    # 1-day TAGP rate: (550 - 500) / 1 = 50.0
    assert fv.growth_rates.delta_tagp_1d == pytest.approx(50.0)

    # 7-day LAI rate: (1.9 - 1.2) / 7 = 0.1
    assert fv.growth_rates.delta_lai_7d == pytest.approx(0.1)
    # 7-day TAGP rate: (550 - 200) / 7 = 50.0
    assert fv.growth_rates.delta_tagp_7d == pytest.approx(50.0)

    assert fv.feature_flat_dict["delta_lai_1d"] == pytest.approx(0.1)
    assert fv.feature_flat_dict["delta_tagp_7d"] == pytest.approx(50.0)


def test_feature_engine_water_stress():
    """Verify cumulative water stress calculation and SM deficit."""
    engine = FeatureEngine()
    start_date = datetime.date(2026, 7, 1)

    daily_outputs = []
    # 5 days with full water (RFTRA = 1.0), 5 days with 0.4 deficit (RFTRA = 0.6)
    for i in range(10):
        d = start_date + datetime.timedelta(days=i)
        rftra = 1.0 if i < 5 else 0.6
        daily_outputs.append({
            "date": d,
            "lai": 2.0,
            "tagp": 1000.0,
            "rftra": rftra,
            "sm": 0.20,
        })

    as_of = datetime.date(2026, 7, 10)
    fv = engine.compute_features(as_of_date=as_of, daily_outputs=daily_outputs)

    # Cumulative deficit = 5 * (1.0 - 0.6) = 2.0
    assert fv.water_stress.cumulative_rftra_deficit == pytest.approx(2.0)
    # Recent 7 days average RFTRA: (1.0 + 1.0 + 0.6*5) / 7 = 5.0 / 7 = 0.714285
    assert fv.water_stress.mean_rftra_7d == pytest.approx(5.0 / 7.0)
    # Soil moisture deficit below 0.35: 0.35 - 0.20 = 0.15
    assert fv.water_stress.current_sm_deficit == pytest.approx(0.15)


def test_feature_engine_thermal_stress():
    """Verify heat and cold stress indicator calculation from weather records."""
    engine = FeatureEngine(heat_threshold_c=35.0, cold_threshold_c=5.0)
    start_date = datetime.date(2026, 7, 1)

    daily_outputs = [{"date": start_date + datetime.timedelta(days=i), "lai": 1.5} for i in range(10)]
    weather_records = []
    for i in range(10):
        d = start_date + datetime.timedelta(days=i)
        # Heat stress on days 2 and 4 (Tmax=37), Cold stress on day 6 (Tmin=3)
        tmax = 37.0 if i in (2, 4) else 30.0
        tmin = 3.0 if i == 6 else 18.0
        weather_records.append({
            "date": d,
            "tmax": tmax,
            "tmin": tmin,
        })

    as_of = datetime.date(2026, 7, 10)
    fv = engine.compute_features(
        as_of_date=as_of,
        daily_outputs=daily_outputs,
        weather_records=weather_records,
    )

    assert fv.thermal_stress.cumulative_heat_days == 2
    assert fv.thermal_stress.cumulative_cold_days == 1
    assert fv.thermal_stress.max_tmax_7d == pytest.approx(37.0)


def test_feature_engine_assimilation_diagnostics():
    """Verify EnKF innovation statistics and ensemble spread extraction."""
    engine = FeatureEngine()
    as_of = datetime.date(2026, 7, 10)

    assimilation_states = [
        {
            "cycle_date": datetime.date(2026, 7, 3),
            "innovation": {"lai": 0.2, "sm": -0.02},
            "ensemble_spread_prior": {"lai": 0.5, "sm": 0.05},
            "posterior_spread": {"lai": 0.3, "sm": 0.03},
            "state_update_magnitude": {"lai": 0.15, "sm": 0.015},
        },
        {
            "cycle_date": datetime.date(2026, 7, 7),
            "innovation": {"lai": 0.4, "sm": -0.04},
            "ensemble_spread_prior": {"lai": 0.4, "sm": 0.04},
            "posterior_spread": {"lai": 0.2, "sm": 0.02},
            "state_update_magnitude": {"lai": 0.25, "sm": 0.025},
        },
    ]

    fv = engine.compute_features(
        as_of_date=as_of,
        daily_outputs=[{"date": as_of, "lai": 2.5}],
        assimilation_states=assimilation_states,
    )

    assert fv.assimilation_diagnostics.assimilation_cycles_count == 2
    # Mean innovation: lai = (0.2 + 0.4)/2 = 0.3, sm = (-0.02 - 0.04)/2 = -0.03
    assert fv.assimilation_diagnostics.mean_innovation["lai"] == pytest.approx(0.3)
    assert fv.assimilation_diagnostics.mean_innovation["sm"] == pytest.approx(-0.03)

    # Latest cycle (July 7)
    assert fv.assimilation_diagnostics.latest_innovation["lai"] == pytest.approx(0.4)
    assert fv.assimilation_diagnostics.prior_spread["lai"] == pytest.approx(0.4)
    assert fv.assimilation_diagnostics.posterior_spread["lai"] == pytest.approx(0.2)

    assert fv.feature_flat_dict["mean_innov_lai"] == pytest.approx(0.3)
    assert fv.feature_flat_dict["prior_spread_lai"] == pytest.approx(0.4)


def test_feature_engine_observation_quality_and_age():
    """Verify observation counts, quality scores, sources, and age calculations."""
    engine = FeatureEngine()
    as_of = datetime.date(2026, 7, 10)

    observations = [
        {"obs_date": datetime.date(2026, 7, 2), "status": "VALID", "source": "SATELLITE", "quality_score": 0.90},
        {"obs_date": datetime.date(2026, 7, 5), "status": "REJECTED", "source": "IOT_SENSOR", "quality_score": 0.20},
        {"obs_date": datetime.date(2026, 7, 7), "status": "VALID", "source": "SMARTPHONE_GRVI", "quality_score": 0.80},
    ]

    fv = engine.compute_features(
        as_of_date=as_of,
        daily_outputs=[{"date": as_of, "lai": 2.0}],
        observations=observations,
    )

    assert fv.observation_quality.total_obs_count == 3
    assert fv.observation_quality.valid_obs_count == 2
    assert fv.observation_quality.rejected_obs_count == 1
    # Mean valid quality score: (0.90 + 0.80) / 2 = 0.85
    assert fv.observation_quality.mean_quality_score == pytest.approx(0.85)
    # Latest valid obs date = July 7. as_of = July 10. Age = 3 days.
    assert fv.observation_quality.latest_obs_age_days == pytest.approx(3.0)
    assert "SATELLITE" in fv.observation_quality.obs_sources_present
    assert "SMARTPHONE_GRVI" in fv.observation_quality.obs_sources_present


def test_temporal_leakage_safety():
    """Verify strict temporal leakage prevention — records after as_of_date MUST be ignored."""
    engine = FeatureEngine()
    as_of = datetime.date(2026, 7, 5)

    # Daily outputs extending to July 10 (future data after July 5 cut-off)
    daily_outputs = [
        {"date": datetime.date(2026, 7, 1), "lai": 1.0, "tagp": 100.0, "rftra": 1.0},
        {"date": datetime.date(2026, 7, 5), "lai": 1.5, "tagp": 200.0, "rftra": 1.0},
        # Future records (SHOULD BE FILTERED OUT)
        {"date": datetime.date(2026, 7, 8), "lai": 4.0, "tagp": 1000.0, "rftra": 0.1},
        {"date": datetime.date(2026, 7, 10), "lai": 6.0, "tagp": 2000.0, "rftra": 0.0},
    ]

    # Future observations (SHOULD BE FILTERED OUT)
    observations = [
        {"obs_date": datetime.date(2026, 7, 3), "status": "VALID", "source": "SATELLITE", "quality_score": 0.95},
        {"obs_date": datetime.date(2026, 7, 9), "status": "VALID", "source": "SATELLITE", "quality_score": 0.99},
    ]

    fv = engine.compute_features(
        as_of_date=as_of,
        daily_outputs=daily_outputs,
        observations=observations,
    )

    # Current LAI must be July 5 value (1.5), not July 10 value (6.0)
    assert fv.current_lai == pytest.approx(1.5)
    assert fv.current_tagp == pytest.approx(200.0)

    # Water stress deficit must NOT include July 8/10 stress
    assert fv.water_stress.cumulative_rftra_deficit == pytest.approx(0.0)

    # Observation count must be 1 (July 3), ignoring July 9
    assert fv.observation_quality.total_obs_count == 1
    # Age relative to July 5: July 5 - July 3 = 2 days
    assert fv.observation_quality.latest_obs_age_days == pytest.approx(2.0)
