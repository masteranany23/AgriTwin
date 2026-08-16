"""
tests/test_enkf_diagnostics.py
===============================

Unit tests for compact EnKF diagnostics extraction from ensemble matrices and DB records.
"""

import datetime
import uuid
import numpy as np
import pytest

from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.benchmarking.enkf_diagnostics import EnKFDiagnosticsExtractor


def test_extract_from_matrices():
    """Verify matrix-level diagnostic extraction for innovation, spread, counts, and update magnitude."""
    cycle_date = datetime.date(2025, 6, 1)

    # 2 variables (LAI, SM), N=3 ensemble members
    X_f = np.array([
        [1.0, 1.2, 0.8],  # LAI mean = 1.0, std = 0.2
        [0.2, 0.22, 0.18], # SM mean = 0.2, std = 0.02
    ])
    X_a = np.array([
        [1.5, 1.7, 1.3],  # LAI mean = 1.5, std = 0.2
        [0.25, 0.27, 0.23], # SM mean = 0.25, std = 0.02
    ])
    y = np.array([1.6, np.nan])  # Observation for LAI only
    d = np.array([0.6, np.nan])  # Innovation for LAI only

    diag = EnKFDiagnosticsExtractor.extract_from_matrices(
        cycle_date=cycle_date,
        X_f=X_f,
        X_a=X_a,
        y=y,
        d=d,
        raw_obs_count=3,
        qc_obs_count=1,
        state_vars=["LAI", "SM"],
    )

    assert diag.cycle_date == cycle_date
    assert diag.valid_obs_count == 1
    assert diag.rejected_obs_count == 2
    assert diag.innovation["lai"] == pytest.approx(0.6)
    assert diag.innovation["sm"] is None

    # State update magnitude: |1.5 - 1.0| = 0.5 for LAI, |0.25 - 0.2| = 0.05 for SM
    assert diag.state_update_magnitude["lai"] == pytest.approx(0.5)
    assert diag.state_update_magnitude["sm"] == pytest.approx(0.05)

    # Spreads
    assert diag.ensemble_spread_prior["lai"] == pytest.approx(0.2)
    assert diag.posterior_spread["lai"] == pytest.approx(0.2)


def test_extract_from_db_state_and_summarize():
    """Verify DB state extraction and run summary aggregation."""
    sim_id = uuid.uuid4()
    run_id = uuid.uuid4()
    d1 = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)

    state = AssimilationState(
        id=uuid.uuid4(),
        simulation_run_id=sim_id,
        assimilation_run_id=run_id,
        assimilation_time=d1,
        ensemble_mean={"lai": 1.5, "sm": 0.25},
        ensemble_covariance={"lai": 0.04, "sm": 0.0004},
        observation_vector={"lai": 1.6},
        innovation_vector={"lai": 0.6},
        kalman_gain={},
        updated_state_vector={"lai": 1.5, "sm": 0.25},
        forecast_state_vector={"lai": 1.0, "sm": 0.20},
        number_of_members=3,
        observation_count=1,
    )

    cycle_diag = EnKFDiagnosticsExtractor.extract_from_db_state(state, raw_obs_count=2)
    assert cycle_diag.valid_obs_count == 1
    assert cycle_diag.rejected_obs_count == 1
    assert cycle_diag.state_update_magnitude["lai"] == pytest.approx(0.5)

    summary = EnKFDiagnosticsExtractor.summarize_run(
        simulation_id=sim_id,
        assimilation_run_id=run_id,
        cycles=[cycle_diag],
    )

    assert summary.total_cycles == 1
    assert summary.total_valid_obs == 1
    assert summary.total_rejected_obs == 1
    assert summary.avg_state_update_magnitude["lai"] == pytest.approx(0.5)
