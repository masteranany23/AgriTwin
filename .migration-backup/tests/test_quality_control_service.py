"""tests/test_quality_control_service.py — Unit Tests for QualityControlService
=============================================================================

Validates the centralized QualityControlService logic:
- Explicit ObservationStatus evaluation (VALID, OUTLIER, MISSING, REJECTED)
- Physical bounds validation (LAI, SM, state vector variables)
- Satellite cloud cover masking
- Quality score threshold gating
- Statistical Z-score outlier detection vs forecast ensemble
- Batch observation filtering
"""

import math
import uuid
import numpy as np
import pytest

from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.services.quality_control_service import QualityControlService, QCConfig, QCResult


@pytest.fixture
def qc_service():
    return QualityControlService()


def create_mock_observation(
    value=3.0,
    variable_name="LAI",
    source=ObservationSource.SATELLITE,
    quality_score=80,
    cloud_cover=0.05,
    status=ObservationStatus.VALID,
):
    """Helper to instantiate mock Observation dataclass/ORM object."""
    obs = Observation(
        id=uuid.uuid4(),
        field_id=uuid.uuid4(),
        variable_name=variable_name,
        value=value,
        source=source,
        quality_score=quality_score,
        cloud_cover=cloud_cover,
        status=status,
    )
    return obs


# ── Explicit Status Tests ───────────────────────────────────────────────────

def test_status_valid(qc_service):
    obs = create_mock_observation(value=3.5, variable_name="LAI")
    result = qc_service.evaluate_observation(obs)
    assert result.status == ObservationStatus.VALID
    assert result.passed is True
    assert result.is_valid is True
    assert result.reason is None


def test_status_missing_none_or_nan(qc_service):
    obs_none = create_mock_observation(value=None)
    res1 = qc_service.evaluate_observation(obs_none)
    assert res1.status == ObservationStatus.MISSING
    assert res1.passed is False
    assert res1.is_valid is False

    obs_nan = create_mock_observation(value=math.nan)
    res2 = qc_service.evaluate_observation(obs_nan)
    assert res2.status == ObservationStatus.MISSING
    assert res2.passed is False
    assert res2.is_valid is False


def test_status_rejected_low_quality_score(qc_service):
    obs = create_mock_observation(quality_score=30)
    result = qc_service.evaluate_observation(obs)
    assert result.status == ObservationStatus.REJECTED
    assert result.passed is False
    assert "Quality score" in result.reason


def test_status_rejected_high_cloud_cover(qc_service):
    obs = create_mock_observation(source=ObservationSource.SATELLITE, cloud_cover=0.45)
    result = qc_service.evaluate_observation(obs)
    assert result.status == ObservationStatus.REJECTED
    assert result.passed is False
    assert "Cloud cover" in result.reason


def test_status_rejected_unsupported_source(qc_service):
    config = QCConfig(include_sources=["SENSOR"])
    obs = create_mock_observation(source=ObservationSource.SATELLITE)
    result = qc_service.evaluate_observation(obs, config=config)
    assert result.status == ObservationStatus.REJECTED
    assert result.passed is False
    assert "Source" in result.reason


def test_status_outlier_physical_bounds_lai(qc_service):
    # LAI bounds default to [0.0, 8.0]
    obs_negative = create_mock_observation(value=-1.0, variable_name="LAI")
    res1 = qc_service.evaluate_observation(obs_negative)
    assert res1.status == ObservationStatus.OUTLIER
    assert res1.passed is False
    assert "physical bounds" in res1.reason

    obs_excessive = create_mock_observation(value=12.0, variable_name="LAI")
    res2 = qc_service.evaluate_observation(obs_excessive)
    assert res2.status == ObservationStatus.OUTLIER
    assert res2.passed is False
    assert "physical bounds" in res2.reason


def test_status_outlier_physical_bounds_sm(qc_service):
    # Soil moisture bounds default to [0.0, 0.60]
    obs_high_sm = create_mock_observation(value=0.85, variable_name="SM")
    res = qc_service.evaluate_observation(obs_high_sm)
    assert res.status == ObservationStatus.OUTLIER
    assert res.passed is False
    assert "physical bounds" in res.reason


def test_status_outlier_z_score_ensemble(qc_service):
    obs = create_mock_observation(value=7.5, variable_name="LAI")
    ens_mean = 3.0
    ens_std = 0.5  # z = (7.5 - 3.0)/0.5 = 9.0 > max_z_score 3.0
    res = qc_service.evaluate_observation(obs, ens_mean=ens_mean, ens_std=ens_std)
    assert res.status == ObservationStatus.OUTLIER
    assert res.passed is False
    assert res.z_score == pytest.approx(9.0)
    assert "Z-score" in res.reason


# ── Standalone Helper Tests ──────────────────────────────────────────────────

def test_check_physical_bounds_custom(qc_service):
    passed, reason = qc_service.check_physical_bounds("LAI", 4.0)
    assert passed is True

    passed_custom, _ = qc_service.check_physical_bounds("LAI", 10.0, config=QCConfig(custom_bounds={"LAI": (0.0, 15.0)}))
    assert passed_custom is True


def test_check_cloud_cover_fractions_and_percentages(qc_service):
    passed1, _ = qc_service.check_cloud_cover(0.15, max_cloud_cover=0.20)
    assert passed1 is True

    passed2, _ = qc_service.check_cloud_cover(25.0, max_cloud_cover=20.0)
    assert passed2 is False


# ── Batch Observation Filtering Tests ────────────────────────────────────────

def test_filter_observations_mixed_batch(qc_service):
    obs_valid = create_mock_observation(value=2.5, variable_name="LAI", quality_score=90)
    obs_cloudy = create_mock_observation(value=3.0, variable_name="LAI", cloud_cover=0.50)
    obs_outlier = create_mock_observation(value=15.0, variable_name="LAI")
    obs_low_qs = create_mock_observation(value=2.0, variable_name="LAI", quality_score=20)

    observations = [obs_valid, obs_cloudy, obs_outlier, obs_low_qs]

    # Create dummy ensemble matrices for z-score gating: row 0 corresponds to LAI
    X_f = np.tile(np.array([[2.5], [0.3], [100.0], [500.0], [500.0], [500.0]]), (1, 10))
    x_mean_f = np.array([2.5, 0.3, 100.0, 500.0, 500.0, 500.0])

    filtered = qc_service.filter_observations(observations, X_f=X_f, x_mean_f=x_mean_f)
    assert len(filtered) == 1
    assert filtered[0].value == 2.5
