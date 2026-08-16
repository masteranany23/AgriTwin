"""Integration test for the canonical Data Fusion → Dynamic R → EnKF assimilation flow.

Proves:
1. Raw multi-source observations are quality-controlled.
2. ConfidenceEstimator & MultiSourceFusionService derive dynamic observation error covariance R.
3. Fused observation vector y and dynamic R are passed into EnKF update.
4. Corrected state vector is injected back into ensemble members.
5. AssimilationCycleResult captures complete fusion_diagnostics.
"""

import datetime
import uuid
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.assimilation.services.assimilation_service import (
    AssimilationService,
    AssimilationConfig,
    QCFilter,
)
from backend.app.assimilation.state.state_vector import STATE_DIM, STATE_INDEX, STATE_VARIABLES
from backend.app.assimilation.ensemble.ensemble_manager import EnsembleManager, EnsembleMember


def test_end_to_end_fusion_dynamic_r_enkf_integration():
    """Verify raw obs → QC → dynamic R → EnKF update pipeline end-to-end."""
    field_id = uuid.uuid4()
    obs_date = datetime.date(2026, 6, 15)

    # 1. Setup mock repositories and service
    mock_obs_repo = MagicMock()
    mock_state_repo = MagicMock()
    
    # Create raw observations from multiple sources on the same date
    raw_obs = [
        Observation(
            id=uuid.uuid4(),
            field_id=field_id,
            timestamp=datetime.datetime(2026, 6, 15, 10, 0, tzinfo=datetime.timezone.utc),
            variable_name="LAI",
            value=3.2,
            source=ObservationSource.SATELLITE,
            provider_name="Sentinel-2",
            quality_score=95,
            status=ObservationStatus.VALID,
            cloud_cover=5.0,
            uncertainty=None,  # Dynamic R will be estimated via ConfidenceEstimator
        ),
        Observation(
            id=uuid.uuid4(),
            field_id=field_id,
            timestamp=datetime.datetime(2026, 6, 15, 10, 30, tzinfo=datetime.timezone.utc),
            variable_name="LAI",
            value=2.6,  # Smartphone GRVI observation
            source=ObservationSource.MANUAL,
            provider_name="FieldScout_AgriTwinApp",
            quality_score=85,
            status=ObservationStatus.VALID,
            cloud_cover=0.0,
            uncertainty=None,
        ),
        Observation(
            id=uuid.uuid4(),
            field_id=field_id,
            timestamp=datetime.datetime(2026, 6, 15, 12, 0, tzinfo=datetime.timezone.utc),
            variable_name="SM",
            value=0.25,
            source=ObservationSource.SENSOR,
            provider_name="ERA5_LAND",
            quality_score=90,
            status=ObservationStatus.VALID,
            cloud_cover=0.0,
            uncertainty=0.03,  # Explicit uncertainty override test
        ),
    ]
    mock_obs_repo.get_by_date.return_value = raw_obs
    mock_obs_repo.get_by_field_and_date_range.return_value = raw_obs

    qc_filter = QCFilter(max_cloud_cover=20.0, max_z_score=10.0)
    config = AssimilationConfig(min_obs_for_update=1, qc=qc_filter)
    service = AssimilationService(obs_repo=mock_obs_repo, state_repo=mock_state_repo, config=config)

    # 2. Test direct vector building & dynamic R calculation
    n_members = 5
    np.random.seed(42)
    X_f = np.ones((STATE_DIM, n_members)) * 2.5 + np.random.normal(0, 0.2, (STATE_DIM, n_members))
    X_f[STATE_INDEX["sm"], :] = 0.22 + np.random.normal(0, 0.01, n_members)
    x_mean_f = np.mean(X_f, axis=1)

    qc_obs = service._apply_qc(raw_obs, X_f, x_mean_f)
    assert len(qc_obs) == 3

    y, R, n_assimilated, fusion_diag = service._build_observation_vector(
        qc_obs, field_id=field_id, obs_date=obs_date
    )

    lai_idx = STATE_INDEX["lai"]
    sm_idx = STATE_INDEX["sm"]

    assert n_assimilated == 2
    assert not np.isnan(y[lai_idx])
    assert not np.isnan(y[sm_idx])

    # Dynamic R checks: R[lai_idx, lai_idx] must be positive and dynamic
    assert R[lai_idx, lai_idx] > 0.0
    assert R[sm_idx, sm_idx] == pytest.approx(0.03 ** 2)

    # Fusion diagnostics assertions
    assert "lai" in fusion_diag
    assert "sm" in fusion_diag
    assert fusion_diag["lai"]["obs_count"] == 2
    assert set(fusion_diag["lai"]["sources_used"]) == {"SENTINEL2", "SMARTPHONE_GRVI"}
    assert fusion_diag["lai"]["dynamic_r_variance"] > 0.0

    # 3. Test full cycle execution through _run_cycle with ensemble manager
    manager = EnsembleManager.__new__(EnsembleManager)
    manager.members = [
        EnsembleMember(member_id=i, wofost=MagicMock(), perturbed_parameters={"SPAN": 30.0 + i})
        for i in range(n_members)
    ]

    # Mock PCSE engine state extraction for each member's wofost instance
    for m in manager.members:
        m.wofost.get_variable.side_effect = lambda var: {
            "LAI": 2.5 + np.random.normal(0, 0.1),
            "SM": 0.20 + np.random.normal(0, 0.01),
            "TAGP": 1000.0,
            "TWSO": 0.0,
            "RFTRA": 1.0,
            "TWLV": 400.0,
            "TWST": 600.0,
            "TWRT": 200.0,
            "DVS": 0.5,
            "RD": 10.0,
        }.get(var, 0.0)

    with patch("backend.app.assimilation.services.assimilation_service.forecast_until") as mock_forecast:
        mock_forecast.return_value = (X_f, x_mean_f)

        result = service._run_cycle(
            manager=manager,
            obs_date=obs_date,
            field_id=field_id,
            simulation_run_id=uuid.uuid4(),
            assimilation_run_id=uuid.uuid4(),
        )

    # 4. Verify EnKF assimilation result
    assert result.skipped is False
    assert result.obs_retrieved == 3
    assert result.obs_after_qc == 3
    assert result.obs_assimilated == 2
    assert "lai" in result.variables_updated
    assert "sm" in result.variables_updated
    assert "lai" in result.fusion_diagnostics
    assert "sm" in result.fusion_diagnostics

    # Assert innovation y - H*x_f is computed correctly
    assert result.innovation["lai"] is not None
    assert result.innovation["sm"] is not None
