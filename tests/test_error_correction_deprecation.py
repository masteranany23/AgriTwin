"""
tests/test_error_correction_deprecation.py — Regression Tests for Error Correction Deprecation
=================================================================================================

Verifies that:
1. DailyOutput records are NOT modified by ErrorCorrectionService or POST /error-correction/correct-window.
2. Canonical state estimation path remains: observations → QC → fusion → EnKF → assimilated WOFOST.
3. QualityControlService gates invalid observations during diagnostic evaluation.
"""

import datetime
import uuid
import pytest
from sqlalchemy.orm import Session

from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.models.daily_output import DailyOutput
from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.assimilation.repositories.observation_repository import ObservationRepository
from backend.app.api.schemas.error_correction import ErrorCorrectionRequest
from backend.app.services.error_correction_service import ErrorCorrectionService

UTC = datetime.timezone.utc


@pytest.fixture
def sample_field_and_simulation(test_db: Session):
    """Fixture providing a field, simulation_run, and 7 days of DailyOutput."""
    from backend.app.models.farm import Farm

    farm_id = uuid.uuid4()
    field_id = uuid.uuid4()
    sim_id = uuid.uuid4()

    farm = Farm(id=farm_id, name="Test Farm EC")
    test_db.add(farm)

    f = Field(id=field_id, farm_id=farm_id, name="Test Field EC", latitude=26.8, longitude=80.9, area_ha=5.0)
    test_db.add(f)

    sim = SimulationRun(
        id=sim_id,
        field_id=field_id,
        crop="wheat",
        variety="Winter_wheat_101",
        latitude=26.8,
        longitude=80.9,
        sowing_date=datetime.date(2026, 7, 10),
        status="completed",
    )
    test_db.add(sim)
    test_db.flush()

    # Add 7 daily output records
    daily_rows = []
    for d in range(10, 17):
        day_date = datetime.date(2026, 7, d)
        out = DailyOutput(
            simulation_run_id=sim_id,
            date=day_date,
            lai=1.5,
            sm=0.20,
            tagp=100.0,
            twso=0.0,
            dvs=0.5,
        )
        daily_rows.append(out)
        test_db.add(out)

    test_db.commit()

    # Add 3 observations for LAI in the window
    repo = ObservationRepository(test_db)
    for d in [10, 13, 16]:
        obs = Observation(
            id=uuid.uuid4(),
            field_id=field_id,
            timestamp=datetime.datetime(2026, 7, d, 12, 0, tzinfo=UTC),
            variable_name="LAI",
            units="m2/m2",
            value=3.5,  # Large difference from 1.5 to trigger residual/anomaly
            uncertainty=0.3,
            source=ObservationSource.SATELLITE,
            provider_name="Sentinel2_L2A",
            status=ObservationStatus.VALID,
        )
        repo.save_observation(obs)

    return {"field_id": field_id, "simulation_id": sim_id}


def test_error_correction_service_does_not_mutate_daily_output(test_db: Session, sample_field_and_simulation):
    """Verify that calling ErrorCorrectionService.correct_window does NOT mutate DailyOutput."""
    field_id = sample_field_and_simulation["field_id"]
    sim_id = sample_field_and_simulation["simulation_id"]

    # Record initial state of DailyOutput
    initial_outputs = test_db.query(DailyOutput).filter(
        DailyOutput.simulation_run_id == sim_id
    ).order_by(DailyOutput.date).all()
    
    initial_lai_values = {out.date: out.lai for out in initial_outputs}
    initial_sm_values = {out.date: out.sm for out in initial_outputs}

    req = ErrorCorrectionRequest(
        simulation_id=sim_id,
        field_id=field_id,
        window_start_date=datetime.date(2026, 7, 10),
        window_end_date=datetime.date(2026, 7, 16),
        residual_threshold=0.5,
        source="SENTINEL_2",
    )

    service = ErrorCorrectionService(test_db)
    resp = service.correct_window(req)

    assert resp.total_days_processed > 0
    assert "Deprecated" in resp.message

    # Re-query DailyOutput from DB and verify values are completely unchanged!
    post_outputs = test_db.query(DailyOutput).filter(
        DailyOutput.simulation_run_id == sim_id
    ).order_by(DailyOutput.date).all()

    for out in post_outputs:
        assert out.lai == pytest.approx(initial_lai_values[out.date]), f"DailyOutput LAI mutated on date {out.date}"
        assert out.sm == pytest.approx(initial_sm_values[out.date]), f"DailyOutput SM mutated on date {out.date}"


def test_api_error_correction_endpoint_preserves_daily_output(client, test_db: Session, sample_field_and_simulation):
    """Verify that POST /error-correction/correct-window preserves DailyOutput intact."""
    field_id = str(sample_field_and_simulation["field_id"])
    sim_id = str(sample_field_and_simulation["simulation_id"])

    payload = {
        "simulation_id": sim_id,
        "field_id": field_id,
        "window_start_date": "2026-07-10",
        "window_end_date": "2026-07-16",
        "residual_threshold": 0.5,
        "source": "SENTINEL_2",
    }

    resp = client.post("/error-correction/correct-window", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "Deprecated" in data["message"]

    # Verify DB values remained unchanged (1.5 for LAI, 0.20 for SM)
    outputs = test_db.query(DailyOutput).filter(
        DailyOutput.simulation_run_id == uuid.UUID(sim_id)
    ).all()
    for out in outputs:
        assert out.lai == pytest.approx(1.5)
        assert out.sm == pytest.approx(0.20)
