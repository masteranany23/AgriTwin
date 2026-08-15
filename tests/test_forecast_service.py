"""
tests/test_forecast_service.py
===============================

Unit and API integration tests for AgriTwin ForecastService:
- Trajectory length verification
- Mean, std, and 95% prediction interval bounds validation
- Harvest yield forecast and uncertainty metrics
- GET /assimilation/{simulation_id}/forecast endpoint
"""

import datetime
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.farm import Farm
from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.services.forecast_service import ForecastService


@pytest.fixture
def test_setup(test_engine):
    """Fixture providing Farm, Field, and SimulationRun records."""
    farm_id = uuid.uuid4()
    field_id = uuid.uuid4()

    with Session(test_engine) as db:
        farm = Farm(id=farm_id, name="Forecast Farm")
        db.add(farm)
        db.commit()

        field = Field(
            id=field_id,
            farm_id=farm_id,
            name="Forecast Field",
            latitude=52.0,
            longitude=5.5,
            elevation_m=12.0,
        )
        db.add(field)
        db.commit()

        sim_run = SimulationRun(
            field_id=field.id,
            crop="wheat",
            variety="Winter_wheat_101",
            sowing_date=datetime.date(2026, 3, 1),
            harvest_date=datetime.date(2026, 5, 1),
            latitude=52.0,
            longitude=5.5,
            status="COMPLETED",
            run_type="irrigated",
            use_real_weather=False,
        )
        db.add(sim_run)
        db.commit()
        db.refresh(sim_run)
        sim_id = sim_run.id

    return {"farm_id": farm_id, "field_id": field_id, "sim_id": sim_id}


def test_forecast_service_trajectory_length_mean_std_bounds(test_engine, test_setup):
    """Verify trajectory length, mean, std, and 95% prediction interval bounds."""
    sim_id = test_setup["sim_id"]

    with Session(test_engine) as db:
        sim_run = db.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
        service = ForecastService(db)

        # Run forecast with 5 ensemble members for fast execution
        response = service.generate_forecast(simulation_id=sim_run.id, ensemble_size=5)

        # 1. Trajectory Diagnostics
        assert response.diagnostics.simulation_id == str(sim_run.id)
        assert response.diagnostics.ensemble_size == 5
        assert response.diagnostics.crop_name == "wheat"

        # 2. Check Trajectories
        assert "LAI" in response.trajectories
        assert "TWSO" in response.trajectories
        assert "TAGP" in response.trajectories

        lai_traj = response.trajectories["LAI"]
        expected_days = (sim_run.harvest_date - sim_run.sowing_date).days + 1
        assert len(lai_traj) > 0
        assert len(lai_traj) <= expected_days

        # 3. Check Statistical Bounds (min <= pi_lower <= mean <= pi_upper <= max)
        for point in lai_traj:
            assert point.min_val <= point.pi_lower_95 + 1e-6
            assert point.pi_lower_95 <= point.mean + 1e-6
            assert point.mean <= point.pi_upper_95 + 1e-6
            assert point.pi_upper_95 <= point.max_val + 1e-6
            assert point.std >= 0.0

        # 4. Check Yield Forecast & Uncertainty Metrics
        yf = response.yield_forecast
        assert yf.harvest_date == lai_traj[-1].date
        assert yf.mean_yield_kg_ha >= 0.0
        assert yf.pi_lower_95_kg_ha <= yf.mean_yield_kg_ha <= yf.pi_upper_95_kg_ha
        assert yf.min_yield_kg_ha <= yf.max_yield_kg_ha

        um = response.uncertainty_metrics
        assert um.yield_cv >= 0.0
        assert um.yield_pi_width_kg_ha == pytest.approx(yf.pi_upper_95_kg_ha - yf.pi_lower_95_kg_ha)


def test_forecast_api_endpoint(client: TestClient, test_setup):
    """Test GET /assimilation/{simulation_id}/forecast endpoint via TestClient."""
    sim_id = test_setup["sim_id"]

    res = client.get(f"/assimilation/{sim_id}/forecast?ensemble_size=5")
    assert res.status_code == 200, res.text
    data = res.json()

    assert "diagnostics" in data
    assert "yield_forecast" in data
    assert "uncertainty_metrics" in data
    assert "trajectories" in data

    assert data["diagnostics"]["simulation_id"] == str(sim_id)
    assert data["diagnostics"]["ensemble_size"] == 5
    assert len(data["trajectories"]["LAI"]) > 0
