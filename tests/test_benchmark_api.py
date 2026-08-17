"""
tests/test_benchmark_api.py
============================

API tests for scientific benchmarking and EnKF diagnostics endpoints.
"""

import datetime
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.session import get_db
from backend.app.models.simulation_run import SimulationRun
from backend.app.models.daily_output import DailyOutput
from backend.app.models.assimilation_run import AssimilationRun
from backend.app.assimilation.models.assimilation_state import AssimilationState

client = TestClient(app)


def test_evaluate_benchmark_endpoint_no_gt(test_db):
    """Test POST /benchmark/evaluate when no matching ground truth is provided."""
    app.dependency_overrides[get_db] = lambda: test_db
    try:
        sim_id = uuid.uuid4()
        sim_run = SimulationRun(
            id=sim_id,
            crop="wheat",
            variety="Winter_wheat_105",
            sowing_date=datetime.date(2025, 4, 1),
            harvest_date=datetime.date(2025, 8, 1),
            latitude=52.0,
            longitude=5.0,
            status="SUCCESS",
        )
        test_db.add(sim_run)
        test_db.commit()

        payload = {
            "simulation_id": str(sim_id),
            "variable": "LAI",
            "ground_truth": []
        }

        res = client.post("/benchmark/evaluate", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["has_ground_truth"] is False
        assert data["open_loop"]["sample_size"] == 0
        assert data["assimilated"]["sample_size"] == 0
        assert "No real ground truth" in data["message"]
    finally:
        app.dependency_overrides.clear()


def test_evaluate_benchmark_endpoint_with_gt(test_db):
    """Test POST /benchmark/evaluate when real ground truth is provided."""
    app.dependency_overrides[get_db] = lambda: test_db
    try:
        sim_id = uuid.uuid4()
        d1 = datetime.date(2025, 5, 1)

        sim_run = SimulationRun(
            id=sim_id,
            crop="wheat",
            variety="Winter_wheat_105",
            sowing_date=datetime.date(2025, 4, 1),
            harvest_date=datetime.date(2025, 8, 1),
            latitude=52.0,
            longitude=5.0,
            status="SUCCESS",
        )
        test_db.add(sim_run)

        # Add daily output row
        d_out = DailyOutput(
            simulation_run_id=sim_id,
            date=d1,
            lai=1.0,
            sm=0.2,
            tagp=500.0,
            twso=0.0,
        )
        test_db.add(d_out)
        test_db.commit()

        payload = {
            "simulation_id": str(sim_id),
            "variable": "LAI",
            "ground_truth": [
                {"date": str(d1), "variable": "LAI", "value": 1.5}
            ]
        }

        res = client.post("/benchmark/evaluate", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["has_ground_truth"] is True
        assert data["open_loop"]["sample_size"] == 1
        assert data["assimilated"]["sample_size"] == 1
    finally:
        app.dependency_overrides.clear()


def test_get_enkf_diagnostics_endpoint(test_db):
    """Test GET /assimilation/{simulation_id}/diagnostics endpoint."""
    app.dependency_overrides[get_db] = lambda: test_db
    try:
        sim_id = uuid.uuid4()
        run_id = uuid.uuid4()
        d1 = datetime.datetime(2025, 5, 1, tzinfo=datetime.timezone.utc)

        sim_run = SimulationRun(
            id=sim_id,
            crop="wheat",
            variety="Winter_wheat_105",
            sowing_date=datetime.date(2025, 4, 1),
            harvest_date=datetime.date(2025, 8, 1),
            latitude=52.0,
            longitude=5.0,
            status="SUCCESS",
        )
        test_db.add(sim_run)

        assim_run = AssimilationRun(
            id=run_id,
            simulation_id=sim_id,
            ensemble_size=10,
            status="COMPLETED",
            total_cycles=1,
            executed_cycles=1,
            skipped_cycles=0,
            observations_used=1,
        )
        test_db.add(assim_run)

        state = AssimilationState(
            id=uuid.uuid4(),
            simulation_run_id=sim_id,
            assimilation_run_id=run_id,
            assimilation_time=d1,
            ensemble_mean={"lai": 1.5},
            ensemble_covariance={"lai": 0.04},
            observation_vector={"lai": 1.6},
            innovation_vector={"lai": 0.6},
            kalman_gain={},
            updated_state_vector={"lai": 1.5},
            forecast_state_vector={"lai": 1.0},
            number_of_members=10,
            observation_count=1,
        )
        test_db.add(state)
        test_db.commit()

        res = client.get(f"/assimilation/{sim_id}/diagnostics")
        assert res.status_code == 200
        data = res.json()

        assert data["simulation_id"] == str(sim_id)
        assert data["total_cycles"] == 1
        assert data["total_valid_obs"] == 1
        assert len(data["cycles"]) == 1
        assert data["cycles"][0]["valid_obs_count"] == 1
    finally:
        app.dependency_overrides.clear()
