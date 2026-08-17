"""
tests/test_forecast_service.py
===============================

Unit and API integration tests for AgriTwin ForecastService:
- Trajectory length verification
- Mean, std, and 95% prediction interval bounds validation
- Extended response reporting: open-loop, assimilated, hybrid, uncertainty, observations, mode, explanation
- Fallback behavior when no validated residual model exists (never claims hybrid prediction)
- Hybrid result reporting when a validated residual model is active
- GET /assimilation/{simulation_id}/forecast endpoint
"""

import datetime
import uuid
from typing import Optional
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.models.farm import Farm
from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.residual.base import ResidualModel
from backend.app.residual.registry import global_residual_registry
from backend.app.residual.schemas import CorrectedYieldPrediction, ModelMetadata, ResidualPrediction
from backend.app.services.forecast_service import ForecastService


class DummyValidatedResidualModel(ResidualModel):
    """Mock validated residual model for testing hybrid forecast response."""

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="wheat_validated_v1",
            name="Wheat Validated Bias Correction",
            version="1.0.0",
            description="Validated bias correction model",
            validated=True,
            supported_crops=["wheat"],
            supported_regions=["GLOBAL"],
        )

    def is_applicable(self, crop: str, region: Optional[str] = None) -> bool:
        return crop.lower() == "wheat"

    def is_available(self, crop: str, region: Optional[str] = None) -> bool:
        return self.is_applicable(crop, region)

    def predict_residual(self, crop: str, region: Optional[str] = None, **kwargs) -> ResidualPrediction:
        return ResidualPrediction(
            residual_correction_kg_ha=350.0,
            residual_uncertainty_kg_ha=50.0,
            is_validated_correction=True,
            model_id="wheat_validated_v1",
            model_version="1.0.0",
        )

    def predict_uncertainty(self, crop: str, region: Optional[str] = None, **kwargs) -> float:
        return 50.0

    def apply_correction(self, assimilated_yield: float, crop: str, region: Optional[str] = None, **kwargs) -> CorrectedYieldPrediction:
        return CorrectedYieldPrediction(
            assimilated_yield_kg_ha=assimilated_yield,
            residual_correction_kg_ha=350.0,
            corrected_yield_kg_ha=assimilated_yield + 350.0,
            residual_uncertainty_kg_ha=50.0,
            is_validated_correction=True,
            model_id="wheat_validated_v1",
            model_version="1.0.0",
        )


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
            yield_kg_ha=4500.0,
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

        # 5. Check Extended Forecast Response Fields (Fallback / Default NoResidualModel mode)
        assert response.open_loop_result.mean_yield_kg_ha == 4500.0
        assert response.assimilated_result.mean_yield_kg_ha == pytest.approx(yf.mean_yield_kg_ha)
        
        # CRITICAL SAFETY REQUIREMENT: Must be None when no validated residual model exists
        assert response.hybrid_result is None
        assert response.forecast_mode in ["ASSIMILATED_ENSEMBLE", "OPEN_LOOP_BASELINE"]
        assert "No validated ML residual model is active" in response.confidence_explanation
        assert isinstance(response.observation_summary.active_sources, list)


def test_forecast_service_with_validated_residual_model(test_engine, test_setup):
    """Verify hybrid_result is populated when a validated residual model is registered."""
    sim_id = test_setup["sim_id"]
    dummy_model = DummyValidatedResidualModel()

    # Register validated model in global registry
    global_residual_registry.register_model(dummy_model)

    try:
        with Session(test_engine) as db:
            sim_run = db.query(SimulationRun).filter(SimulationRun.id == sim_id).first()
            service = ForecastService(db)

            response = service.generate_forecast(simulation_id=sim_run.id, ensemble_size=5)

            assert response.hybrid_result is not None
            assert response.hybrid_result.model_id == "wheat_validated_v1"
            assert response.hybrid_result.residual_correction_kg_ha == 350.0
            assert response.hybrid_result.corrected_yield_kg_ha == response.assimilated_result.mean_yield_kg_ha + 350.0
            assert response.forecast_mode == "HYBRID_RESIDUAL"
            assert "Applied validated residual model 'wheat_validated_v1'" in response.confidence_explanation

    finally:
        # Restore default registry state
        global_residual_registry.reset()


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

    # Extended response assertion
    assert "open_loop_result" in data
    assert "assimilated_result" in data
    assert "hybrid_result" in data
    assert data["hybrid_result"] is None
    assert "observation_summary" in data
    assert "forecast_mode" in data
    assert "confidence_explanation" in data

    assert data["diagnostics"]["simulation_id"] == str(sim_id)
    assert data["diagnostics"]["ensemble_size"] == 5
    assert len(data["trajectories"]["LAI"]) > 0
