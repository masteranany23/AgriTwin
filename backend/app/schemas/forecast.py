"""
backend/app/schemas/forecast.py
===============================

Pydantic schemas for the Ensemble Forward Trajectory Forecast service.
"""

import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class VariableDailyStats(BaseModel):
    """Daily forecast trajectory statistics for a single state variable."""
    date: datetime.date = Field(..., description="Forecast date.")
    mean: float = Field(..., description="Ensemble mean value.")
    std: float = Field(..., description="Ensemble standard deviation across members.")
    pi_lower_95: float = Field(..., description="95% prediction interval lower bound (2.5th percentile).")
    pi_upper_95: float = Field(..., description="95% prediction interval upper bound (97.5th percentile).")
    min_val: float = Field(..., description="Minimum member trajectory value.")
    max_val: float = Field(..., description="Maximum member trajectory value.")


class YieldForecast(BaseModel):
    """Harvest / yield (TWSO) statistical forecast at crop maturity."""
    harvest_date: datetime.date = Field(..., description="Expected harvest / maturity date.")
    mean_yield_kg_ha: float = Field(..., description="Ensemble mean yield forecast [kg/ha].")
    std_yield_kg_ha: float = Field(..., description="Ensemble yield standard deviation [kg/ha].")
    pi_lower_95_kg_ha: float = Field(..., description="95% prediction interval lower bound [kg/ha].")
    pi_upper_95_kg_ha: float = Field(..., description="95% prediction interval upper bound [kg/ha].")
    min_yield_kg_ha: float = Field(..., description="Minimum member yield [kg/ha].")
    max_yield_kg_ha: float = Field(..., description="Maximum member yield [kg/ha].")


class UncertaintyMetrics(BaseModel):
    """Statistical uncertainty metrics for the forecast trajectory."""
    yield_cv: float = Field(..., description="Coefficient of Variation (std / mean) for harvest yield.")
    yield_pi_width_kg_ha: float = Field(..., description="Width of the 95% prediction interval for yield [kg/ha].")
    yield_relative_uncertainty_pct: float = Field(..., description="Relative 95% PI width as a percentage of mean yield.")
    mean_trajectory_cv: Dict[str, float] = Field(
        default_factory=dict,
        description="Average trajectory Coefficient of Variation (std / mean) per state variable."
    )


class ForecastDiagnostics(BaseModel):
    """Model execution and assimilation context diagnostics for the forecast."""
    simulation_id: str = Field(..., description="Parent SimulationRun ID.")
    forecast_start_date: datetime.date = Field(..., description="Date from which forward trajectories were generated.")
    target_harvest_date: datetime.date = Field(..., description="Target harvest date.")
    forecast_horizon_days: int = Field(..., description="Total length of forecast trajectory in days.")
    ensemble_size: int = Field(..., description="Number of ensemble members executed in forecast forward run.")
    assimilated_cycles_count: int = Field(0, description="Total EnKF assimilation cycles completed prior to forecast.")
    latest_assimilation_date: Optional[datetime.date] = Field(None, description="Timestamp of latest EnKF state update.")
    crop_name: str = Field(..., description="Crop type.")
    variety_name: str = Field(..., description="Crop variety.")


class ForecastResponse(BaseModel):
    """Complete ensemble forward trajectory forecast response."""
    diagnostics: ForecastDiagnostics = Field(..., description="Forecast execution and assimilation diagnostics.")
    yield_forecast: YieldForecast = Field(..., description="Harvest yield forecast and statistics.")
    uncertainty_metrics: UncertaintyMetrics = Field(..., description="Yield and trajectory uncertainty metrics.")
    trajectories: Dict[str, List[VariableDailyStats]] = Field(
        ...,
        description="Daily forecast trajectories per state variable (LAI, SM, TAGP, TWSO, DVS, RFTRA)."
    )
