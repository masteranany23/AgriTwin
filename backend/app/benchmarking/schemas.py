"""
backend/app/benchmarking/schemas.py
====================================

Pydantic schemas for benchmark evaluation requests, metric comparisons,
and compact EnKF diagnostics.
"""

import datetime
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class GroundTruthPoint(BaseModel):
    """External ground truth observation point for scientific evaluation."""
    date: datetime.date = Field(..., description="Observation date")
    variable: str = Field(..., description="Variable name (e.g., LAI, SM, TWSO, TAGP)")
    value: float = Field(..., description="Real ground truth measurement")


class BenchmarkEvaluateRequest(BaseModel):
    """Request payload to benchmark open-loop vs assimilated predictions against real ground truth."""
    simulation_id: uuid.UUID = Field(..., description="ID of the baseline open-loop simulation run")
    variable: str = Field(default="LAI", description="Target variable to evaluate (e.g. LAI, SM, TWSO, TAGP)")
    ground_truth: List[GroundTruthPoint] = Field(..., description="List of real ground truth measurements")
    confidence_level: float = Field(default=0.95, ge=0.50, le=0.99, description="Confidence interval level for coverage calculation")


class MetricSet(BaseModel):
    """Statistical metric set for prediction accuracy."""
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    bias: Optional[float] = Field(None, description="Bias (Mean Error: predicted - ground truth)")
    r2: Optional[float] = Field(None, description="Coefficient of Determination R²")
    ensemble_interval_coverage: Optional[float] = Field(
        None, description="Fraction of ground truth observations within ensemble confidence interval"
    )
    sample_size: int = Field(0, description="Number of aligned ground truth observation points evaluated")


class BenchmarkComparisonResult(BaseModel):
    """Comparison of open-loop WOFOST vs assimilated WOFOST against real external ground truth."""
    simulation_id: uuid.UUID
    variable: str
    has_ground_truth: bool = Field(..., description="True if real external ground truth observations were provided and matched")
    message: str = Field(..., description="Evaluation status message")
    open_loop: MetricSet
    assimilated: MetricSet
    rmse_reduction: Optional[float] = Field(None, description="Absolute decrease in RMSE (open_loop_rmse - assimilated_rmse)")
    rmse_improvement_pct: Optional[float] = Field(None, description="Percentage reduction in RMSE")


class CompactCycleDiagnostics(BaseModel):
    """Compact EnKF diagnostic metrics for a single assimilation cycle."""
    cycle_date: datetime.date
    variables_updated: List[str]
    valid_obs_count: int = Field(..., description="Number of valid observations assimilated in this cycle")
    rejected_obs_count: int = Field(..., description="Number of observations rejected by QC/state filtering")
    innovation: Dict[str, Optional[float]] = Field(..., description="Innovation vector (y - H*x_mean)")
    ensemble_spread_prior: Dict[str, Optional[float]] = Field(..., description="Prior ensemble standard deviation per state variable")
    posterior_spread: Dict[str, Optional[float]] = Field(..., description="Posterior ensemble standard deviation per state variable")
    state_update_magnitude: Dict[str, Optional[float]] = Field(..., description="Absolute update magnitude |x_post - x_prior| per variable")


class RunDiagnosticsSummary(BaseModel):
    """Summary of compact EnKF diagnostics across an entire assimilation run."""
    simulation_id: uuid.UUID
    assimilation_run_id: Optional[uuid.UUID]
    total_cycles: int
    executed_cycles: int
    total_valid_obs: int
    total_rejected_obs: int
    avg_state_update_magnitude: Dict[str, Optional[float]]
    avg_innovation: Dict[str, Optional[float]]
    mean_prior_spread: Dict[str, Optional[float]]
    mean_posterior_spread: Dict[str, Optional[float]]
    cycles: List[CompactCycleDiagnostics]
