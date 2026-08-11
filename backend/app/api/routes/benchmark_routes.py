"""
backend/app/api/routes/benchmark_routes.py
===========================================

API endpoints for scientific validation benchmarking and compact EnKF diagnostics.
"""

import logging
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.models.simulation_run import SimulationRun
from backend.app.models.assimilation_run import AssimilationRun
from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.assimilation.services.assimilation_visualization_service import AssimilationVisualizationService
from backend.app.benchmarking.evaluator import BenchmarkEvaluator
from backend.app.benchmarking.enkf_diagnostics import EnKFDiagnosticsExtractor
from backend.app.benchmarking.schemas import (
    BenchmarkComparisonResult,
    BenchmarkEvaluateRequest,
    RunDiagnosticsSummary,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/evaluate",
    response_model=BenchmarkComparisonResult,
    summary="Benchmark open-loop vs. assimilated WOFOST against real ground truth",
    description=(
        "Compares open-loop WOFOST simulation outputs against assimilated WOFOST outputs "
        "using real external ground truth measurements. Computes MAE, RMSE, Bias, R², "
        "and ensemble confidence interval coverage. Never fabricates ground truth."
    ),
    tags=["Benchmarking"],
)
def evaluate_benchmark(
    request: BenchmarkEvaluateRequest,
    db: Session = Depends(get_db),
) -> BenchmarkComparisonResult:
    """Evaluate open-loop vs. assimilated simulation accuracy against supplied real ground truth."""
    # 1. Fetch baseline simulation run
    sim_run = db.query(SimulationRun).filter(SimulationRun.id == request.simulation_id).first()
    if not sim_run:
        raise HTTPException(
            status_code=404,
            detail=f"SimulationRun with ID {request.simulation_id} not found."
        )

    # 2. Retrieve comparative timeseries data via visualization service
    vis_service = AssimilationVisualizationService(db)
    ts_data = vis_service.get_timeseries(request.simulation_id)

    target_var = request.variable.upper()
    if target_var not in ts_data:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{request.variable}' is not supported for benchmarking. Allowed: {list(ts_data.keys())}"
        )

    series_points = ts_data[target_var]
    open_loop_series: Dict = {}
    assimilated_series: Dict = {}

    for pt in series_points:
        d = pt["date"]
        open_loop_series[d] = pt["open_loop"]
        assimilated_series[d] = pt["assimilated"]

    # 3. Perform evaluation using BenchmarkEvaluator
    evaluator = BenchmarkEvaluator()
    result = evaluator.evaluate(
        simulation_id=request.simulation_id,
        variable=request.variable,
        open_loop_series=open_loop_series,
        assimilated_series=assimilated_series,
        ground_truth_points=request.ground_truth,
    )

    return result


@router.get(
    "/diagnostics/{simulation_id}",
    response_model=RunDiagnosticsSummary,
    summary="Get compact EnKF diagnostics for an assimilation run",
    description=(
        "Returns compact EnKF diagnostics (innovation, prior/posterior ensemble spread, "
        "valid/rejected observation counts, state update magnitude) across all cycles of an assimilation run."
    ),
    tags=["Benchmarking", "Assimilation"],
)
def get_enkf_diagnostics(
    simulation_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> RunDiagnosticsSummary:
    """Fetch compact EnKF diagnostics summary for the latest assimilation run of a simulation."""
    sim_run = db.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
    if not sim_run:
        raise HTTPException(
            status_code=404,
            detail=f"SimulationRun with ID {simulation_id} not found."
        )

    latest_run = (
        db.query(AssimilationRun)
        .filter(AssimilationRun.simulation_id == simulation_id)
        .order_by(AssimilationRun.started_at.desc())
        .first()
    )
    if not latest_run:
        raise HTTPException(
            status_code=404,
            detail=f"No assimilation runs found for simulation run ID {simulation_id}."
        )

    states = (
        db.query(AssimilationState)
        .filter(AssimilationState.assimilation_run_id == latest_run.id)
        .order_by(AssimilationState.assimilation_time.asc())
        .all()
    )

    cycle_diagnostics = [
        EnKFDiagnosticsExtractor.extract_from_db_state(s) for s in states
    ]

    return EnKFDiagnosticsExtractor.summarize_run(
        simulation_id=simulation_id,
        assimilation_run_id=latest_run.id,
        cycles=cycle_diagnostics,
    )
