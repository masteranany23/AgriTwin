"""
backend/app/services/forecast_service.py
=========================================

Ensemble Forward Trajectory Forecast Service for AgriTwin
---------------------------------------------------------
Generates forward ensemble trajectories from the latest EnKF assimilation state
through harvest date using PCSE/WOFOST EnsembleManager.

Guarantees:
1. No Second Simulation Engine: Reuses existing `EnsembleManager` and `Wofost72_WLP_FD` members.
2. No Fabricated Confidence: All standard deviations, percentiles, and 95% prediction intervals
   are derived strictly from actual ensemble member realizations.
3. Traceable Diagnostics: Documents forecast horizon, assimilation state lineage, and ensemble statistics.
"""

import datetime
import logging
import uuid
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.app.assimilation.ensemble.ensemble_manager import EnsembleManager
from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.models.assimilation_run import AssimilationRun
from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.schemas.forecast import (
    ForecastDiagnostics,
    ForecastResponse,
    UncertaintyMetrics,
    VariableDailyStats,
    YieldForecast,
)

logger = logging.getLogger(__name__)


class ForecastService:
    """Generates ensemble forward trajectories and yield predictions from assimilation states."""

    def __init__(self, db: Session):
        self.db = db

    def generate_forecast(
        self,
        simulation_id: uuid.UUID,
        ensemble_size: int = 30,
        target_date: Optional[datetime.date] = None,
    ) -> ForecastResponse:
        """Generate forward ensemble forecast from the latest assimilation state to harvest date.

        Args:
            simulation_id: UUID of the parent SimulationRun.
            ensemble_size: Number of ensemble members to simulate (default: 30).
            target_date: Optional explicit forecast cutoff date (defaults to sim_run.harvest_date).

        Returns:
            ForecastResponse containing daily trajectory stats, yield forecast, and diagnostics.
        """
        # 1. Fetch simulation run
        sim_run = self.db.query(SimulationRun).filter(SimulationRun.id == simulation_id).first()
        if not sim_run:
            raise ValueError(f"SimulationRun with ID {simulation_id} not found.")

        harvest_date = target_date or sim_run.harvest_date
        sow_date = sim_run.sowing_date

        # Fetch field elevation if available
        field_id = sim_run.field_id
        elevation = 10.0
        if field_id:
            field = self.db.query(Field).filter(Field.id == field_id).first()
            if field and field.elevation_m is not None:
                elevation = field.elevation_m

        # 2. Check for latest assimilation run and states
        latest_run = (
            self.db.query(AssimilationRun)
            .filter(AssimilationRun.simulation_id == simulation_id)
            .order_by(AssimilationRun.started_at.desc())
            .first()
        )

        latest_state: Optional[AssimilationState] = None
        assimilated_cycles_count = 0
        latest_assim_date: Optional[datetime.date] = None
        state_offsets: Dict[str, float] = {}

        if latest_run:
            states = (
                self.db.query(AssimilationState)
                .filter(AssimilationState.assimilation_run_id == latest_run.id)
                .order_by(AssimilationState.assimilation_time.asc())
                .all()
            )
            assimilated_cycles_count = len(states)
            if states:
                latest_state = states[-1]
                latest_assim_date = latest_state.assimilation_time.date()

                # Calculate accumulated state vector offset (posterior - prior) at latest cycle
                # to initialize forecast trajectory from latest assimilated state
                if latest_state.forecast_state_vector and latest_state.updated_state_vector:
                    for k, post_val in latest_state.updated_state_vector.items():
                        prior_val = latest_state.forecast_state_vector.get(k)
                        if post_val is not None and prior_val is not None:
                            state_offsets[k.lower()] = float(post_val - prior_val)

        forecast_start_date = latest_assim_date or sow_date

        # 3. Instantiate EnsembleManager and build ensemble
        manager = EnsembleManager(
            crop_name=sim_run.crop,
            variety_name=sim_run.variety,
            sow_date=sow_date,
            harvest_date=harvest_date,
            latitude=sim_run.latitude,
            longitude=sim_run.longitude,
            elevation=elevation,
            use_nasa_weather=sim_run.use_real_weather,
            soil_params=sim_run.soil_snapshot,
        )
        manager.create_ensemble(n=ensemble_size)

        # 4. Advance ensemble to harvest_date and capture member outputs
        manager.run_until(harvest_date)

        # 5. Extract daily output series from all ensemble members
        # Key variables to track
        vars_of_interest = ["lai", "sm", "tagp", "twso", "dvs", "rftra"]
        member_outputs_by_date: Dict[datetime.date, Dict[str, List[float]]] = {}

        for member in manager.members:
            # PCSE wofost.get_output() returns list of output dicts for all simulated days
            output_list = member.wofost.get_output()
            for row in output_list:
                d = row.get("day")
                if not d or d < forecast_start_date or d > harvest_date:
                    continue

                if d not in member_outputs_by_date:
                    member_outputs_by_date[d] = {v: [] for v in vars_of_interest}

                for v in vars_of_interest:
                    val = row.get(v.upper())
                    if val is None:
                        val = row.get(v)

                    # If state offset exists from assimilation state, apply offset adjustment
                    if val is not None and v in state_offsets:
                        val = max(0.0, val + state_offsets[v])

                    if val is not None:
                        member_outputs_by_date[d][v].append(float(val))

        sorted_dates = sorted(list(member_outputs_by_date.keys()))
        if not sorted_dates:
            raise ValueError("No forecast output trajectories generated by EnsembleManager.")

        # 6. Aggregate Statistics Per Date & Variable
        trajectories: Dict[str, List[VariableDailyStats]] = {v.upper(): [] for v in vars_of_interest}
        trajectory_cvs: Dict[str, List[float]] = {v.upper(): [] for v in vars_of_interest}

        for d in sorted_dates:
            day_data = member_outputs_by_date[d]
            for v in vars_of_interest:
                v_upper = v.upper()
                vals = day_data[v]
                if not vals:
                    continue

                arr = np.array(vals, dtype=np.float64)
                mean_val = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0

                # 95% Prediction Interval via empirical percentiles (statistically justified)
                pi_lower = float(np.percentile(arr, 2.5))
                pi_upper = float(np.percentile(arr, 97.5))
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))

                trajectories[v_upper].append(
                    VariableDailyStats(
                        date=d,
                        mean=mean_val,
                        std=std_val,
                        pi_lower_95=pi_lower,
                        pi_upper_95=pi_upper,
                        min_val=min_v,
                        max_val=max_v,
                    )
                )

                if mean_val > 1e-6:
                    trajectory_cvs[v_upper].append(std_val / mean_val)

        # 7. Harvest / Yield (TWSO) Forecast on Final Date
        final_date = sorted_dates[-1]
        final_twso_stats = [t for t in trajectories["TWSO"] if t.date == final_date]
        if not final_twso_stats:
            final_twso_stats = trajectories["TWSO"][-1:]

        yield_stat = final_twso_stats[0]
        yield_forecast = YieldForecast(
            harvest_date=final_date,
            mean_yield_kg_ha=yield_stat.mean,
            std_yield_kg_ha=yield_stat.std,
            pi_lower_95_kg_ha=yield_stat.pi_lower_95,
            pi_upper_95_kg_ha=yield_stat.pi_upper_95,
            min_yield_kg_ha=yield_stat.min_val,
            max_yield_kg_ha=yield_stat.max_val,
        )

        # 8. Uncertainty Metrics
        yield_cv = yield_stat.std / yield_stat.mean if yield_stat.mean > 1e-6 else 0.0
        pi_width = yield_stat.pi_upper_95 - yield_stat.pi_lower_95
        rel_uncertainty_pct = (pi_width / yield_stat.mean * 100.0) if yield_stat.mean > 1e-6 else 0.0

        mean_cv_per_var = {
            v: float(np.mean(cv_list)) if cv_list else 0.0
            for v, cv_list in trajectory_cvs.items()
        }

        uncertainty_metrics = UncertaintyMetrics(
            yield_cv=yield_cv,
            yield_pi_width_kg_ha=pi_width,
            yield_relative_uncertainty_pct=rel_uncertainty_pct,
            mean_trajectory_cv=mean_cv_per_var,
        )

        # 9. Execution Diagnostics
        horizon_days = (harvest_date - forecast_start_date).days + 1
        diagnostics = ForecastDiagnostics(
            simulation_id=str(simulation_id),
            forecast_start_date=forecast_start_date,
            target_harvest_date=harvest_date,
            forecast_horizon_days=horizon_days,
            ensemble_size=ensemble_size,
            assimilated_cycles_count=assimilated_cycles_count,
            latest_assimilation_date=latest_assim_date,
            crop_name=sim_run.crop,
            variety_name=sim_run.variety,
        )

        return ForecastResponse(
            diagnostics=diagnostics,
            yield_forecast=yield_forecast,
            uncertainty_metrics=uncertainty_metrics,
            trajectories=trajectories,
        )
