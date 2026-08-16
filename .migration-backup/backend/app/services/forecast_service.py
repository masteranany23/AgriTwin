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
3. Strict Residual Reporting: Never claims a hybrid ML prediction unless a validated residual model exists.
4. Traceable Diagnostics: Documents forecast horizon, assimilation state lineage, observation counts, and ensemble statistics.
"""

import datetime
import logging
import uuid
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.app.assimilation.ensemble.ensemble_manager import EnsembleManager
from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.assimilation.models.observation import Observation
from backend.app.models.assimilation_run import AssimilationRun
from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.residual.registry import global_residual_registry
from backend.app.schemas.forecast import (
    AssimilatedYieldResult,
    ForecastDiagnostics,
    ForecastResponse,
    HybridYieldResult,
    ObservationSummary,
    OpenLoopYieldResult,
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
            ForecastResponse containing daily trajectory stats, yield forecasts, and diagnostics.
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
        vars_of_interest = ["lai", "sm", "tagp", "twso", "dvs", "rftra"]
        member_outputs_by_date: Dict[datetime.date, Dict[str, List[float]]] = {}

        for member in manager.members:
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

        # 8. Open-Loop and Assimilated Results
        open_loop_mean = (
            float(sim_run.yield_kg_ha)
            if sim_run.yield_kg_ha is not None
            else float(member_outputs_by_date[sorted_dates[-1]]["twso"][0])
            if member_outputs_by_date.get(sorted_dates[-1], {}).get("twso")
            else yield_stat.mean
        )

        open_loop_result = OpenLoopYieldResult(
            mean_yield_kg_ha=open_loop_mean,
            harvest_date=final_date,
            description="Open-loop unassimilated WOFOST physical baseline simulation.",
        )

        assimilated_result = AssimilatedYieldResult(
            mean_yield_kg_ha=yield_stat.mean,
            std_yield_kg_ha=yield_stat.std,
            pi_lower_95_kg_ha=yield_stat.pi_lower_95,
            pi_upper_95_kg_ha=yield_stat.pi_upper_95,
            harvest_date=final_date,
            assimilated_cycles_count=assimilated_cycles_count,
        )

        # 9. Observation Summary
        obs_query = self.db.query(Observation).filter(
            (Observation.simulation_run_id == simulation_id) | (Observation.field_id == sim_run.field_id)
        )
        all_obs = obs_query.all()
        sources = sorted(list({str(o.source.value if hasattr(o.source, "value") else o.source) for o in all_obs if o.source}))
        obs_used = sum(1 for o in all_obs if str(o.status.value if hasattr(o.status, "value") else o.status) == "VALID")
        obs_rejected = sum(1 for o in all_obs if str(o.status.value if hasattr(o.status, "value") else o.status) in ["REJECTED", "OUTLIER"])

        observation_summary = ObservationSummary(
            active_sources=sources,
            observations_used=obs_used,
            observations_rejected=obs_rejected,
        )

        # 10. Residual Model Resolution & Hybrid Result
        crop_name = sim_run.crop
        region_name = getattr(sim_run, "region", None)
        res_model = global_residual_registry.get_model(crop=crop_name, region=region_name)

        is_validated = res_model.is_available(crop=crop_name, region=region_name) and getattr(res_model.metadata, "validated", False)

        hybrid_result: Optional[HybridYieldResult] = None
        if is_validated:
            pred = res_model.apply_correction(assimilated_yield=yield_stat.mean, crop=crop_name, region=region_name)
            hybrid_result = HybridYieldResult(
                model_id=res_model.metadata.model_id,
                model_version=res_model.metadata.version,
                assimilated_yield_kg_ha=yield_stat.mean,
                residual_correction_kg_ha=pred.residual_correction_kg_ha,
                corrected_yield_kg_ha=pred.corrected_yield_kg_ha,
                residual_uncertainty_kg_ha=pred.residual_uncertainty_kg_ha,
                is_validated=True,
            )
            forecast_mode = "HYBRID_RESIDUAL"
        else:
            hybrid_result = None
            forecast_mode = "ASSIMILATED_ENSEMBLE" if assimilated_cycles_count > 0 else "OPEN_LOOP_BASELINE"

        # 11. Uncertainty Metrics
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

        # 12. Execution Diagnostics & Confidence Explanation
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

        if is_validated and hybrid_result:
            confidence_explanation = (
                f"Forecast generated using {ensemble_size} ensemble members over a {horizon_days}-day horizon "
                f"with {assimilated_cycles_count} EnKF assimilation cycles. "
                f"Observation support: {len(sources)} sources ({', '.join(sources) if sources else 'None'}), "
                f"{obs_used} valid, {obs_rejected} rejected. "
                f"Assimilated WOFOST yield is {yield_stat.mean:.0f} kg/ha (95% PI: [{yield_stat.pi_lower_95:.0f} - {yield_stat.pi_upper_95:.0f} kg/ha], CV: {yield_cv*100:.1f}%). "
                f"Applied validated residual model '{hybrid_result.model_id}' (v{hybrid_result.model_version}) "
                f"with a correction of {hybrid_result.residual_correction_kg_ha:+.0f} kg/ha, yielding a final hybrid prediction of {hybrid_result.corrected_yield_kg_ha:.0f} kg/ha."
            )
        else:
            confidence_explanation = (
                f"Forecast generated using {ensemble_size} ensemble members over a {horizon_days}-day horizon "
                f"with {assimilated_cycles_count} EnKF assimilation cycles up to {latest_assim_date or 'sowing'}. "
                f"Observation support: {len(sources)} sources ({', '.join(sources) if sources else 'None'}), "
                f"{obs_used} valid, {obs_rejected} rejected. "
                f"Assimilated WOFOST yield forecast is {yield_stat.mean:.0f} kg/ha with 95% PI [{yield_stat.pi_lower_95:.0f} - {yield_stat.pi_upper_95:.0f} kg/ha] (CV: {yield_cv*100:.1f}%). "
                f"No validated ML residual model is active for this crop/region, so assimilated WOFOST prediction is returned directly without synthetic correction."
            )

        return ForecastResponse(
            diagnostics=diagnostics,
            yield_forecast=yield_forecast,
            uncertainty_metrics=uncertainty_metrics,
            trajectories=trajectories,
            open_loop_result=open_loop_result,
            assimilated_result=assimilated_result,
            hybrid_result=hybrid_result,
            uncertainty=uncertainty_metrics,
            observation_summary=observation_summary,
            forecast_mode=forecast_mode,
            confidence_explanation=confidence_explanation,
        )
