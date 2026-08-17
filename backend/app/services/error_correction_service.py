"""
Error Correction Service (Refactored & Deprecated Direct Mutation)
===================================================================

Service for evaluating residual errors between WOFOST model outputs and observations.

CANONICAL ARCHITECTURE NOTE:
The primary state-estimation pipeline is:
    observations → QualityControlService → DataFusionPipeline → EnKF → assimilated WOFOST.

This service is refactored to serve solely as a diagnostic/residual calculator for backward
compatibility. It NEVER mutates DailyOutput database records directly.
"""

import numpy as np
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
import logging

from backend.app.models.daily_output import DailyOutput
from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.services.quality_control_service import QualityControlService
from backend.app.api.schemas.error_correction import (
    ErrorCorrectionRequest,
    ErrorCorrectionResponse,
    DailyCorrectionRecord
)

logger = logging.getLogger(__name__)


class ErrorCorrectionService:
    """
    Diagnostic service for residual calculation between WOFOST outputs and observations.
    
    DEPRECATION WARNING:
    Direct DB mutation of DailyOutput is disabled. State estimation must be performed
    via the canonical EnKF pipeline (AssimilationService).
    """
    
    def __init__(self, db_session: Session, qc_service: Optional[QualityControlService] = None):
        self.db = db_session
        self.qc_service = qc_service or QualityControlService()
        self.MODEL_UNCERTAINTY = 0.15  
        
    def _compute_kalman_gain(self, obs_uncertainty: float) -> float:
        """Optimal diagnostic Kalman Gain calculation: Gain = Model_Err² / (Model_Err² + Obs_Err²)."""
        prior_variance = self.MODEL_UNCERTAINTY ** 2
        obs_variance = obs_uncertainty ** 2
        return prior_variance / (prior_variance + obs_variance)

    def _get_qc_filtered_observations(
        self, 
        field_id: UUID, 
        start_date: date, 
        end_date: date, 
        variable: str = "LAI"
    ) -> Dict[date, float]:
        """Fetch interpolated observation data, applying QualityControlService gates."""
        from backend.app.assimilation.repositories.observation_repository import ObservationRepository
        from backend.app.api.schemas.interpolation import InterpolationRequest
        from backend.app.services.temporal_interpolation_service import TemporalInterpolationService

        obs_repo = ObservationRepository(self.db)
        raw_observations = obs_repo.get_by_variable(variable_name=variable, field_id=field_id, status=None)
        
        # Pass observations through QualityControlService filter
        valid_observations = []
        for obs in raw_observations:
            qc_res = self.qc_service.evaluate_observation(obs)
            if qc_res.status == ObservationStatus.VALID:
                valid_observations.append(obs)

        if len(valid_observations) < 2:
            return {}

        obs_dates = [o.timestamp.date() for o in valid_observations]
        obs_values = [o.value for o in valid_observations]
        
        target_dates = []
        curr = start_date
        while curr <= end_date:
            target_dates.append(curr)
            curr += timedelta(days=1)

        req = InterpolationRequest(
            observation_dates=obs_dates,
            observation_values=obs_values,
            target_dates=target_dates,
            method="cubic_spline",
            max_allowed_gap_days=15
        )
        
        resp = TemporalInterpolationService().interpolate(req)
        
        result = {}
        for d, val in zip(target_dates, resp.interpolated_values):
            if val is not None and not np.isnan(val):
                result[d] = float(val)
        return result

    def _get_daily_outputs(self, simulation_id: UUID, start_date: date, end_date: date) -> Dict[date, dict]:
        """Fetch WOFOST daily state outputs for the window."""
        outputs = self.db.query(DailyOutput).filter(
            DailyOutput.simulation_run_id == simulation_id,
            DailyOutput.date >= start_date,
            DailyOutput.date <= end_date
        ).order_by(DailyOutput.date).all()
        
        return {
            out.date: {
                "lai": out.lai or 0.0,
                "sm": out.sm or 0.0,
                "dvs": out.dvs or 0.0,
                "tagp": out.tagp or 0.0,
            }
            for out in outputs
        }

    def correct_window(self, request: ErrorCorrectionRequest) -> ErrorCorrectionResponse:
        """
        Diagnostic entry point: Computes residual analysis and recommended gain
        WITHOUT mutating DailyOutput database records.
        """
        window_days = (request.window_end_date - request.window_start_date).days
        if window_days != 6:
            return ErrorCorrectionResponse(
                simulation_id=request.simulation_id,
                window_start=request.window_start_date,
                window_end=request.window_end_date,
                total_days_processed=0,
                anomalies_detected=0,
                anomalies_corrected=0,
                correction_summary=[],
                message=f"Window must be 7 days. Got {window_days+1}."
            )

        wofost_data = self._get_daily_outputs(
            request.simulation_id, 
            request.window_start_date, 
            request.window_end_date
        )
        
        if not wofost_data:
            return ErrorCorrectionResponse(
                simulation_id=request.simulation_id,
                window_start=request.window_start_date,
                window_end=request.window_end_date,
                total_days_processed=0,
                anomalies_detected=0,
                anomalies_corrected=0,
                correction_summary=[],
                message="No WOFOST data found."
            )

        sat_lai_data = self._get_qc_filtered_observations(
            request.field_id, 
            request.window_start_date, 
            request.window_end_date, 
            "LAI"
        )
        sat_sm_data = self._get_qc_filtered_observations(
            request.field_id, 
            request.window_start_date, 
            request.window_end_date, 
            "SM"
        )

        corrections = []
        anomalies_count = 0
        corrected_count = 0
        current_date = request.window_start_date

        while current_date <= request.window_end_date:
            wofost_state = wofost_data.get(current_date)
            if not wofost_state:
                current_date += timedelta(days=1)
                continue

            sat_lai = sat_lai_data.get(current_date)
            residual = 0.0
            is_anomaly = False
            gain = 0.0
            corrected_lai = wofost_state["lai"]
            
            if sat_lai is not None:
                wofost_lai = wofost_state["lai"]
                residual = sat_lai - wofost_lai
                is_anomaly = abs(residual) > request.residual_threshold

                if is_anomaly:
                    anomalies_count += 1
                    if request.source == "FARMER_PHOTO":
                        obs_uncertainty = 0.3
                    elif request.source == "MODIS":
                        obs_uncertainty = 0.2
                    else:  # SENTINEL_2
                        obs_uncertainty = 0.1
                    
                    gain = self._compute_kalman_gain(obs_uncertainty)
                    corrected_lai = wofost_lai + gain * residual
                    corrected_count += 1
                    # NOTE: Direct DB update removed to enforce canonical QC -> Fusion -> EnKF pipeline

            corrections.append(
                DailyCorrectionRecord(
                    date=current_date,
                    variable="LAI",
                    wofost_value=wofost_state["lai"],
                    satellite_value=sat_lai,
                    residual=residual,
                    was_anomaly=is_anomaly,
                    correction_applied=(corrected_lai - wofost_state["lai"]),
                    corrected_value=corrected_lai,
                    blending_weight=gain
                ).model_dump()
            )
            current_date += timedelta(days=1)

        logger.info(
            "correct_window diagnostic evaluated for sim=%s field=%s: %d anomalies detected. DailyOutput unchanged.",
            request.simulation_id, request.field_id, anomalies_count
        )

        return ErrorCorrectionResponse(
            simulation_id=request.simulation_id,
            window_start=request.window_start_date,
            window_end=request.window_end_date,
            total_days_processed=len(corrections),
            anomalies_detected=anomalies_count,
            anomalies_corrected=corrected_count,
            correction_summary=corrections,
            message=(
                "Deprecated: /error-correction/correct-window evaluates diagnostic residuals via QualityControlService "
                "without mutating DailyOutput. Use /assimilation/run-season for canonical EnKF state assimilation."
            )
        )
