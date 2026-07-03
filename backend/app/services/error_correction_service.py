"""
Error Correction Service
========================

Service for correcting WOFOST model outputs using adaptive Kalman Gain.

Research Features:
- STEP 1: Adaptive Kalman Gain based on observation uncertainty
- STEP 2: Joint LAI + Soil Moisture assimilation
"""

import numpy as np
from datetime import date, timedelta
from typing import Dict, Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
import logging

from backend.app.models.daily_output import DailyOutput
from backend.app.api.schemas.error_correction import (
    ErrorCorrectionRequest,
    ErrorCorrectionResponse,
    DailyCorrectionRecord
)

logger = logging.getLogger(__name__)


class ErrorCorrectionService:
    """
    Service for error correction using adaptive Kalman Gain.
    
    Corrects WOFOST model outputs by blending with satellite observations
    using observation-uncertainty-dependent Kalman Gains.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        # Research-backed uncertainty parameters
        # WOFOST model uncertainty is generally ~15%
        self.MODEL_UNCERTAINTY = 0.15  
        
    def _compute_kalman_gain(self, obs_uncertainty: float) -> float:
        """
        RESEARCH STEP 1: Optimal Kalman Gain (instead of heuristic 0.8/0.2).
        Gain = Model_Error² / (Model_Error² + Observation_Error²)
        
        If observation is bad (R=0.3) -> Gain is low (0.2) -> Trust model.
        If observation is good (R=0.1) -> Gain is high (0.6) -> Trust satellite.
        """
        prior_variance = self.MODEL_UNCERTAINTY ** 2
        obs_variance = obs_uncertainty ** 2
        return prior_variance / (prior_variance + obs_variance)

    def _get_interpolated_observations(
        self, 
        field_id: UUID, 
        start_date: date, 
        end_date: date, 
        variable: str = "LAI"
    ) -> Dict[date, float]:
        """Fetch interpolated satellite/ERA5 data for a specific variable."""
        from backend.app.assimilation.repositories.observation_repository import ObservationRepository
        from backend.app.api.schemas.interpolation import InterpolationRequest
        from backend.app.services.temporal_interpolation_service import TemporalInterpolationService

        obs_repo = ObservationRepository(self.db)
        observations = obs_repo.get_by_variable(variable_name=variable, field_id=field_id)
        
        if len(observations) < 2:
            return {}

        obs_dates = [o.timestamp.date() for o in observations]
        obs_values = [o.value for o in observations]
        
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
        """Fetch WOFOST states for the window."""
        outputs = self.db.query(DailyOutput).filter(
            DailyOutput.simulation_run_id == simulation_id,
            DailyOutput.date >= start_date,
            DailyOutput.date <= end_date
        ).order_by(DailyOutput.date).all()
        
        return {
            out.date: {
                "lai": out.lai or 0.0,
                "sm": out.sm or 0.0,  # RESEARCH STEP 2: We need SM for joint assimilation
                "dvs": out.dvs or 0.0,
                "tagp": out.tagp or 0.0,
            }
            for out in outputs
        }

    def correct_window(self, request: ErrorCorrectionRequest) -> ErrorCorrectionResponse:
        """
        Main entry point: Corrects both LAI and SM using adaptive Kalman Gain.
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

        # RESEARCH STEP 2: Fetch BOTH LAI and SM observations
        sat_lai_data = self._get_interpolated_observations(
            request.field_id, 
            request.window_start_date, 
            request.window_end_date, 
            "LAI"
        )
        sat_sm_data = self._get_interpolated_observations(
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

            # --- PROCESS LAI ---
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
                    # RESEARCH STEP 1: Adaptive Kalman Gain
                    if request.source == "FARMER_PHOTO":
                        obs_uncertainty = 0.3
                    elif request.source == "MODIS":
                        obs_uncertainty = 0.2
                    else:  # SENTINEL_2
                        obs_uncertainty = 0.1
                    
                    gain = self._compute_kalman_gain(obs_uncertainty)
                    
                    # Correction: x_new = x_old + K * (y - x_old)
                    corrected_lai = wofost_lai + gain * residual
                    corrected_count += 1
                    
                    # Update DB
                    self._update_database(request.simulation_id, current_date, "lai", corrected_lai)

            # --- PROCESS SOIL MOISTURE (RESEARCH STEP 2: Joint Assimilation) ---
            sat_sm = sat_sm_data.get(current_date)
            if sat_sm is not None:
                wofost_sm = wofost_state["sm"]
                residual_sm = sat_sm - wofost_sm
                # Use higher uncertainty for SM (ERA5-Land is reanalysis, ~20% error)
                gain_sm = self._compute_kalman_gain(0.2) 
                if abs(residual_sm) > 0.05:  # SM threshold
                    corrected_sm = wofost_sm + gain_sm * residual_sm
                    self._update_database(request.simulation_id, current_date, "sm", corrected_sm)

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

        return ErrorCorrectionResponse(
            simulation_id=request.simulation_id,
            window_start=request.window_start_date,
            window_end=request.window_end_date,
            total_days_processed=len(corrections),
            anomalies_detected=anomalies_count,
            anomalies_corrected=corrected_count,
            correction_summary=corrections,
            message=f"Processed {len(corrections)} days. Corrected LAI+SM jointly."
        )

    def _update_database(self, simulation_id: UUID, date_obj: date, variable: str, value: float):
        """Update DailyOutput with corrected value."""
        daily_output = self.db.query(DailyOutput).filter(
            DailyOutput.simulation_run_id == simulation_id,
            DailyOutput.date == date_obj
        ).first()
        if daily_output:
            setattr(daily_output, variable.lower(), value)
            self.db.commit()
