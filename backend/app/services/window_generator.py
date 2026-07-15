"""
Window Generator Service
========================

Generates ML training windows from WOFOST outputs with ERA5-Land features.

Research Features:
- STEP 4: Heat Strain Hours (hours with temp > 34°C)
- STEP 5: 4-Layer Soil Moisture from ERA5-Land
- STEP 6: DVS Phase Grouping & Relative Error
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
import logging

from backend.app.models.daily_output import DailyOutput
from backend.app.models.simulation_run import SimulationRun
from backend.app.api.schemas.window_preprocessing import WindowGenerationResponse

logger = logging.getLogger(__name__)


class WindowGenerator:
    """
    Service for generating ML training windows from WOFOST outputs.
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session

    def _fetch_daily_data(self, simulation_id: UUID) -> pd.DataFrame:
        """Fetch WOFOST + ERA5-Land (4-layer SM) + Heat Strain + Satellite LAI."""
        sim_run = self.db.query(SimulationRun).filter(
            SimulationRun.id == simulation_id
        ).first()
        
        if not sim_run:
            return pd.DataFrame()
            
        outputs = self.db.query(DailyOutput).filter(
            DailyOutput.simulation_run_id == simulation_id
        ).order_by(DailyOutput.date).all()
        
        if not outputs:
            return pd.DataFrame()

        target_dates = [out.date for out in outputs]
        start_date = target_dates[0] - timedelta(days=14)
        end_date = target_dates[-1]

        # Initialize ERA5-Land weather provider
        from backend.app.services.weather_service import WeatherService
        weather_service = WeatherService()
        
        try:
            wdp = weather_service.get_weather_provider(
                latitude=sim_run.latitude,
                longitude=sim_run.longitude,
                start_date=start_date,
                end_date=end_date,
                source="ERA5_LAND"
            )
        except Exception as e:
            logger.error(f"ERA5-Land fetch failed: {e}. Using fallback values.")
            wdp = None

        # Fetch interpolated satellite LAI for comparison
        satellite_lai_dict = self._fetch_interpolated_satellite_lai(
            sim_run.field_id, 
            target_dates
        )

        data = []
        for out in outputs:
            if wdp:
                weather = self._get_weather_for_date(wdp, out.date)
            else:
                weather = {
                    "temp_max": 25.0, "temp_avg": 25.0, "precip": 0.0, 
                    "radiation": 20.0, "heat_strain_hours": 0,
                    "sm_layer_1": 0.3, "sm_layer_2": 0.3, 
                    "sm_layer_3": 0.3, "sm_layer_4": 0.3
                }
            
            # Get satellite LAI and calculate residual
            wofost_lai = out.lai or 0.0
            sat_lai = satellite_lai_dict.get(out.date, wofost_lai)  # Fallback to WOFOST
            residual = sat_lai - wofost_lai
            
            data.append({
                "date": out.date,
                # WOFOST outputs
                "LAI": wofost_lai,
                "SM": out.sm or 0.0,
                "DVS": out.dvs or 0.0,
                "TAGP": out.tagp or 0.0,
                "TWSO": out.twso or 0.0,
                # Weather extremes
                "TEMP_MAX": weather.get("temp_max", 25.0),
                "TEMP_AVG": weather.get("temp_avg", 25.0),
                "PRECIP": weather.get("precip", 0.0),
                "RADIATION": weather.get("radiation", 20.0),
                "HEAT_STRAIN_HOURS": weather.get("heat_strain_hours", 0),
                # 4-Layer Soil Moisture
                "SM_LAYER_1": weather.get("sm_layer_1", 0.3),
                "SM_LAYER_2": weather.get("sm_layer_2", 0.3),
                "SM_LAYER_3": weather.get("sm_layer_3", 0.3),
                "SM_LAYER_4": weather.get("sm_layer_4", 0.3),
                # Satellite comparison (RESEARCH STEP 6)
                "SAT_LAI": sat_lai,
                "RESIDUAL": residual,
            })
        
        return pd.DataFrame(data).set_index("date")

    def _fetch_interpolated_satellite_lai(
        self, 
        field_id: Optional[UUID], 
        target_dates: List[date]
    ) -> Dict[date, float]:
        """
        Fetch and interpolate satellite LAI observations for residual calculation.
        
        Args:
            field_id: Field UUID for observation lookup
            target_dates: List of dates to interpolate for
            
        Returns:
            Dictionary mapping date to interpolated LAI value
        """
        if field_id is None:
            return {}
        
        try:
            from backend.app.services.temporal_interpolation_service import TemporalInterpolationService
            from backend.app.assimilation.repositories.observation_repository import ObservationRepository
            from backend.app.api.schemas.interpolation import InterpolationRequest
            
            # Fetch LAI observations from DB
            obs_repo = ObservationRepository(self.db)
            observations = obs_repo.get_by_variable(
                variable_name="LAI", 
                field_id=field_id
            )
            
            # Need at least 2 observations for interpolation
            if len(observations) < 2:
                logger.info(f"Insufficient LAI observations ({len(observations)}) for interpolation. Skipping residual calculation.")
                return {}
            
            # Extract dates and values
            obs_dates = [o.timestamp.date() for o in observations]
            obs_values = [o.value for o in observations]
            
            # Create interpolation request
            request = InterpolationRequest(
                observation_dates=obs_dates,
                observation_values=obs_values,
                target_dates=target_dates,
                method="cubic_spline",
                max_allowed_gap_days=15  # Allow slightly longer gaps for residual calc
            )
            
            # Perform interpolation
            interp_service = TemporalInterpolationService()
            response = interp_service.interpolate(request)
            
            # Build date-to-value dictionary
            result = {}
            for date_val, lai_val in zip(response.interpolated_dates, response.interpolated_values):
                if lai_val is not None:  # Skip None values (cloud gaps)
                    result[date_val] = lai_val
            
            logger.info(f"Interpolated satellite LAI for {len(result)}/{len(target_dates)} dates")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to fetch/interpolate satellite LAI: {e}")
            return {}

    def _get_weather_for_date(self, wdp, date_obj: date) -> Dict:
        """Fetch ERA5-Land data for a specific date."""
        try:
            wdata = wdp(date_obj)
            hourly_temps = getattr(wdata, 'hourly_temp', [25]*24)
            
            # RESEARCH STEP 4: Count hours above 34°C
            heat_strain_hours = sum(1 for t in hourly_temps if t > 34.0)
            
            return {
                "temp_max": wdata.TMAX,
                "temp_avg": wdata.TEMP,
                "precip": wdata.RAIN * 10.0 if wdata.RAIN else 0.0,
                "radiation": wdata.IRRAD / 1e6 if wdata.IRRAD else 20.0,
                "heat_strain_hours": heat_strain_hours,
                "sm_layer_1": getattr(wdata, 'sm_0_7cm', 0.3),
                "sm_layer_2": getattr(wdata, 'sm_7_28cm', 0.3),
                "sm_layer_3": getattr(wdata, 'sm_28_100cm', 0.3),
                "sm_layer_4": getattr(wdata, 'sm_100_289cm', 0.3),
            }
        except Exception as e:
            logger.warning(f"Weather fetch failed for {date_obj}: {e}")
            return {
                "temp_max": 25.0, "temp_avg": 25.0, "precip": 0.0, 
                "radiation": 20.0, "heat_strain_hours": 0, 
                "sm_layer_1": 0.3, "sm_layer_2": 0.3, 
                "sm_layer_3": 0.3, "sm_layer_4": 0.3
            }

    def generate_windows(self, request) -> WindowGenerationResponse:
        """Generate sliding windows with ML features."""
        df = self._fetch_daily_data(request.simulation_id)
        
        if df.empty:
            return WindowGenerationResponse(
                simulation_id=request.simulation_id,
                total_windows_generated=0,
                features_used=[],
                normalization_scalers={},
                start_date=date.today(),
                end_date=date.today(),
                message="No data found for simulation."
            )

        dynamic_features = [
            "LAI", "SM", "DVS", "TAGP", "TWSO",
            "TEMP_MAX", "TEMP_AVG", "PRECIP", "RADIATION", "HEAT_STRAIN_HOURS",
            "SM_LAYER_1", "SM_LAYER_2", "SM_LAYER_3", "SM_LAYER_4"
        ]
        
        feature_matrix = df[dynamic_features].values
        dates = list(df.index)
        n_days = len(df)
        window_size = request.window_size
        stride = request.stride

        windows = []
        for start_idx in range(0, n_days - window_size - 1, stride):
            end_idx = start_idx + window_size - 1
            target_idx = end_idx + 1
            
            window_data = feature_matrix[start_idx:end_idx+1]
            target_dvs = df.iloc[target_idx]["DVS"]
            target_lai = df.iloc[target_idx]["LAI"]
            target_twso = df.iloc[target_idx]["TWSO"]
            
            # RESEARCH STEP 6: DVS Phase grouping
            if target_dvs < 0.5:
                phase = "VEGETATIVE"
            elif target_dvs < 1.0:
                phase = "REPRODUCTIVE"
            else:
                phase = "GRAIN_FILL"

            lai_residual = df.iloc[target_idx].get("RESIDUAL", 0) if "RESIDUAL" in df.columns else 0
            
            flattened = window_data.flatten().tolist()
            
            windows.append({
                "start_date": dates[start_idx],
                "end_date": dates[end_idx],
                "target_date": dates[target_idx],
                "flattened_features": flattened,
                "target_lai": target_lai,
                "target_yield": target_twso,
                "target_phase": phase,
                "dvs_value": target_dvs,
                "relative_error": lai_residual / (target_lai + 0.01),
            })

        # Normalize
        scalers = {}
        if request.normalize and windows:
            mat = np.array([w["flattened_features"] for w in windows])
            for i in range(mat.shape[1]):
                min_v, max_v = np.min(mat[:, i]), np.max(mat[:, i])
                if max_v - min_v > 0:
                    mat[:, i] = (mat[:, i] - min_v) / (max_v - min_v)
                    scalers[f"feature_{i}"] = {"min": float(min_v), "max": float(max_v)}
            for idx, w in enumerate(windows):
                w["flattened_features"] = mat[idx].tolist()

        return WindowGenerationResponse(
            simulation_id=request.simulation_id,
            total_windows_generated=len(windows),
            features_used=dynamic_features,
            normalization_scalers=scalers,
            start_date=dates[0],
            end_date=dates[-1],
            message=f"Generated {len(windows)} windows with Phase labels and 4-layer SM."
        )
