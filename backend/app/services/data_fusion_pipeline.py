"""
Data Fusion Pipeline
====================

Orchestrates the complete Module 3.3 pipeline:
1. Observation Validation (internal - physical bounds)
2. Temporal Alignment (your existing TemporalInterpolationService)
3. Spatial Alignment (SpatialAlignmentService)
4. Confidence Estimation (ConfidenceEstimator)
5. Multi-source Fusion (MultiSourceFusionService)

RESEARCH ALIGNMENT:
- Option C: Sentinel-2 fails during monsoon (>70% cloud cover)
- Option D: Farmer photos (GRVI) provide reality check (30% weight, R=0.3)
- Option E: ERA5-Land provides 4-layer soil moisture
- EnKF: Uses confidence-derived R values for optimal Kalman Gain
"""

import logging

from datetime import date, timedelta
from uuid import UUID
from sqlalchemy.orm import Session

from backend.app.services.temporal_interpolation_service import TemporalInterpolationService
from backend.app.services.spatial_alignment_service import SpatialAlignmentService
from backend.app.services.confidence_estimator import ConfidenceEstimator
from backend.app.services.multi_source_fusion_service import MultiSourceFusionService
from backend.app.services.quality_control_service import QualityControlService
from backend.app.assimilation.repositories.observation_repository import ObservationRepository
from backend.app.api.schemas.fusion import (
    DataFusionPipelineRequest,
    DataFusionPipelineResponse,
    DailyFusedState,
    FusionRequest,
    ConfidenceRequest,
    SpatialAlignmentRequest,
    InterpolationRequest,
    ObservationSource,
    QualityFlag
)

logger = logging.getLogger(__name__)


class DataFusionPipeline:
    """
    Orchestrates the complete data fusion workflow.
    
    This service ties together all Module 3.3 submodules:
    - QualityControl (centralized service)
    - TemporalAlignment (your existing service)
    - SpatialAlignment (new)
    - ConfidenceEstimation (new)
    - MultiSourceFusion (new)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.temporal = TemporalInterpolationService(db)
        self.spatial = SpatialAlignmentService(db)
        self.confidence = ConfidenceEstimator()
        self.fusion = MultiSourceFusionService(db)
        self.obs_repo = ObservationRepository(db)
        self.qc = QualityControlService()
    
    def process_daily(self, field_id: UUID, target_date: date, crop_type: str = "wheat") -> DailyFusedState:
        """
        Process a single day through the entire Module 3.3 pipeline.
        
        Pipeline Flow:
        1. Fetch raw observations for this day (Module 3.2)
        2. Observation Validation via QualityControlService (physical bounds, outlier rejection)
        3. Temporal Alignment (interpolate if needed - your service)
        4. Spatial Alignment (resample to field grid)
        5. Confidence Estimation (compute R for EnKF)
        6. Multi-source Fusion (combine into single LAI/SM)
        """
        
        # ============================================================
        # STEP 1: Fetch Raw Observations (Module 3.2)
        # ============================================================
        raw_obs = self.obs_repo.get_by_date(field_id, target_date)
        
        if not raw_obs:
            return DailyFusedState(
                date=target_date,
                lai=None,
                sm=None,
                confidence=0.0,
                quality_flag=QualityFlag.NO_DATA,
                sources_used=[],
                message="No raw observations for this date."
            )
        
        # ============================================================
        # STEP 2: Observation Validation (QualityControlService)
        # ============================================================
        validated = []
        for obs in raw_obs:
            res = self.qc.evaluate_observation(obs)
            if res.is_valid:
                validated.append(obs)
            else:
                var_name = getattr(obs, "variable_name", None) or getattr(obs, "variable", "")
                val = getattr(obs, "value", None)
                logger.warning(
                    f"Observation {getattr(obs, 'id', obs)} failed validation. "
                    f"Status: {res.status.value}, Reason: {res.reason}, "
                    f"Variable: {var_name}, Value: {val}"
                )
        
        if not validated:
            return DailyFusedState(
                date=target_date,
                lai=None,
                sm=None,
                confidence=0.0,
                quality_flag=QualityFlag.INVALID,
                sources_used=[],
                message="All observations failed validation (quality control / physical bounds)."
            )
        
        # ============================================================
        # STEP 3: Temporal Alignment (Your Existing Service)
        # ============================================================
        # Get 7-day window for interpolation
        start_window = target_date - timedelta(days=7)
        end_window = target_date + timedelta(days=7)
        
        # Fetch all observations in the window
        window_obs = self.obs_repo.get_by_date_range(field_id, start_window, end_window)
        
        # Separate LAI and SM observations
        lai_obs = [o for o in window_obs if o.variable == "LAI"]
        sm_obs = [o for o in window_obs if o.variable == "SM"]
        
        # Interpolate LAI
        lai_interpolated = None
        if len(lai_obs) >= 2:
            lai_req = InterpolationRequest(
                observation_dates=[o.timestamp.date() for o in lai_obs],
                observation_values=[o.value for o in lai_obs],
                target_dates=[target_date],
                method="cubic_spline",
                max_allowed_gap_days=10  # Monsoon trigger
            )
            lai_resp = self.temporal.interpolate(lai_req)
            if lai_resp.interpolated_values and lai_resp.interpolated_values[0] is not None:
                lai_interpolated = lai_resp.interpolated_values[0]
            elif lai_resp.quality_flags and "HOLD_OPEN_LOOP" in str(lai_resp.quality_flags):
                # Monsoon gap: hold open-loop (use WOFOST later)
                pass
        
        # Interpolate SM
        sm_interpolated = None
        if len(sm_obs) >= 2:
            sm_req = InterpolationRequest(
                observation_dates=[o.timestamp.date() for o in sm_obs],
                observation_values=[o.value for o in sm_obs],
                target_dates=[target_date],
                method="cubic_spline",
                max_allowed_gap_days=10
            )
            sm_resp = self.temporal.interpolate(sm_req)
            if sm_resp.interpolated_values and sm_resp.interpolated_values[0] is not None:
                sm_interpolated = sm_resp.interpolated_values[0]
        
        # Build observation list for spatial alignment
        obs_for_spatial = []
        if lai_interpolated is not None:
            obs_for_spatial.append({
                "source": "SENTINEL2",
                "latitude": getattr(raw_obs[0], 'latitude', 28.5),
                "longitude": getattr(raw_obs[0], 'longitude', 77.0),
                "value": lai_interpolated,
                "variable": "LAI",
                "resolution": "HIGH"
            })
        if sm_interpolated is not None:
            obs_for_spatial.append({
                "source": "ERA5_LAND",
                "latitude": getattr(raw_obs[0], 'latitude', 28.5),
                "longitude": getattr(raw_obs[0], 'longitude', 77.0),
                "value": sm_interpolated,
                "variable": "SM",
                "resolution": "LOW"
            })
        
        # ============================================================
        # STEP 4: Spatial Alignment (NEW)
        # ============================================================
        if obs_for_spatial:
            spatial_req = SpatialAlignmentRequest(
                field_id=field_id,
                observations=obs_for_spatial,
                target_resolution=10.0
            )
            spatial_resp = self.spatial.align_observations(spatial_req)
            aligned_obs = spatial_resp.aligned_observations
        else:
            aligned_obs = []
        
        # ============================================================
        # STEP 5: Confidence Estimation (NEW)
        # ============================================================
        # Build confidence requests for each aligned observation
        confidence_results = []
        for obs in aligned_obs:
            source = ObservationSource(obs.get("source", "SENTINEL2"))
            conf_req = ConfidenceRequest(
                source=source,
                value=obs.get("value", 0.0),
                cloud_cover=0.0,  # Would fetch from weather service
                viewing_angle=obs.get("viewing_angle", 0),
                sensor_health=1.0,
                days_since_observation=0,
                field_id=field_id
            )
            conf_resp = self.confidence.compute_confidence(conf_req)
            confidence_results.append({
                "source": source.value,
                "value": obs.get("value", 0.0),
                "confidence": conf_resp.confidence_score,
                "observation_error_r": conf_resp.observation_error_r,
                "factors": conf_resp.factors
            })
        
        # ============================================================
        # STEP 6: Multi-source Fusion (NEW)
        # ============================================================
        if confidence_results:
            # Get cloud cover for fusion weighting
            cloud_cover = self._get_cloud_cover(field_id, target_date)
            
            fusion_req = FusionRequest(
                field_id=field_id,
                date=target_date,
                observations=confidence_results,
                cloud_cover=cloud_cover,
                crop_type=crop_type
            )
            fusion_resp = self.fusion.fuse_lai(fusion_req)
            
            # Build response
            return DailyFusedState(
                date=target_date,
                lai=fusion_resp.fused_lai,
                sm=None,  # Would need SM fusion separately
                confidence=fusion_resp.fused_confidence,
                quality_flag=QualityFlag(fusion_resp.quality_flag),
                sources_used=fusion_resp.contributing_sources,
                message=fusion_resp.message
            )
        else:
            return DailyFusedState(
                date=target_date,
                lai=None,
                sm=None,
                confidence=0.0,
                quality_flag=QualityFlag.NO_DATA,
                sources_used=[],
                message="No observations after spatial alignment."
            )
    
    def _get_cloud_cover(self, field_id: UUID, target_date: date) -> float:
        """
        Fetch cloud cover for the field on the given date.
        
        In production, this would call a weather API or GEE.
        """
        # TODO: Integrate with weather_service or GEE
        # For now, return mock value
        return 30.0  # 30% cloud cover
    
    def process_range(self, request: DataFusionPipelineRequest) -> DataFusionPipelineResponse:
        """
        Process a date range through the entire Module 3.3 pipeline.
        """
        results = []
        current = request.start_date
        
        while current <= request.end_date:
            day_result = self.process_daily(
                request.field_id, 
                current,
                request.crop_type or "wheat"
            )
            results.append(day_result)
            current += timedelta(days=1)
        
        # Calculate temporal coverage
        valid_days = sum(1 for r in results if r.lai is not None)
        total_days = len(results)
        coverage = valid_days / total_days if total_days > 0 else 0
        
        return DataFusionPipelineResponse(
            field_id=request.field_id,
            daily_fused_states=results,
            temporal_coverage=round(coverage, 2),
            message=(
                f"Processed {total_days} days. "
                f"{valid_days} days had valid data ({coverage*100:.0f}% coverage)."
            )
        )
