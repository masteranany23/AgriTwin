"""
Fusion & Interpolation Module Schemas
=====================================

Pydantic models for:
1. Temporal Interpolation (Existing - your code)
2. Spatial Alignment (New - Module 3.3)
3. Confidence Estimation (New - Module 3.3)
4. Multi-source Fusion (New - Module 3.3)
5. Complete Data Fusion Pipeline (New - Module 3.3)

Research Alignment:
- Monsoon Cloud-Gap Trigger (max_allowed_gap_days)
- Observation-uncertainty-dependent R values for EnKF
- Multi-source fusion with cloud-cover weighting
"""

from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Tuple, Literal
from uuid import UUID
from enum import Enum


# ===========================================================================
# ENUMS
# ===========================================================================

class ObservationSource(str, Enum):
    """Source of observation data."""
    SENTINEL2       = "SENTINEL2"
    SENTINEL1_SAR   = "SENTINEL1_SAR"
    MODIS           = "MODIS"
    SMARTPHONE_GRVI = "SMARTPHONE_GRVI"
    SMARTPHONE_RGB  = "SMARTPHONE_RGB"
    ERA5_LAND       = "ERA5_LAND"
    NASA_POWER      = "NASA_POWER"
    FUSED           = "FUSED"
    # General / Heterogeneous observation sources
    SATELLITE       = "SATELLITE"
    SENSOR          = "SENSOR"
    WEATHER         = "WEATHER"
    MANUAL          = "MANUAL"
    MODEL           = "MODEL"
    IOT_SENSOR      = "IOT_SENSOR"
    WEATHER_STATION = "WEATHER_STATION"
    MANUAL_SCOUT    = "MANUAL_SCOUT"


class SpatialResolution(str, Enum):
    """Spatial resolution category."""
    HIGH = "HIGH"       # ~10m (Sentinel-2)
    MEDIUM = "MEDIUM"   # ~100m-1km (MODIS)
    LOW = "LOW"         # ~9km (ERA5-Land)
    POINT = "POINT"     # GPS coordinate (farmer photo)


class QualityFlag(str, Enum):
    """Quality flag for fused observations."""
    HIGH = "HIGH"           # Clear sky, high confidence
    MEDIUM = "MEDIUM"       # Partly cloudy, moderate confidence
    LOW = "LOW"             # Cloudy, low confidence (monsoon)
    HOLD_OPEN_LOOP = "HOLD_OPEN_LOOP"  # No observation, use pure WOFOST
    NO_DATA = "NO_DATA"     # No data available
    INVALID = "INVALID"     # Data failed validation


# ===========================================================================
# TEMPORAL INTERPOLATION SCHEMAS (EXISTING - YOUR CODE)
# ===========================================================================

class InterpolationRequest(BaseModel):
    """
    Request body for temporal interpolation with cloud-gap detection.
    
    RESEARCH FEATURE: Monsoon Cloud-Gap Trigger
    If gap > max_allowed_gap_days, returns None (hold open-loop)
    instead of hallucinating LAI values.
    """
    observation_dates: List[date] = Field(
        ..., 
        description="Dates when satellite data is available"
    )
    observation_values: List[float] = Field(
        ..., 
        description="LAI or SM values at those dates"
    )
    target_dates: List[date] = Field(
        ..., 
        description="All daily dates to fill (e.g., whole season)"
    )
    method: Literal["linear", "cubic_spline", "savgol"] = Field(
        default="cubic_spline",
        description="Interpolation method: linear, cubic_spline, or savgol (Savitzky-Golay)"
    )
    max_allowed_gap_days: int = Field(
        default=10,
        description="Monsoon Cloud-Gap Trigger: If gap > this, hold open-loop instead of interpolating"
    )
    
    @field_validator('observation_dates', 'observation_values')
    def validate_lengths(cls, v, info):
        """Ensure observation_dates and observation_values have same length."""
        if info.field_name == 'observation_dates':
            return v
        # We'll validate in the service instead for simplicity
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation_dates": ["2024-07-01", "2024-07-10", "2024-07-20"],
                "observation_values": [2.5, 3.2, 4.1],
                "target_dates": ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04"],
                "method": "cubic_spline",
                "max_allowed_gap_days": 10
            }
        }


class InterpolationResponse(BaseModel):
    """
    Response body for temporal interpolation with monsoon gap handling.
    
    RESEARCH FEATURE: Quality Flags
    - HOLD_OPEN_LOOP: Gap > max_allowed_gap_days (monsoon)
    - interpolated: Normal interpolation
    - satellite_observation: Actual observation date
    """
    interpolated_dates: List[date]
    interpolated_values: List[Optional[float]]  # None = cloud gap -> hold open-loop
    quality_flags: List[Dict[str, Any]]  # e.g., {"date": "2024-07-15", "action": "HOLD_OPEN_LOOP", "reason": "Cloud gap > 10 days"}
    method_used: str
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "interpolated_dates": ["2024-07-01", "2024-07-02", "2024-07-15"],
                "interpolated_values": [2.5, 2.6, None],
                "quality_flags": [
                    {"date": "2024-07-15", "action": "HOLD_OPEN_LOOP", "reason": "Cloud gap > 10 days"}
                ],
                "method_used": "cubic_spline",
                "message": "Done. 1 large gap forced to open-loop."
            }
        }


# ===========================================================================
# SPATIAL ALIGNMENT SCHEMAS (NEW - MODULE 3.3)
# ===========================================================================

class SpatialAlignmentRequest(BaseModel):
    """
    Request to align observations from different resolutions to a unified field grid.
    
    Why this matters for India:
    - Sentinel-2: 10m (HIGH)
    - ERA5-Land: 9km (LOW) — needs resampling to field level
    - Farmer Photo: GPS point (POINT) — needs snapping to grid
    """
    field_id: UUID = Field(..., description="Field ID to align observations for")
    observations: List[Dict[str, Any]] = Field(
        ..., 
        description="List of raw observations with lat/lon/resolution"
    )
    target_resolution: Optional[float] = Field(
        default=10.0, 
        description="Target resolution in meters (default: 10m = Sentinel-2)"
    )
    field_boundary_geojson: Optional[Dict] = Field(
        default=None, 
        description="Optional GeoJSON polygon of field boundary"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "550e8400-e29b-41d4-a716-446655440000",
                "observations": [
                    {"source": "SENTINEL2", "latitude": 28.5, "longitude": 77.0, "value": 2.5, "resolution": "HIGH"},
                    {"source": "ERA5_LAND", "latitude": 28.5, "longitude": 77.0, "value": 0.35, "resolution": "LOW"},
                    {"source": "SMARTPHONE_GRVI", "latitude": 28.5001, "longitude": 77.0001, "value": 0.45, "resolution": "POINT"}
                ],
                "target_resolution": 10.0
            }
        }


class SpatialAlignmentResponse(BaseModel):
    """Response containing spatially aligned observations."""
    field_id: UUID
    aligned_observations: List[Dict[str, Any]]  # All observations now on the same grid
    grid_metadata: Dict[str, Any] = Field(
        ..., 
        description="resolution, bounds, area, etc."
    )
    message: str


# ===========================================================================
# CONFIDENCE ESTIMATION SCHEMAS (NEW - MODULE 3.3)
# ===========================================================================

class ConfidenceRequest(BaseModel):
    """
    Request to compute confidence for an observation.
    
    RESEARCH FEATURE: Source-specific R values for EnKF
    - Sentinel-2: R=0.10 (high confidence)
    - Farmer Photo: R=0.30 (low confidence - "gentle nudge")
    - ERA5-Land: R=0.15 (moderate confidence)
    """
    source: ObservationSource
    value: float
    cloud_cover: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)
    viewing_angle: Optional[float] = Field(default=0.0, ge=0.0, le=45.0)
    sensor_health: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    days_since_observation: Optional[int] = Field(default=0, ge=0)
    field_id: Optional[UUID] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "source": "SMARTPHONE_GRVI",
                "value": 0.45,
                "cloud_cover": 0,
                "viewing_angle": 0,
                "sensor_health": 1.0,
                "days_since_observation": 0
            }
        }


class ConfidenceResponse(BaseModel):
    """
    Response containing confidence score and observation error (R) for EnKF.
    
    RESEARCH FEATURE: R = Base_R + (1 - Confidence) * (1 - Base_R)
    - If confidence is high (1.0), R = Base_R (e.g., 0.10 for Sentinel-2)
    - If confidence is low (0.0), R approaches 1.0 (complete distrust)
    """
    source: ObservationSource
    original_value: float
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    observation_error_r: float = Field(..., gt=0.0, le=1.0, description="R value for EnKF")
    factors: Dict[str, float] = Field(..., description="Breakdown of factors affecting confidence")
    message: str


# ===========================================================================
# MULTI-SOURCE FUSION SCHEMAS (NEW - MODULE 3.3)
# ===========================================================================

class FusionRequest(BaseModel):
    """
    Request to fuse multiple LAI/SM observations into a single estimate.
    
    RESEARCH FEATURE: Cloud-cover adaptive weighting
    - Clear (<40%): 70% Sentinel-2, 10% SAR, 20% Photo
    - Partly (40-70%): Weighted blend S2 + SAR
    - Cloudy (>70%): 70% SAR, 30% Photo (monsoon mode)
    """
    field_id: UUID
    date: date
    observations: List[Dict[str, Any]] = Field(
        ..., 
        description="List of {source, value, confidence, metadata}"
    )
    cloud_cover: Optional[float] = Field(default=0.0, ge=0.0, le=100.0)
    crop_type: Optional[str] = Field(default="wheat", description="wheat, rice, maize, etc.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "550e8400-e29b-41d4-a716-446655440000",
                "date": "2024-07-15",
                "observations": [
                    {"source": "SENTINEL2", "value": 2.5, "confidence": 0.85},
                    {"source": "SENTINEL1_SAR", "value": 0.6, "confidence": 0.70},
                    {"source": "SMARTPHONE_GRVI", "value": 0.45, "confidence": 0.60}
                ],
                "cloud_cover": 60,
                "crop_type": "rice"
            }
        }


class FusionResponse(BaseModel):
    """Response containing fused LAI and SM estimates."""
    field_id: UUID
    date: date
    fused_lai: float
    fused_sm: Optional[float] = None
    fused_confidence: float = Field(..., ge=0.0, le=1.0)
    source_weights: Dict[str, float] = Field(
        ..., 
        description="e.g., {'SENTINEL2': 0.6, 'SMARTPHONE_GRVI': 0.3, ...}"
    )
    contributing_sources: List[str]
    quality_flag: QualityFlag
    message: str


# ===========================================================================
# COMPLETE DATA FUSION PIPELINE SCHEMAS (NEW - MODULE 3.3)
# ===========================================================================

class DataFusionPipelineRequest(BaseModel):
    """
    Orchestrates the entire Module 3.3 pipeline:
    1. Observation Validation (internal)
    2. Temporal Alignment (your existing service)
    3. Spatial Alignment (new)
    4. Confidence Estimation (new)
    5. Multi-source Fusion (new)
    """
    field_id: UUID
    start_date: date
    end_date: date
    force_source: Optional[ObservationSource] = None
    crop_type: Optional[str] = Field(default="wheat")
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_id": "550e8400-e29b-41d4-a716-446655440000",
                "start_date": "2024-07-01",
                "end_date": "2024-07-31",
                "crop_type": "rice"
            }
        }


class DailyFusedState(BaseModel):
    """Single day's fused state from the pipeline."""
    date: date
    lai: Optional[float] = None
    sm: Optional[float] = None
    confidence: float
    quality_flag: QualityFlag
    sources_used: List[str]
    message: str


class DataFusionPipelineResponse(BaseModel):
    """
    Final output after running Validation -> Temporal -> Spatial -> Confidence -> Fusion.
    """
    field_id: UUID
    daily_fused_states: List[DailyFusedState]
    temporal_coverage: float = Field(
        ..., 
        description="Percentage of days with valid data (0.0 to 1.0)"
    )
    message: str