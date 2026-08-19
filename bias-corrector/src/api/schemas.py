"""
Pydantic schemas for AgriTwin Bias Correction API.
"""
from typing import List, Any
from pydantic import BaseModel, Field



class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    
    state: str = Field(..., description="State name")
    district: str = Field(..., description="District name")
    crop_key: str = Field(..., description="Crop identifier (e.g., 'Wheat_Rabi')")
    year: int = Field(..., description="Year of prediction", ge=2000, le=2030)
    
    wofost_yield: float = Field(..., description="WOFOST simulated yield (kg/ha)", ge=0)
    latitude: float = Field(..., description="Latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Longitude", ge=-180, le=180)
    
    # Optional features (can be 0 if not available)
    lai_mean: float = Field(default=0.0, description="Leaf Area Index mean", ge=0)
    ndvi_mean: float = Field(default=0.0, description="Normalized Difference Vegetation Index mean", ge=0, le=1)
    ndre_mean: float = Field(default=0.0, description="Normalized Difference Red Edge mean", ge=0, le=1)
    rainfall_total: float = Field(default=0.0, description="Total rainfall (mm)", ge=0)
    temperature_mean: float = Field(default=0.0, description="Mean temperature (°C)", ge=-50, le=50)
    soil_moisture_mean: float = Field(default=0.0, description="Mean soil moisture (m³/m³)", ge=0, le=1)
    
    @validator('wofost_yield', 'latitude', 'longitude')
    def validate_required_fields(cls, v, field):
        """Validate required fields."""
        if v is None:
            raise ValueError(f"{field.name} is required")
        return v


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    original_yield: float = Field(..., description="Original WOFOST yield (kg/ha)")
    corrected_yield: float = Field(..., description="Bias-corrected yield (kg/ha)")
    correction_factor: float = Field(..., description="Correction factor applied")
    confidence_interval: List[float] = Field(..., description="95% confidence interval [lower, upper]")
    model_version: str = Field(..., description="Model version used")
    warnings: List[str] = Field(default=[], description="Any warnings from the prediction")
    
    @validator('correction_factor')
    def validate_correction_factor(cls, v):
        """Validate correction factor."""
        if v <= 0:
            raise ValueError("Correction factor must be positive")
        return v
    
    @validator('confidence_interval')
    def validate_confidence_interval(cls, v):
        """Validate confidence interval."""
        if len(v) != 2:
            raise ValueError("Confidence interval must have 2 values")
        lower, upper = v
        if lower > upper:
            raise ValueError("Lower bound must be <= upper bound")
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        """Validate batch size."""
        if len(v) == 0:
            raise ValueError("At least one prediction required")
        if len(v) > 1000:
            raise ValueError("Maximum batch size is 1000 predictions")
        return v


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total: int = Field(..., description="Total number of predictions")
    successful: int = Field(..., description="Number of successful predictions")
    failed: int = Field(..., description="Number of failed predictions")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    
    @validator('successful', 'failed')
    def validate_counts(cls, v, values):
        """Validate that successful + failed equals total."""
        if 'total' in values and v > values['total']:
            raise ValueError("Cannot have more successes/failures than total predictions")
        return v


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(..., description="Service status: 'healthy', 'degraded', 'unhealthy'")
    version: str = Field(..., description="Model version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Model type: 'ensemble' or 'ensemble_gp'")
    ics_ratios_loaded: bool = Field(..., description="Whether ICS ratios are loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    
    @validator('status')
    def validate_status(cls, v):
        """Validate status value."""
        valid_statuses = ['healthy', 'degraded', 'unhealthy']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v
