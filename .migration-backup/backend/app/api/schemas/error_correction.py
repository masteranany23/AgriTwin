"""
Error Correction Schemas
========================

Pydantic models for error correction API requests and responses.
"""

from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional, Literal
from uuid import UUID


class ErrorCorrectionRequest(BaseModel):
    """Request body for 7-day window error correction with joint LAI+SM assimilation."""
    simulation_id: UUID
    field_id: UUID
    window_start_date: date
    window_end_date: date  # Should be 7 days after start
    residual_threshold: float = Field(
        default=0.5, 
        description="LAI threshold for flagging anomalies"
    )
    source: Literal["FARMER_PHOTO", "SENTINEL_2", "MODIS"] = Field(
        default="SENTINEL_2", 
        description="Observation source - affects uncertainty in Kalman Gain"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_id": "123e4567-e89b-12d3-a456-426614174000",
                "field_id": "123e4567-e89b-12d3-a456-426614174001",
                "window_start_date": "2020-07-10",
                "window_end_date": "2020-07-16",
                "residual_threshold": 0.5,
                "source": "SENTINEL_2"
            }
        }


class ErrorCorrectionResponse(BaseModel):
    """Response body for error correction with joint LAI+SM assimilation."""
    simulation_id: UUID
    window_start: date
    window_end: date
    total_days_processed: int
    anomalies_detected: int
    anomalies_corrected: int
    correction_summary: List[dict]  # Details per day (DailyCorrectionRecord)
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "simulation_id": "123e4567-e89b-12d3-a456-426614174000",
                "window_start": "2020-07-10",
                "window_end": "2020-07-16",
                "total_days_processed": 7,
                "anomalies_detected": 3,
                "anomalies_corrected": 3,
                "correction_summary": [],
                "message": "Processed 7 days. Corrected LAI+SM jointly."
            }
        }


class DailyCorrectionRecord(BaseModel):
    """Individual day correction record with adaptive Kalman Gain."""
    date: date
    variable: str  # "LAI", "SM", etc.
    wofost_value: float
    satellite_value: Optional[float] = None  # From interpolation, may be None in cloud gaps
    residual: float = 0.0
    was_anomaly: bool = False
    correction_applied: float = 0.0
    corrected_value: float
    blending_weight: float = 0.0  # The adaptive Kalman Gain (0 = trust model, 1 = trust obs)
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2020-07-15",
                "variable": "LAI",
                "wofost_value": 3.2,
                "satellite_value": 3.8,
                "residual": 0.6,
                "was_anomaly": True,
                "correction_applied": 0.36,
                "corrected_value": 3.56,
                "blending_weight": 0.6
            }
        }
