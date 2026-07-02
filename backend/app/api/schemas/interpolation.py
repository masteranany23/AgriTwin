"""
Interpolation Schemas
====================

Pydantic models for temporal interpolation API requests and responses.
"""

from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional, Literal


class InterpolationRequest(BaseModel):
    """Request body for temporal interpolation with cloud-gap detection."""
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "observation_dates": ["2020-07-01", "2020-07-10", "2020-07-20"],
                "observation_values": [2.5, 3.2, 4.1],
                "target_dates": ["2020-07-01", "2020-07-02", "2020-07-03"],
                "method": "cubic_spline",
                "max_allowed_gap_days": 10
            }
        }


class InterpolationResponse(BaseModel):
    """Response body for temporal interpolation with monsoon gap handling."""
    interpolated_dates: List[date]
    interpolated_values: List[Optional[float]]  # None values indicate cloud gaps (hold open-loop)
    quality_flags: List[dict]  # e.g., {"date": "2020-07-15", "action": "HOLD_OPEN_LOOP", "reason": "Cloud gap > 10 days"}
    method_used: str
    message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "interpolated_dates": ["2020-07-01", "2020-07-02"],
                "interpolated_values": [2.5, 2.6],
                "quality_flags": [
                    {"date": "2020-07-15", "action": "HOLD_OPEN_LOOP", "reason": "Cloud gap > 10 days"}
                ],
                "method_used": "cubic_spline",
                "message": "Done. 1 large gaps forced to open-loop."
            }
        }
