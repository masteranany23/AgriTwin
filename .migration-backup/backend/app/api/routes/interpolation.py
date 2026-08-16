"""
Interpolation Routes
===================

FastAPI routes for temporal interpolation endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from backend.app.api.schemas.interpolation import (
    InterpolationRequest,
    InterpolationResponse
)
from backend.app.services.temporal_interpolation_service import TemporalInterpolationService

router = APIRouter()


@router.post(
    "/fill-gaps",
    response_model=InterpolationResponse,
    status_code=status.HTTP_200_OK,
    summary="Fill temporal gaps in observations",
    description="Interpolates missing dates with monsoon cloud-gap detection"
)
async def fill_gaps(request: InterpolationRequest):
    """
    Fills temporal gaps between satellite observations with monsoon cloud-gap handli

    Monsoon Cloud-Gap Trigger
    
    **Interpolation Methods:**
    - **linear**: Simple linear interpolation (fastest, least accurate)
    - **cubic_spline**: Smooth cubic spline interpolation (recommended)
    - **savgol**: Savitzky-Golay filter for noise reduction
    
    **Cloud-Gap Detection:**
    - Detects gaps > max_allowed_gap_days (default: 10 days)
    - For large gaps (monsoon clouds): Returns None instead of interpolating
    - Signals EnKF to hold open-loop (no assimilation) during these periods
    
    **Quality Flags:**
    - `{"date": "...", "type": "interpolated"}` - Successfully interpolated
    - `{"date": "...", "action": "HOLD_OPEN_LOOP", "reason": "Cloud gap > 10 days"}` - Gap detected
    
    **Example:**
    If satellite data is available on July 1, July 25 (24-day gap):
    - Days 1-10: Interpolated normally
    - Days 11-25: Set to None (hold open-loop)
    
    **Returns:**
    - Interpolated values (None for large gaps)
    - Quality flags per date
    - Gap detection summary
    """
    try:
        service = TemporalInterpolationService()
        result = service.interpolate(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interpolation failed: {str(e)}"
        )
