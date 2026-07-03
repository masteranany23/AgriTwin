"""
Error Correction Routes
=======================

FastAPI routes for WOFOST error correction using adaptive Kalman Gain.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.api.schemas.error_correction import (
    ErrorCorrectionRequest,
    ErrorCorrectionResponse
)
from backend.app.services.error_correction_service import ErrorCorrectionService

router = APIRouter()


@router.post(
    "/error-correction/correct-window",
    response_model=ErrorCorrectionResponse,
    summary="Correct WOFOST outputs using satellite data",
    description="Applies adaptive Kalman Gain correction to 7-day windows"
)
async def correct_window(
    request: ErrorCorrectionRequest,
    db: Session = Depends(get_db)
):
    """
    Corrects anomalies in a 7-day window using adaptive Kalman Gain.
    
    **Research Features Implemented:**
    - STEP 1: Adaptive Kalman Gain based on observation uncertainty
    - STEP 2: Joint LAI + Soil Moisture assimilation
    - Works with interpolated satellite LAI and ERA5-Land SM
    
    **Process:**
    1. Fetches WOFOST outputs (LAI, SM, DVS, TAGP)
    2. Retrieves interpolated satellite observations (LAI & SM)
    3. Computes Kalman Gain: K = Model_Error² / (Model_Error² + Obs_Error²)
    4. Applies correction: x_new = x_old + K * (observation - x_old)
    5. Updates DailyOutput table with corrected values
    
    **Observation Uncertainty:**
    - FARMER_PHOTO: 30% uncertainty (R=0.3) → Lower gain, trust model more
    - SENTINEL_2: 10% uncertainty (R=0.1) → Higher gain, trust satellite more
    - MODIS: 20% uncertainty (R=0.2) → Moderate gain
    
    **Returns:**
    - Correction summary for each day
    - Anomaly detection count
    - Blending weights (Kalman Gains) applied
    """
    try:
        service = ErrorCorrectionService(db)
        result = service.correct_window(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error correction failed: {str(e)}"
        )
