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
    summary="[DEPRECATED] Diagnostic window residual analysis",
    description=(
        "DEPRECATED: Evaluates diagnostic residuals between WOFOST outputs and QC-filtered "
        "observations. Does NOT mutate DailyOutput in the database. "
        "The canonical state-estimation path is: observations → QC → fusion → EnKF → assimilated WOFOST."
    ),
    deprecated=True,
)
async def correct_window(
    request: ErrorCorrectionRequest,
    db: Session = Depends(get_db)
):
    """
    Diagnostic residual calculation endpoint (Deprecated for direct state mutation).
    
    **Canonical Architecture Note:**
    State estimation must proceed through the sequential EnKF pipeline via `POST /assimilation/run-season`.
    This endpoint evaluates diagnostic residuals and recommended Kalman gains for auditability,
    passing observations through QualityControlService without altering DailyOutput database rows.
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
