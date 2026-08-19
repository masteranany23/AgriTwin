from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session


from backend.app.db.session import get_db
from backend.app.api.schemas.fusion import (
    SpatialAlignmentRequest, SpatialAlignmentResponse,
    ConfidenceRequest, ConfidenceResponse,
    FusionRequest, FusionResponse,
    DataFusionPipelineRequest, DataFusionPipelineResponse,
    InterpolationRequest,
    InterpolationResponse,
)
from backend.app.services.spatial_alignment_service import SpatialAlignmentService
from backend.app.services.confidence_estimator import ConfidenceEstimator
from backend.app.services.multi_source_fusion_service import MultiSourceFusionService
from backend.app.services.data_fusion_pipeline import DataFusionPipeline
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
    Fills temporal gaps between satellite observations with monsoon cloud-gap handling

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


@router.post("/spatial-align", response_model=SpatialAlignmentResponse)
async def spatial_align(
    request: SpatialAlignmentRequest,
    db: Session = Depends(get_db)
):
    """Align multi-resolution observations to a unified field grid."""
    try:
        service = SpatialAlignmentService(db)
        return service.align_observations(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence", response_model=ConfidenceResponse)
async def compute_confidence(
    request: ConfidenceRequest
):
    """Compute confidence score and observation error (R) for EnKF."""
    try:
        service = ConfidenceEstimator()
        return service.compute_confidence(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fuse", response_model=FusionResponse)
async def fuse_sources(
    request: FusionRequest,
    db: Session = Depends(get_db)
):
    """Fuse multiple LAI/SM sources into a single best estimate."""
    try:
        service = MultiSourceFusionService(db)
        return service.fuse_lai(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pipeline", response_model=DataFusionPipelineResponse)
async def run_fusion_pipeline(
    request: DataFusionPipelineRequest,
    db: Session = Depends(get_db)
):
    """
    Run the complete Module 3.3 Data Fusion pipeline:
    Validation → Temporal → Spatial → Confidence → Fusion.
    """
    try:
        pipeline = DataFusionPipeline(db)
        return pipeline.process_range(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
