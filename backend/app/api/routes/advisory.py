"""
api/routes/advisory.py — Farmer Advisory & Decision Support Endpoints
=====================================================================

Endpoints:
  POST /advisory/recommend-crop        → Crop selection & profit ranking (Farmer Need 1)
  GET  /advisory/field/{id}/daily      → Daily actionable alerts (irrigation, nitrogen, weather) (Needs 4, 5, 7)
  GET  /advisory/field/{id}/summary    → End-to-end dashboard & WhatsApp card (Needs 3, 6, UI)
"""

import datetime
import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.api.schemas.advisory import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    FieldAdvisoryResponse,
    FarmerSummaryResponse,
)
from backend.app.services.crop_recommendation_service import CropRecommendationService
from backend.app.services.advisory_service import AdvisoryService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/recommend-crop",
    response_model=CropRecommendationResponse,
    summary="Recommend best crops for a field & season",
    description=(
        "Evaluates candidate crops based on geographic location, season (Kharif/Rabi/Zaid), "
        "soil characteristics, and economic return (MSP vs cultivation cost) to recommend "
        "the most profitable crop choice."
    ),
    tags=["Farmer Advisory & Decision Support"],
)
def recommend_crops(
    request: CropRecommendationRequest,
    db: Session = Depends(get_db),
) -> CropRecommendationResponse:
    """Evaluate candidate crops and return ranked recommendations with profit breakdown."""
    try:
        service = CropRecommendationService(db)
        return service.recommend_crops(request)
    except Exception as e:
        logger.error("Crop recommendation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate crop recommendations: {str(e)}")


@router.get(
    "/field/{field_id}/daily",
    response_model=FieldAdvisoryResponse,
    summary="Get today's actionable alerts for a field",
    description=(
        "Evaluates the current simulation and observation state of a field to produce "
        "specific action items (e.g. irrigate tomorrow morning, apply urea within 3 days) "
        "in English and Hindi."
    ),
    tags=["Farmer Advisory & Decision Support"],
)
def get_field_daily_advisory(
    field_id: uuid.UUID,
    date: Optional[datetime.date] = Query(None, description="Target date (default: today)"),
    db: Session = Depends(get_db),
) -> FieldAdvisoryResponse:
    """Generate daily field advisory and alerts."""
    try:
        service = AdvisoryService(db)
        return service.get_field_daily_advisory(field_id=field_id, target_date=date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Daily advisory generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate daily advisory: {str(e)}")


@router.get(
    "/field/{field_id}/summary",
    response_model=FarmerSummaryResponse,
    summary="Get comprehensive farmer summary card",
    description=(
        "Returns the complete end-to-end dashboard card including current stage, "
        "expected yield (kg/ha and Quintals/acre), calibrated confidence interval, "
        "historical comparison, and ready-to-display WhatsApp / SMS cards in English and Hindi."
    ),
    tags=["Farmer Advisory & Decision Support"],
)
def get_farmer_summary(
    field_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> FarmerSummaryResponse:
    """Compile executive summary card for farmer's phone."""
    try:
        service = AdvisoryService(db)
        return service.get_farmer_summary(field_id=field_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Farmer summary compilation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compile farmer summary: {str(e)}")
