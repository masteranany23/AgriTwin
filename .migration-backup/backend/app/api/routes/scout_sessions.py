"""
api/routes/scout_sessions.py — W-Shape GRVI Protocol Endpoints
================================================================

POST   /fields/{field_id}/scout-session  → Upload 5 smartphone photos (W-Shape protocol)
GET    /fields/{field_id}/scout-sessions → List all scout sessions for a field
GET    /scout-sessions/{session_id}      → Get single scout session details

W-Shape Protocol:
-----------------
- Farmer takes exactly 5 photos in W-shape pattern across field
- Photos must have GPS EXIF data
- Backend computes median GRVI (Green-Red Vegetation Index)
- Converts GRVI to LAI estimate
- Sets 30% observation error ("Gentle Nudge") for EnKF
- Rejects outliers (muddy puddles, weeds) using 2-sigma filter

See docs/w_shape_grvi_protocol.md for full protocol specification.
"""

import logging
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import uuid

from backend.app.db.session import get_db
from backend.app.repositories.field_repository import FieldRepository

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/{field_id}/scout-session",
    status_code=201,
    summary="Submit W-Shape scout session (5 photos)",
    description=(
        "Upload exactly 5 smartphone photos following the W-Shape protocol. "
        "Backend computes median GRVI and converts to LAI estimate. "
        "Photos must have GPS EXIF data. See docs/w_shape_grvi_protocol.md"
    ),
    tags=["Scout Sessions"],
)
async def create_scout_session(
    field_id: uuid.UUID,
    images: List[UploadFile] = File(..., description="Exactly 5 JPEG images with GPS EXIF"),
    session_notes: str = Form(None, description="Optional notes (e.g., 'W-shape walk, clear sky')"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Process W-Shape scout session with 5 smartphone photos.
    
    Pipeline:
    1. Validate 5 images, check total size < 30 MB
    2. Extract GPS EXIF from each image
    3. Compress images to 1280×1280 @ 75% quality
    4. Calculate GRVI per image
    5. Reject outliers (2-sigma filter)
    6. Compute median GRVI
    7. Convert GRVI to LAI: LAI = 0.5 + (GRVI × 3.0)
    8. Set uncertainty = LAI × 0.30 (30% "Gentle Nudge")
    9. Create observation record
    10. Return session details
    
    Returns:
        dict: Session ID, estimated LAI, observation error, confidence, quality score
    
    Raises:
        HTTPException: If validation fails (wrong count, no GPS, size exceeded)
    """
    # Step 1: Validate image count
    if len(images) != 5:
        raise HTTPException(
            status_code=400,
            detail=f"Exactly 5 images required for W-shape protocol. Received {len(images)}."
        )
    
    # Step 2: Check field exists
    repo = FieldRepository(db)
    field = repo.get_field(field_id)
    if field is None:
        raise HTTPException(status_code=404, detail=f"Field {field_id} not found.")
    
    # TODO: Implement full processing pipeline:
    # - GPS EXIF extraction (piexif)
    # - Image compression (PIL)
    # - GRVI calculation (numpy)
    # - Outlier rejection (2-sigma filter)
    # - LAI estimation
    # - Spatial alignment (snap to 10m grid)
    # - Create FieldObservation record
    
    logger.info(
        "POST /fields/%s/scout-session → %d images, notes=%r",
        field_id, len(images), session_notes
    )
    
    # Placeholder response (MVP)
    return {
        "session_id": str(uuid.uuid4()),
        "field_id": str(field_id),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "processing_status": "pending",
        "message": (
            "W-Shape GRVI protocol received. "
            "Full implementation pending (GPS extraction, GRVI calculation, LAI conversion). "
            "See docs/w_shape_grvi_protocol.md for complete specification."
        ),
        "next_steps": [
            "Implement GPS EXIF extraction (piexif)",
            "Implement GRVI calculation (numpy)",
            "Implement LAI conversion and uncertainty quantification",
            "Create FieldObservation record",
            "Trigger data fusion pipeline"
        ]
    }


@router.get(
    "/{field_id}/scout-sessions",
    summary="List scout sessions for a field",
    description="Retrieve all W-Shape scout sessions for a specific field",
    tags=["Scout Sessions"],
)
def list_scout_sessions(
    field_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict:
    """
    List all scout sessions for a field.
    
    TODO: Implement database query for scout sessions.
    
    Returns:
        dict: List of scout sessions with metadata
    """
    repo = FieldRepository(db)
    field = repo.get_field(field_id)
    if field is None:
        raise HTTPException(status_code=404, detail=f"Field {field_id} not found.")
    
    return {
        "field_id": str(field_id),
        "total": 0,
        "sessions": [],
        "message": "Scout session history pending implementation"
    }


@router.get(
    "/scout-sessions/{session_id}",
    summary="Get scout session details",
    description="Retrieve detailed information about a specific scout session",
    tags=["Scout Sessions"],
)
def get_scout_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> dict:
    """
    Get details of a specific scout session.
    
    TODO: Implement database query for session details.
    
    Returns:
        dict: Session details including GRVI, LAI, confidence, quality score
    """
    return {
        "session_id": str(session_id),
        "message": "Scout session detail retrieval pending implementation"
    }
