"""
tests/test_multi_source_fusion.py
==================================

Unit tests for MultiSourceFusionService verifying uncertainty-informed weighting,
weight shifts based on observation error R / uncertainty, fallback behavior,
and preservation of cloud-cover filters.
"""

from datetime import date
from uuid import uuid4

import pytest

from backend.app.api.schemas.fusion import FusionRequest
from backend.app.services.multi_source_fusion_service import MultiSourceFusionService


@pytest.fixture
def fusion_service():
    """Fixture providing MultiSourceFusionService with dummy DB session."""
    return MultiSourceFusionService(db_session=None)


def test_lower_uncertainty_observation_receives_greater_influence(fusion_service):
    """Verify that an observation with lower uncertainty receives a higher weight."""
    field_id = uuid4()
    today = date(2024, 7, 15)

    # Observation 1: Sentinel-2 with high uncertainty (std_dev = 0.50 -> R = 0.25)
    # Observation 2: Smartphone GRVI with low uncertainty (std_dev = 0.05 -> R = 0.0025)
    req = FusionRequest(
        field_id=field_id,
        date=today,
        observations=[
            {
                "source": "SENTINEL2",
                "value": 2.0,
                "uncertainty": 0.50,  # High uncertainty
            },
            {
                "source": "SMARTPHONE_GRVI",
                "value": 0.8,  # GRVI ~0.8 maps to LAI ~3.86
                "uncertainty": 0.05,  # Very low uncertainty
            },
        ],
        cloud_cover=20.0,
    )

    resp = fusion_service.fuse_lai(req)

    assert "SENTINEL2" in resp.source_weights
    assert "SMARTPHONE_GRVI" in resp.source_weights
    # Smartphone GRVI has much lower uncertainty, so its weight must be strictly higher
    assert resp.source_weights["SMARTPHONE_GRVI"] > resp.source_weights["SENTINEL2"]


def test_decreasing_uncertainty_increases_weight(fusion_service):
    """Verify that reducing an observation's uncertainty increases its fused weight."""
    field_id = uuid4()
    today = date(2024, 7, 15)

    # Case A: Sentinel-2 has standard high uncertainty
    req_a = FusionRequest(
        field_id=field_id,
        date=today,
        observations=[
            {"source": "SENTINEL2", "value": 3.0, "uncertainty": 0.40},
            {"source": "SENTINEL1_SAR", "value": 0.5, "uncertainty": 0.20},
        ],
        cloud_cover=20.0,
    )
    resp_a = fusion_service.fuse_lai(req_a)

    # Case B: Sentinel-2 uncertainty is reduced to 0.05
    req_b = FusionRequest(
        field_id=field_id,
        date=today,
        observations=[
            {"source": "SENTINEL2", "value": 3.0, "uncertainty": 0.05},
            {"source": "SENTINEL1_SAR", "value": 0.5, "uncertainty": 0.20},
        ],
        cloud_cover=20.0,
    )
    resp_b = fusion_service.fuse_lai(req_b)

    # Sentinel-2 weight in Case B should be greater than in Case A
    assert resp_b.source_weights["SENTINEL2"] > resp_a.source_weights["SENTINEL2"]


def test_fallback_when_uncertainty_omitted(fusion_service):
    """Verify safe fallback behavior when explicit uncertainty fields are omitted."""
    field_id = uuid4()
    today = date(2024, 7, 15)

    req = FusionRequest(
        field_id=field_id,
        date=today,
        observations=[
            {"source": "SENTINEL2", "value": 3.0},
            {"source": "SENTINEL1_SAR", "value": 0.5},
        ],
        cloud_cover=10.0,
    )

    resp = fusion_service.fuse_lai(req)

    assert resp.fused_lai > 0.0
    assert resp.fused_confidence > 0.0
    assert len(resp.contributing_sources) == 2
    assert "SENTINEL2" in resp.source_weights
    assert "SENTINEL1_SAR" in resp.source_weights


def test_cloud_filtering_preserved(fusion_service):
    """Verify that >70% cloud cover excludes Sentinel-2 regardless of low uncertainty."""
    field_id = uuid4()
    today = date(2024, 7, 15)

    req = FusionRequest(
        field_id=field_id,
        date=today,
        observations=[
            {"source": "SENTINEL2", "value": 3.0, "uncertainty": 0.01},
            {"source": "SENTINEL1_SAR", "value": 0.5, "uncertainty": 0.20},
        ],
        cloud_cover=80.0,  # Heavy cloud cover (>70%)
    )

    resp = fusion_service.fuse_lai(req)

    # Sentinel-2 must be filtered out due to cloud cover
    assert "SENTINEL2" not in resp.contributing_sources
    assert resp.source_weights.get("SENTINEL2", 0.0) == 0.0
    assert "SENTINEL1_SAR" in resp.contributing_sources
    assert resp.quality_flag == "LOW"
