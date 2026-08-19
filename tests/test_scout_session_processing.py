"""
tests/test_scout_session_processing.py
======================================

Unit and integration tests for W-Shape Smartphone GRVI Protocol.
"""

import io
import uuid
import pytest
import numpy as np
from PIL import Image

from backend.app.models.farm import Farm
from backend.app.models.field import Field
from backend.app.assimilation.models.observation import Observation


def _create_synthetic_jpeg(color: tuple[int, int, int] = (40, 180, 40)) -> bytes:
    """Helper to generate an in-memory RGB JPEG image."""
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_scout_session_w_shape_upload(client, test_db):
    """Test 5-photo W-Shape scouting upload, GRVI calculation, and DB persistence."""
    # 1. Create Farm & Field
    farm = Farm(name="Scout Test Farm")
    test_db.add(farm)
    test_db.commit()

    field = Field(farm_id=farm.id, name="Ram Singh Scout Field", latitude=26.8, longitude=80.9)
    test_db.add(field)
    test_db.commit()
    test_db.refresh(field)

    # 2. Prepare 5 synthetic crop photos (healthy green)
    img_bytes_green = _create_synthetic_jpeg((40, 180, 40))   # G > R -> Positive GRVI
    img_bytes_yellow = _create_synthetic_jpeg((160, 160, 40)) # Yellowish node

    files = [
        ("images", ("node1.jpg", img_bytes_green, "image/jpeg")),
        ("images", ("node2.jpg", img_bytes_green, "image/jpeg")),
        ("images", ("node3.jpg", img_bytes_yellow, "image/jpeg")),
        ("images", ("node4.jpg", img_bytes_green, "image/jpeg")),
        ("images", ("node5.jpg", img_bytes_green, "image/jpeg")),
    ]
    data = {"session_notes": "Morning field walk, clear sky"}

    resp = client.post(
        f"/fields/{field.id}/scout-session",
        files=files,
        data=data,
    )
    assert resp.status_code == 201
    res_json = resp.json()

    assert res_json["field_id"] == str(field.id)
    assert res_json["processing_status"] == "completed"
    assert len(res_json["node_grvi_values"]) == 5
    assert res_json["estimated_lai"] > 0.0
    assert res_json["observation_uncertainty"] > 0.0
    assert "observation_id" in res_json

    # 3. Check DB observation persistence
    obs_id = uuid.UUID(res_json["observation_id"])
    obs_row = test_db.query(Observation).filter(Observation.id == obs_id).first()
    assert obs_row is not None
    assert obs_row.field_id == field.id
    assert obs_row.variable_name == "LAI"
    assert obs_row.value == res_json["estimated_lai"]
    assert obs_row.uncertainty == res_json["observation_uncertainty"]


def test_scout_session_invalid_image_count(client, test_db):
    """Test validation failure when fewer than 5 images are uploaded."""
    farm = Farm(name="Invalid Test Farm")
    test_db.add(farm)
    test_db.commit()

    field = Field(farm_id=farm.id, name="Invalid Count Field", latitude=26.8, longitude=80.9)
    test_db.add(field)
    test_db.commit()

    # Upload only 3 images instead of 5
    files = [
        ("images", ("node1.jpg", _create_synthetic_jpeg(), "image/jpeg")),
        ("images", ("node2.jpg", _create_synthetic_jpeg(), "image/jpeg")),
        ("images", ("node3.jpg", _create_synthetic_jpeg(), "image/jpeg")),
    ]
    resp = client.post(f"/fields/{field.id}/scout-session", files=files)
    assert resp.status_code == 400
    assert "Exactly 5 images required" in resp.json()["detail"]


def test_list_scout_sessions_endpoint(client, test_db):
    """Test retrieving history of scout sessions for a field."""
    farm = Farm(name="History Test Farm")
    test_db.add(farm)
    test_db.commit()

    field = Field(farm_id=farm.id, name="History Field", latitude=26.8, longitude=80.9)
    test_db.add(field)
    test_db.commit()

    # Query scout sessions
    resp = client.get(f"/fields/{field.id}/scout-sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["field_id"] == str(field.id)
    assert "sessions" in data
