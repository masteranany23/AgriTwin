"""
tests/test_farmer_workflow.py
=============================

End-to-End integration test simulating the entire Indian Smallholder Farmer
workflow (Ram Singh persona) across pre-season planning, simulation, remote sensing,
in-field smartphone scouting, EnKF assimilation, and daily advisory generation.
"""

import io
import pytest
from PIL import Image


def _create_synthetic_jpeg(color: tuple[int, int, int] = (50, 190, 50)) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_full_farmer_end_to_end_journey(client, test_db):
    """Execute complete 7-step farmer lifecycle."""
    
    # ── Step 1: Pre-Season Crop Recommendation ────────────────────────────────
    rec_resp = client.post(
        "/advisory/recommend-crop",
        json={
            "latitude": 26.8,
            "longitude": 80.9,
            "season": "rabi",
            "land_area_acres": 2.5,
        },
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert rec_data["top_recommendation"]["net_profit_per_acre_inr"] > 0
    top_crop = rec_data["top_recommendation"]["crop_name"]
    assert top_crop in ["Mustard", "Wheat", "Potato"]

    # ── Step 2: Register Field Plot ───────────────────────────────────────────
    field_resp = client.post(
        "/fields",
        json={
            "name": "Ram Singh Wheat Plot 1",
            "latitude": 26.8,
            "longitude": 80.9,
            "farm_name": "Singh Family Farm",
            "boundary_geojson": {
                "type": "Polygon",
                "coordinates": [[
                    [80.89, 26.79],
                    [80.91, 26.79],
                    [80.91, 26.81],
                    [80.89, 26.81],
                    [80.89, 26.79]
                ]]
            },
        },
    )
    assert field_resp.status_code in (200, 201)
    field_id = field_resp.json()["field_id"]

    # ── Step 3: Run Baseline Open-Loop Simulation ─────────────────────────────
    sim_payload = {
        "latitude": 26.8,
        "longitude": 80.9,
        "crop": "rice",
        "variety": "Rice_IR64",
        "sowing_date": "2020-06-20",
        "harvest_date": "2020-11-10",
        "max_duration": 220,
        "use_real_weather": False,
        "use_real_soil": False,
        "field_id": field_id,
        "irrigation_events": [
            {"date": "2020-07-05", "amount_mm": 50.0},
            {"date": "2020-07-20", "amount_mm": 50.0},
        ],
    }
    sim_resp = client.post("/simulate", json=sim_payload)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    sim_id = sim_data["simulation_id"]
    assert sim_data["metrics"]["peak_lai"] > 0
    assert sim_data["metrics"]["final_tagp_kg_ha"] > 0

    # ── Step 4: Ingest Sentinel-2 LAI Observations ────────────────────────────
    sat_resp = client.get(
        f"/satellite/lai?field_id={field_id}"
        f"&start_date=2020-06-20&end_date=2020-11-10"
        f"&index_name=NDVI&uncertainty=0.3"
    )
    assert sat_resp.status_code == 200
    sat_data = sat_resp.json()
    assert len(sat_data) > 0

    # ── Step 5: Submit W-Shape Smartphone Scouting Photos ─────────────────────
    img_bytes = _create_synthetic_jpeg((45, 175, 45))
    files = [
        ("images", ("node1.jpg", img_bytes, "image/jpeg")),
        ("images", ("node2.jpg", img_bytes, "image/jpeg")),
        ("images", ("node3.jpg", img_bytes, "image/jpeg")),
        ("images", ("node4.jpg", img_bytes, "image/jpeg")),
        ("images", ("node5.jpg", img_bytes, "image/jpeg")),
    ]
    scout_resp = client.post(
        f"/fields/{field_id}/scout-session",
        files=files,
        data={"session_notes": "Ram Singh scouting at Day 45"},
    )
    assert scout_resp.status_code == 201
    scout_data = scout_resp.json()
    assert scout_data["processing_status"] == "completed"
    assert scout_data["estimated_lai"] > 0

    # ── Step 6: Execute EnKF Data Assimilation ────────────────────────────────
    assim_resp = client.post(
        "/assimilation/run",
        json={
            "simulation_id": sim_id,
            "field_id": field_id,
            "ensemble_size": 25,
        },
    )
    assert assim_resp.status_code == 200
    assim_data = assim_resp.json()
    assert assim_data["status"] == "COMPLETED"
    assert assim_data["executed_cycles"] > 0

    # ── Step 7: Retrieve Daily Farmer Advisory & Summary Card ─────────────────
    daily_adv_resp = client.get(f"/advisory/field/{field_id}/daily")
    assert daily_adv_resp.status_code == 200
    daily_adv = daily_adv_resp.json()
    assert daily_adv["field_id"] == field_id
    assert len(daily_adv["advisories"]) >= 1

    summary_resp = client.get(f"/advisory/field/{field_id}/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["field_id"] == field_id
    assert summary["expected_yield_kg_ha"] > 0
    assert summary["confidence_percentage"] > 70.0
    assert "AgriTwin" in summary["card_text_en"]
    assert "एग्रीट्विन" in summary["card_text_hi"]
