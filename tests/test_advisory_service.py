"""
tests/test_advisory_service.py
==============================

Unit and integration tests for Crop Recommendation & Farmer Advisory Engine.
"""

import uuid
import datetime
import pytest

from backend.app.models.farm import Farm
from backend.app.models.field import Field
from backend.app.models.simulation_run import SimulationRun
from backend.app.models.daily_output import DailyOutput
from backend.app.services.crop_recommendation_service import CropRecommendationService
from backend.app.services.advisory_service import AdvisoryService
from backend.app.api.schemas.advisory import CropRecommendationRequest


def test_crop_recommendation_rabi(test_db):
    """Test crop recommendation ranking for Rabi season in Uttar Pradesh."""
    service = CropRecommendationService(test_db)
    req = CropRecommendationRequest(
        latitude=26.8467,
        longitude=80.9462,
        season="rabi",
        land_area_acres=3.0,
    )
    resp = service.recommend_crops(req)

    assert resp.season == "rabi"
    assert resp.land_area_acres == 3.0
    assert len(resp.ranked_options) > 0
    assert resp.top_recommendation is not None
    assert resp.top_recommendation.net_profit_per_acre_inr > 0
    assert resp.top_recommendation.total_net_profit_inr > 0
    assert "Mustard" in [o.crop_name for o in resp.ranked_options]
    assert "Wheat" in [o.crop_name for o in resp.ranked_options]
    assert len(resp.summary_message_hi) > 10
    assert len(resp.summary_message_en) > 10


def test_crop_recommendation_kharif(test_db):
    """Test crop recommendation for Kharif season."""
    service = CropRecommendationService(test_db)
    req = CropRecommendationRequest(
        latitude=26.8,
        longitude=80.9,
        season="kharif",
        land_area_acres=2.0,
    )
    resp = service.recommend_crops(req)

    assert resp.season == "kharif"
    assert any(o.crop_name == "Paddy (Rice)" for o in resp.ranked_options)
    assert any(o.crop_name == "Maize" for o in resp.ranked_options)


def test_advisory_service_low_moisture(test_db):
    """Test that low soil moisture triggers urgent irrigation advisory."""
    farm = Farm(name="Ram Singh Farm")
    test_db.add(farm)
    test_db.commit()

    field = Field(farm_id=farm.id, name="Ram Singh Wheat Plot", latitude=26.8, longitude=80.9)
    test_db.add(field)
    test_db.commit()

    sim_run = SimulationRun(
        field_id=field.id,
        latitude=26.8,
        longitude=80.9,
        crop="wheat",
        variety="Winter_wheat_101",
        sowing_date=datetime.date(2023, 11, 10),
        status="completed",
        yield_kg_ha=4500.0,
        metrics_payload={"final_twso_kg_ha": 4500.0, "peak_lai": 3.8},
    )
    test_db.add(sim_run)
    test_db.commit()

    daily = DailyOutput(
        simulation_run_id=sim_run.id,
        date=datetime.date(2023, 12, 15),
        dvs=0.55,
        lai=2.1,
        sm=0.15,      # severely depleted moisture
        rftra=0.60,   # high water stress
        tagp=1500.0,
        twso=0.0,
    )
    test_db.add(daily)
    test_db.commit()

    service = AdvisoryService(test_db)
    adv_resp = service.get_field_daily_advisory(field.id, target_date=datetime.date(2023, 12, 15))

    assert adv_resp.field_id == field.id
    assert adv_resp.soil_moisture_status == "Critical Deficit"
    assert len(adv_resp.advisories) >= 1
    irr_adv = next((a for a in adv_resp.advisories if a.category == "irrigation"), None)
    assert irr_adv is not None
    assert irr_adv.severity == "critical"
    assert "सिंचाई" in irr_adv.action_hi


def test_advisory_farmer_summary(test_db):
    """Test full farmer summary card compilation."""
    farm = Farm(name="Ram Singh Farm")
    test_db.add(farm)
    test_db.commit()

    field = Field(farm_id=farm.id, name="Ram Singh Demo Farm", latitude=26.8, longitude=80.9)
    test_db.add(field)
    test_db.commit()

    sim_run = SimulationRun(
        field_id=field.id,
        latitude=26.8,
        longitude=80.9,
        crop="wheat",
        variety="HD-2967",
        sowing_date=datetime.date.today() - datetime.timedelta(days=40),
        status="completed",
        yield_kg_ha=4450.0,
        metrics_payload={"final_twso_kg_ha": 4450.0},
    )
    test_db.add(sim_run)
    test_db.commit()

    service = AdvisoryService(test_db)
    summary = service.get_farmer_summary(field.id)

    assert summary.field_id == field.id
    assert summary.expected_yield_kg_ha == 4450.0
    assert summary.expected_yield_quintal_acre > 15.0
    assert summary.confidence_percentage >= 80.0
    assert "एग्रीट्विन किसान सलाह" in summary.card_text_hi
    assert "AgriTwin Farm Advisory" in summary.card_text_en


def test_advisory_api_endpoints(client, test_db):
    """Test HTTP API endpoints for crop recommendation and advisories."""
    # 1. Recommend crop
    rec_payload = {
        "latitude": 26.8,
        "longitude": 80.9,
        "season": "rabi",
        "land_area_acres": 2.5,
    }
    resp = client.post("/advisory/recommend-crop", json=rec_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "top_recommendation" in data
    assert len(data["ranked_options"]) >= 2

    # 2. Register field
    field_resp = client.post("/fields", json={"name": "API Test Field", "latitude": 26.8, "longitude": 80.9})
    assert field_resp.status_code in (200, 201)
    field_id = field_resp.json()["field_id"]

    # 3. Daily advisory
    adv_resp = client.get(f"/advisory/field/{field_id}/daily")
    assert adv_resp.status_code == 200
    adv_data = adv_resp.json()
    assert adv_data["field_id"] == field_id
    assert "advisories" in adv_data

    # 4. Summary card
    sum_resp = client.get(f"/advisory/field/{field_id}/summary")
    assert sum_resp.status_code == 200
    sum_data = sum_resp.json()
    assert sum_data["field_id"] == field_id
    assert "card_text_hi" in sum_data
    assert "card_text_en" in sum_data
