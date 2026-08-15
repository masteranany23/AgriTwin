"""
tests/test_observation_registry_service.py — Observation Registry Service Tests
================================================================================

Tests for ObservationRegistryService facade:
1. Single observation registration across all source types:
   - Sentinel observations (SENTINEL1_SAR, SATELLITE)
   - Weather observations (WEATHER, ERA5_LAND)
   - IoT sensor observations (IOT_SENSOR, SENSOR)
   - Weather-station observations (WEATHER_STATION)
   - Smartphone observations (SMARTPHONE_GRVI)
   - Manual scouting observations (MANUAL_SCOUT, MANUAL)
2. Metadata normalization (units inference, uncertainty defaults, UTC timezones, audit payload).
3. QualityControlService integration during registration.
4. Batch observation registration via facade.
5. API Endpoint testing: POST /observations/register.
"""

import datetime
import uuid
import pytest
from sqlalchemy.orm import Session

from backend.app.assimilation.models.observation import (
    Observation,
    ObservationSource,
    ObservationStatus,
)
from backend.app.assimilation.schemas.observation import (
    ObservationCreate,
    ObservationRegisterRequest,
)
from backend.app.services.observation_registry_service import ObservationRegistryService

UTC = datetime.timezone.utc
NOW = datetime.datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC)


def test_registry_service_sentinel_registration(test_db: Session):
    """Test registering Sentinel SAR & satellite observations."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "LAI",
        "value": 2.5,
        "source": "SENTINEL1_SAR",
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "LAI"
    assert obs.value == pytest.approx(2.5)
    assert obs.source == ObservationSource.SENTINEL1_SAR
    assert obs.units == "m2/m2"  # inferred
    assert obs.uncertainty == pytest.approx(0.30)  # inferred
    assert obs.provider_name == "Sentinel1_SAR"  # inferred
    assert obs.status == ObservationStatus.VALID
    assert "_registry_metadata" in obs.raw_payload


def test_registry_service_weather_registration(test_db: Session):
    """Test registering weather observations."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "AIR_TEMPERATURE",
        "value": 28.5,
        "source": "WEATHER",
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "AIR_TEMPERATURE"
    assert obs.value == pytest.approx(28.5)
    assert obs.source == ObservationSource.WEATHER
    assert obs.units == "degC"
    assert obs.uncertainty == pytest.approx(1.0)
    assert obs.status == ObservationStatus.VALID


def test_registry_service_iot_registration(test_db: Session):
    """Test registering IoT sensor observations."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "SM",
        "value": 0.28,
        "source": "IOT_SENSOR",
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "SM"
    assert obs.value == pytest.approx(0.28)
    assert obs.source == ObservationSource.IOT_SENSOR
    assert obs.units == "cm3/cm3"
    assert obs.uncertainty == pytest.approx(0.04)
    assert obs.status == ObservationStatus.VALID


def test_registry_service_weather_station_registration(test_db: Session):
    """Test registering weather station telemetry."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "RAINFALL",
        "value": 12.0,
        "source": "WEATHER_STATION",
        "provider_name": "Station_Lucknow_01",
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "RAINFALL"
    assert obs.value == pytest.approx(12.0)
    assert obs.source == ObservationSource.WEATHER_STATION
    assert obs.provider_name == "Station_Lucknow_01"
    assert obs.units == "mm/day"
    assert obs.uncertainty == pytest.approx(1.0)


def test_registry_service_smartphone_registration(test_db: Session):
    """Test registering smartphone GRVI photo observations."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "LAI",
        "value": 1.8,
        "source": "SMARTPHONE_GRVI",
        "quality_score": 85,
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "LAI"
    assert obs.value == pytest.approx(1.8)
    assert obs.source == ObservationSource.SMARTPHONE_GRVI
    assert obs.quality_score == 85
    assert obs.units == "m2/m2"


def test_registry_service_manual_scout_registration(test_db: Session):
    """Test registering manual scouting field observations."""
    service = ObservationRegistryService(test_db)
    
    req = {
        "timestamp": NOW,
        "variable_name": "DVS",
        "value": 1.2,
        "source": "MANUAL_SCOUT",
        "notes": "Flowering stage confirmed by agronomist",
    }
    obs = service.register_observation(req)
    
    assert obs.variable_name == "DVS"
    assert obs.value == pytest.approx(1.2)
    assert obs.source == ObservationSource.MANUAL_SCOUT
    assert obs.notes == "Flowering stage confirmed by agronomist"


def test_registry_service_qc_outlier_detection(test_db: Session):
    """Test that QualityControlService marks physical bounds failures as OUTLIER."""
    service = ObservationRegistryService(test_db)
    
    # LAI = 25.0 exceeds physical max bound (8.0)
    req = {
        "timestamp": NOW,
        "variable_name": "LAI",
        "value": 25.0,
        "source": "MANUAL_SCOUT",
    }
    obs = service.register_observation(req)
    
    assert obs.status == ObservationStatus.OUTLIER


def test_registry_service_batch_registration(test_db: Session):
    """Test bulk batch observation registration facade method."""
    service = ObservationRegistryService(test_db)
    
    payloads = [
        {"timestamp": NOW, "variable_name": "LAI", "value": 2.1, "source": "SENTINEL1_SAR"},
        {"timestamp": NOW, "variable_name": "SM", "value": 0.22, "source": "IOT_SENSOR"},
    ]
    saved = service.register_batch(payloads)
    
    assert len(saved) == 2
    assert saved[0].source == ObservationSource.SENTINEL1_SAR
    assert saved[1].source == ObservationSource.IOT_SENSOR


def test_api_post_observations_register(client):
    """Integration test for POST /observations/register endpoint."""
    payload = {
        "timestamp": "2026-08-15T12:00:00+00:00",
        "variable_name": "LAI",
        "value": 2.3,
        "source": "SENTINEL1_SAR",
    }
    resp = client.post("/observations/register", json=payload)
    
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["variable_name"] == "LAI"
    assert data["value"] == pytest.approx(2.3)
    assert data["source"] == "SENTINEL1_SAR"
    assert data["units"] == "m2/m2"
    assert data["uncertainty"] == pytest.approx(0.30)
    assert data["status"] == "VALID"
    assert "id" in data
