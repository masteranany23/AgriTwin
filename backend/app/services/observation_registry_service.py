"""backend/app/services/observation_registry_service.py — Unified Observation Registry Service
=============================================================================================

Provides a thin unified ingestion facade for heterogeneous observation data sources:
- Sentinel observations (Sentinel-2, Sentinel-1 SAR)
- Weather observations (ERA5-Land, NASA POWER)
- IoT sensor observations (Soil moisture probes, telemetry arrays)
- Weather-station observations (On-farm weather stations)
- Smartphone observations (W-shape GRVI field photo measurements)
- Manual scouting observations (Agronomist field logs)

Responsibilities:
1. Normalizes metadata (timestamps to UTC, source enums, units, default uncertainties, provider names).
2. Runs QualityControlService evaluation to set initial lifecycle status (VALID, OUTLIER, REJECTED, MISSING).
3. Attaches normalization audit metadata into `raw_payload`.
4. Delegates DB persistence to ObservationRepository.
"""

import uuid
import datetime
import logging
from typing import Optional, Union, Any

from sqlalchemy.orm import Session

from backend.app.assimilation.models.observation import (
    Observation,
    ObservationSource,
    ObservationStatus,
)
from backend.app.assimilation.repositories.observation_repository import ObservationRepository
from backend.app.assimilation.schemas.observation import (
    ObservationCreate,
    ObservationRegisterRequest,
)
from backend.app.services.quality_control_service import QualityControlService

logger = logging.getLogger(__name__)

# Default unit mapping for standard crop and environmental variables
VARIABLE_DEFAULT_UNITS: dict[str, str] = {
    "LAI": "m2/m2",
    "SM": "cm3/cm3",
    "ROOT_ZONE_SOIL_MOISTURE": "cm3/cm3",
    "ROOT_ZONE_SM": "cm3/cm3",
    "SURFACE_SOIL_MOISTURE": "cm3/cm3",
    "SURFACE_SM": "cm3/cm3",
    "TAGP": "kg/ha",
    "TWSO": "kg/ha",
    "TWLV": "kg/ha",
    "TWST": "kg/ha",
    "TWRT": "kg/ha",
    "DVS": "-",
    "RD": "cm",
    "AIR_TEMPERATURE": "degC",
    "CANOPY_TEMPERATURE": "degC",
    "RELATIVE_HUMIDITY": "%",
    "RAINFALL": "mm/day",
    "NDVI": "-",
    "EVI": "-",
    "GRVI": "-",
}

# Default uncertainty standard deviations (std dev in same units as value)
VARIABLE_DEFAULT_UNCERTAINTY: dict[str, float] = {
    "LAI": 0.30,
    "SM": 0.04,
    "ROOT_ZONE_SOIL_MOISTURE": 0.04,
    "ROOT_ZONE_SM": 0.04,
    "SURFACE_SOIL_MOISTURE": 0.04,
    "SURFACE_SM": 0.04,
    "TAGP": 500.0,
    "TWSO": 200.0,
    "TWLV": 100.0,
    "TWST": 150.0,
    "TWRT": 100.0,
    "DVS": 0.05,
    "RD": 5.0,
    "AIR_TEMPERATURE": 1.0,
    "CANOPY_TEMPERATURE": 1.5,
    "RELATIVE_HUMIDITY": 5.0,
    "RAINFALL": 1.0,
    "NDVI": 0.05,
    "EVI": 0.05,
    "GRVI": 0.05,
}

# Source name mapping / normalization dictionary
SOURCE_MAPPING: dict[str, ObservationSource] = {
    "SATELLITE": ObservationSource.SATELLITE,
    "SENTINEL2": ObservationSource.SATELLITE,
    "SENTINEL1": ObservationSource.SENTINEL1_SAR,
    "SENTINEL1_SAR": ObservationSource.SENTINEL1_SAR,
    "SMARTPHONE": ObservationSource.SMARTPHONE_GRVI,
    "SMARTPHONE_GRVI": ObservationSource.SMARTPHONE_GRVI,
    "SMARTPHONE_RGB": ObservationSource.SMARTPHONE_GRVI,
    "IOT": ObservationSource.IOT_SENSOR,
    "IOT_SENSOR": ObservationSource.IOT_SENSOR,
    "SENSOR": ObservationSource.SENSOR,
    "WEATHER_STATION": ObservationSource.WEATHER_STATION,
    "STATION": ObservationSource.WEATHER_STATION,
    "WEATHER": ObservationSource.WEATHER,
    "ERA5_LAND": ObservationSource.WEATHER,
    "NASA_POWER": ObservationSource.WEATHER,
    "MANUAL_SCOUT": ObservationSource.MANUAL_SCOUT,
    "SCOUT": ObservationSource.MANUAL_SCOUT,
    "MANUAL": ObservationSource.MANUAL,
    "MODEL": ObservationSource.MODEL,
}

# Provider fallback names per source type
PROVIDER_DEFAULT_NAMES: dict[ObservationSource, str] = {
    ObservationSource.SATELLITE: "Sentinel2_L2A",
    ObservationSource.SENTINEL1_SAR: "Sentinel1_SAR",
    ObservationSource.SMARTPHONE_GRVI: "Smartphone_GRVI_Scout",
    ObservationSource.IOT_SENSOR: "IoT_Telemetry_Probe",
    ObservationSource.SENSOR: "InSitu_SoilSensor",
    ObservationSource.WEATHER_STATION: "WeatherStation_OnFarm",
    ObservationSource.WEATHER: "ERA5_Land",
    ObservationSource.MANUAL_SCOUT: "Manual_FieldScout",
    ObservationSource.MANUAL: "Manual_AgronomistLog",
    ObservationSource.MODEL: "WOFOST72_Synthetic",
}


class ObservationRegistryService:
    """Unified observation ingestion and registration facade.
    
    Acts as a thin facade over ObservationRepository, normalizing metadata,
    inferring standard defaults, applying quality control gates, and persisting
    clean Observation ORM records.
    """

    def __init__(
        self,
        db: Session,
        qc_service: Optional[QualityControlService] = None,
    ) -> None:
        self.db = db
        self.repository = ObservationRepository(db)
        self.qc_service = qc_service or QualityControlService()

    def normalize_source(self, source_raw: Union[str, ObservationSource]) -> ObservationSource:
        """Map raw source string or enum to ObservationSource."""
        if isinstance(source_raw, ObservationSource):
            return source_raw
        src_upper = str(source_raw).strip().upper()
        if src_upper in SOURCE_MAPPING:
            return SOURCE_MAPPING[src_upper]
        try:
            return ObservationSource(src_upper)
        except ValueError:
            logger.warning("Unknown observation source '%s', defaulting to SATELLITE", source_raw)
            return ObservationSource.SATELLITE

    def normalize_metadata(
        self,
        payload: Union[dict[str, Any], ObservationCreate, ObservationRegisterRequest],
    ) -> dict[str, Any]:
        """Normalize observation fields into a cleaned dict ready for ORM construction."""
        if isinstance(payload, (ObservationCreate, ObservationRegisterRequest)):
            data = payload.model_dump()
        elif isinstance(payload, dict):
            data = payload.copy()
        else:
            raise ValueError(f"Unsupported observation payload type: {type(payload)}")

        # 1. Normalize Variable Name
        var_name = str(data.get("variable_name", "")).strip().upper()
        if not var_name:
            raise ValueError("variable_name is required for observation registration")
        data["variable_name"] = var_name

        # 2. Normalize Timestamp (Ensure UTC timezone-aware)
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts)
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        data["timestamp"] = ts

        # 3. Normalize Source
        src_enum = self.normalize_source(data.get("source", "SATELLITE"))
        data["source"] = src_enum

        # 4. Infer Provider Name if missing
        if not data.get("provider_name"):
            data["provider_name"] = PROVIDER_DEFAULT_NAMES.get(src_enum, "AgriTwin_Ingestion_Provider")

        # 5. Infer Units if missing
        if not data.get("units"):
            data["units"] = VARIABLE_DEFAULT_UNITS.get(var_name, "-")

        # 6. Infer Uncertainty if missing
        unc = data.get("uncertainty")
        if unc is None or (isinstance(unc, (int, float)) and unc <= 0):
            data["uncertainty"] = VARIABLE_DEFAULT_UNCERTAINTY.get(var_name, 0.10)

        # 7. Normalize raw_payload and add registry provenance
        raw_payload = data.get("raw_payload") or {}
        if not isinstance(raw_payload, dict):
            raw_payload = {"original_payload": raw_payload}
        reg_meta = {
            "registered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "inferred_units": data["units"],
            "inferred_uncertainty": data["uncertainty"],
            "normalized_source": src_enum.value,
        }
        if var_name in ("SURFACE_SOIL_MOISTURE", "SURFACE_SM"):
            reg_meta["observation_depth"] = "0-5 cm"
            reg_meta["observation_support"] = "surface_skin"
            reg_meta["model_target_variable"] = "SM"
        elif var_name in ("ROOT_ZONE_SOIL_MOISTURE", "ROOT_ZONE_SM"):
            reg_meta["observation_depth"] = "0-100 cm"
            reg_meta["observation_support"] = "root_zone"
            reg_meta["model_target_variable"] = "SM"
        elif var_name == "SM":
            if src_enum.value in ("SATELLITE", "SENTINEL1_SAR"):
                reg_meta["observation_depth"] = "0-5 cm"
                reg_meta["observation_support"] = "surface_skin"
                reg_meta["model_target_variable"] = "SM"
            else:
                reg_meta["observation_depth"] = "0-100 cm"
                reg_meta["observation_support"] = "root_zone"
                reg_meta["model_target_variable"] = "SM"
        raw_payload["_registry_metadata"] = reg_meta
        data["raw_payload"] = raw_payload

        return data

    def register_observation(
        self,
        payload: Union[dict[str, Any], ObservationCreate, ObservationRegisterRequest],
    ) -> Observation:
        """Register, normalize, quality-check, and persist a single observation."""
        normalized = self.normalize_metadata(payload)

        # Create transient Observation instance
        obs_id = normalized.get("id") or uuid.uuid4()
        obs = Observation(
            id=obs_id,
            field_id=normalized.get("field_id"),
            simulation_run_id=normalized.get("simulation_run_id"),
            batch_id=normalized.get("batch_id"),
            timestamp=normalized["timestamp"],
            variable_name=normalized["variable_name"],
            units=normalized["units"],
            value=float(normalized["value"]),
            uncertainty=float(normalized["uncertainty"]),
            source=normalized["source"],
            provider_name=normalized["provider_name"],
            latitude=normalized.get("latitude"),
            longitude=normalized.get("longitude"),
            quality_score=normalized.get("quality_score"),
            cloud_cover=normalized.get("cloud_cover"),
            status=ObservationStatus.VALID,  # default
            raw_payload=normalized["raw_payload"],
            notes=normalized.get("notes"),
        )

        # Evaluate initial QC status if status was not explicitly passed as REJECTED/OUTLIER
        explicit_status = normalized.get("status")
        if explicit_status in (ObservationStatus.REJECTED.value, ObservationStatus.OUTLIER.value, ObservationStatus.MISSING.value, ObservationStatus.REJECTED, ObservationStatus.OUTLIER, ObservationStatus.MISSING):
            obs.status = ObservationStatus(explicit_status)
        else:
            qc_res = self.qc_service.evaluate_observation(obs)
            obs.status = qc_res.status

        # Delegate persistence to repository
        return self.repository.save_observation(obs)

    def register_batch(
        self,
        payloads: list[Union[dict[str, Any], ObservationCreate, ObservationRegisterRequest]],
    ) -> list[Observation]:
        """Register, normalize, quality-check, and bulk-persist multiple observations."""
        if not payloads:
            return []

        observations: list[Observation] = []
        for p in payloads:
            normalized = self.normalize_metadata(p)
            obs_id = normalized.get("id") or uuid.uuid4()
            obs = Observation(
                id=obs_id,
                field_id=normalized.get("field_id"),
                simulation_run_id=normalized.get("simulation_run_id"),
                batch_id=normalized.get("batch_id"),
                timestamp=normalized["timestamp"],
                variable_name=normalized["variable_name"],
                units=normalized["units"],
                value=float(normalized["value"]),
                uncertainty=float(normalized["uncertainty"]),
                source=normalized["source"],
                provider_name=normalized["provider_name"],
                latitude=normalized.get("latitude"),
                longitude=normalized.get("longitude"),
                quality_score=normalized.get("quality_score"),
                cloud_cover=normalized.get("cloud_cover"),
                status=ObservationStatus.VALID,
                raw_payload=normalized["raw_payload"],
                notes=normalized.get("notes"),
            )
            explicit_status = normalized.get("status")
            if explicit_status in (ObservationStatus.REJECTED.value, ObservationStatus.OUTLIER.value, ObservationStatus.MISSING.value, ObservationStatus.REJECTED, ObservationStatus.OUTLIER, ObservationStatus.MISSING):
                obs.status = ObservationStatus(explicit_status)
            else:
                qc_res = self.qc_service.evaluate_observation(obs)
                obs.status = qc_res.status
            observations.append(obs)

        return self.repository.save_many(observations)
