"""
backend/app/crops/schemas.py
=============================

Pydantic schemas for the AgriTwin Crop Model Configuration framework.
Defines crop-specific WOFOST parameters, phenology, observation mappings,
calibration metadata, and residual model metadata.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WofostParamDefaults(BaseModel):
    """WOFOST crop parameter defaults and file metadata."""
    default_variety: str = Field(..., description="Default variety key for this crop.")
    crop_file: str = Field(..., description="WOFOST parameter filename (e.g. wheat.yaml).")
    tsum1: Optional[float] = Field(None, description="Temperature sum emergence to anthesis [deg C d].")
    tsum2: Optional[float] = Field(None, description="Temperature sum anthesis to maturity [deg C d].")
    slatb_default: Optional[float] = Field(None, description="Default specific leaf area [ha kg-1].")
    span: Optional[float] = Field(None, description="Leaf life span at 35 deg C [d].")
    tdwi: Optional[float] = Field(None, description="Initial total dry weight [kg ha-1].")


class PhenologyConfig(BaseModel):
    """Phenological stage definitions and simulation start/end rules."""
    crop_start_type: str = Field("sowing", description="'sowing' or 'emergence' (transplanted crops use 'emergence').")
    crop_end_type: str = Field("harvest", description="'harvest', 'maturity', or 'earliest'.")
    dvsi: float = Field(0.0, description="Initial development stage (DVSI > 0 for transplanted crops).")
    dvs_emergence: float = Field(0.0, description="DVS landmark at emergence.")
    dvs_anthesis: float = Field(1.0, description="DVS landmark at anthesis / flowering.")
    dvs_maturity: float = Field(2.0, description="DVS landmark at maturity.")
    max_duration: int = Field(300, description="Maximum crop growth duration in days.")


class ObservationMapping(BaseModel):
    """Mapping between external sensor/satellite variables and WOFOST state variables."""
    sensor_variable: str = Field(..., description="Sensor or observation variable name (e.g. 'NDVI', 'LAI', 'SM').")
    wofost_variable: str = Field(..., description="Corresponding WOFOST state variable (e.g. 'LAI', 'SM', 'TAGP').")
    conversion_factor: float = Field(1.0, description="Linear scaling factor to map sensor value to WOFOST unit.")
    default_std: float = Field(0.1, description="Default observation error standard deviation.")
    unit: str = Field("", description="Observation measurement unit string.")


class CalibrationMetadata(BaseModel):
    """Calibration status, region, and ensemble perturbation parameters."""
    status: str = Field("BASELINE", description="Calibration status ('CALIBRATED', 'BASELINE', 'UNTESTED').")
    calibration_region: str = Field("GLOBAL", description="Geographic region/dataset used for calibration.")
    source: str = Field("WOFOST 7.2 Standard Parameters", description="Data source or reference for parameters.")
    initial_state_variances: Dict[str, float] = Field(
        default_factory=dict,
        description="Default initial variance/uncertainty scales for EnKF perturbation.",
    )


class ResidualModelMetadata(BaseModel):
    """Metadata connecting crop configuration to optional residual yield models."""
    residual_model_id: str = Field("no_residual_v1", description="ID of active residual model artifact.")
    applicable_models: List[str] = Field(
        default_factory=lambda: ["no_residual_v1"],
        description="List of residual model IDs applicable to this crop.",
    )
    supports_residual_correction: bool = Field(True, description="Whether this crop supports post-assimilation residual correction.")


class CropConfig(BaseModel):
    """Complete crop model configuration container."""
    crop_name: str = Field(..., description="Lowercase PCSE crop identifier (e.g. 'wheat', 'rice').")
    display_name: str = Field(..., description="Human-readable crop name.")
    wofost: WofostParamDefaults = Field(..., description="WOFOST parameter defaults.")
    phenology: PhenologyConfig = Field(..., description="Phenological configuration.")
    observation_mappings: List[ObservationMapping] = Field(
        default_factory=list,
        description="Sensor-to-WOFOST variable mappings.",
    )
    calibration: CalibrationMetadata = Field(
        default_factory=CalibrationMetadata,
        description="Calibration metadata and state variance settings.",
    )
    residual_model: ResidualModelMetadata = Field(
        default_factory=ResidualModelMetadata,
        description="Residual model linkage metadata.",
    )

    def get_observation_mapping(self, sensor_variable: str) -> Optional[ObservationMapping]:
        """Find observation mapping for a specific sensor variable name."""
        var_lower = sensor_variable.lower()
        for mapping in self.observation_mappings:
            if mapping.sensor_variable.lower() == var_lower or mapping.wofost_variable.lower() == var_lower:
                return mapping
        return None
