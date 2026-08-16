"""
backend/app/features/schemas.py
===============================

Pydantic schemas and metadata contracts for the AgriTwin Feature Engine.
All feature outputs are guaranteed leakage-safe up to the evaluated target timestamp.
"""

import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WaterStressFeatures(BaseModel):
    """Cumulative water stress indicators up to as_of_date."""
    cumulative_rftra_deficit: float = Field(
        0.0,
        description="Accumulated (1.0 - RFTRA) water stress days up to as_of_date."
    )
    mean_rftra_7d: Optional[float] = Field(
        None,
        description="Mean RFTRA relative transpiration factor over recent 7-day window."
    )
    mean_rftra_14d: Optional[float] = Field(
        None,
        description="Mean RFTRA relative transpiration factor over recent 14-day window."
    )
    current_sm_deficit: Optional[float] = Field(
        None,
        description="Current volumetric soil moisture deficit below 0.35 (estimated FC)."
    )


class ThermalStressFeatures(BaseModel):
    """Thermal stress indicators up to as_of_date when supported by data."""
    cumulative_heat_days: int = Field(
        0,
        description="Count of days with Tmax > 35°C up to as_of_date."
    )
    cumulative_cold_days: int = Field(
        0,
        description="Count of days with Tmin < 5°C up to as_of_date."
    )
    mean_temp_range_7d: Optional[float] = Field(
        None,
        description="Mean daily diurnal temperature range (Tmax - Tmin) over recent 7 days."
    )
    max_tmax_7d: Optional[float] = Field(
        None,
        description="Maximum daily temperature over recent 7 days."
    )


class GrowthRateFeatures(BaseModel):
    """Dynamic biomass and canopy growth rate features."""
    delta_lai_1d: Optional[float] = Field(
        None,
        description="1-day LAI rate of change: (LAI_t - LAI_{t-1}) / 1"
    )
    delta_lai_7d: Optional[float] = Field(
        None,
        description="7-day LAI rate of change: (LAI_t - LAI_{t-7}) / 7"
    )
    delta_tagp_1d: Optional[float] = Field(
        None,
        description="1-day TAGP growth rate [kg/ha/day]: (TAGP_t - TAGP_{t-1}) / 1"
    )
    delta_tagp_7d: Optional[float] = Field(
        None,
        description="7-day TAGP growth rate [kg/ha/day]: (TAGP_t - TAGP_{t-7}) / 7"
    )


class AssimilationDiagnosticFeatures(BaseModel):
    """EnKF innovation and ensemble spread statistics up to as_of_date."""
    assimilation_cycles_count: int = Field(
        0,
        description="Total EnKF assimilation cycles executed up to as_of_date."
    )
    mean_innovation: Dict[str, float] = Field(
        default_factory=dict,
        description="Mean innovation (y - Hx) per state variable across cycles."
    )
    latest_innovation: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest innovation (y - Hx) at the most recent cycle <= as_of_date."
    )
    prior_spread: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest EnKF prior ensemble spread (std) per state variable."
    )
    posterior_spread: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest EnKF posterior ensemble spread (std) per state variable."
    )
    state_update_magnitude: Dict[str, float] = Field(
        default_factory=dict,
        description="Latest EnKF state update magnitude |x_post - x_prior| per variable."
    )


class ObservationQualityFeatures(BaseModel):
    """Observation count, quality metrics, and age statistics up to as_of_date."""
    total_obs_count: int = Field(
        0,
        description="Total observations recorded up to as_of_date."
    )
    valid_obs_count: int = Field(
        0,
        description="Total VALID observations up to as_of_date."
    )
    rejected_obs_count: int = Field(
        0,
        description="Total REJECTED/OUTLIER observations up to as_of_date."
    )
    mean_quality_score: Optional[float] = Field(
        None,
        description="Mean confidence quality score (0.0 - 1.0) of valid observations."
    )
    latest_obs_age_days: Optional[float] = Field(
        None,
        description="Age in days of the most recent valid observation relative to as_of_date."
    )
    obs_sources_present: List[str] = Field(
        default_factory=list,
        description="List of observation source types ingested up to as_of_date."
    )


class FeatureVector(BaseModel):
    """Leakage-safe feature vector assembled at forecast/assimilation timestamp as_of_date."""
    as_of_date: datetime.date = Field(
        ...,
        description="Forecast / assimilation timestamp. All features are computed strictly using data <= as_of_date."
    )
    current_dvs: Optional[float] = Field(None, description="Crop development stage DVS at as_of_date.")
    current_lai: Optional[float] = Field(None, description="Current simulated/assimilated LAI at as_of_date.")
    current_tagp: Optional[float] = Field(None, description="Current simulated/assimilated TAGP at as_of_date.")
    current_sm: Optional[float] = Field(None, description="Current volumetric soil moisture at as_of_date.")
    
    growth_rates: GrowthRateFeatures = Field(default_factory=GrowthRateFeatures)
    water_stress: WaterStressFeatures = Field(default_factory=WaterStressFeatures)
    thermal_stress: ThermalStressFeatures = Field(default_factory=ThermalStressFeatures)
    assimilation_diagnostics: AssimilationDiagnosticFeatures = Field(default_factory=AssimilationDiagnosticFeatures)
    observation_quality: ObservationQualityFeatures = Field(default_factory=ObservationQualityFeatures)

    feature_flat_dict: Dict[str, float] = Field(
        default_factory=dict,
        description="Flattened numerical feature dictionary ready for downstream modeling / tabular analytics."
    )
