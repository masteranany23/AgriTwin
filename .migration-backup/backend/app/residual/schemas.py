"""
backend/app/residual/schemas.py
================================

Pydantic schemas and dataclasses for the ResidualModel abstraction.
Defines metadata, prediction outputs, and corrected yield responses.
"""

import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Metadata describing a residual model artifact."""
    model_id: str = Field(..., description="Unique identifier of the model artifact.")
    name: str = Field(..., description="Human-readable model name.")
    version: str = Field(..., description="Model version string (semver).")
    description: str = Field(..., description="Description of the model and training target.")
    validated: bool = Field(False, description="Whether this model has undergone scientific validation.")
    supported_crops: List[str] = Field(default_factory=lambda: ["*"], description="Crops supported by this model.")
    supported_regions: List[str] = Field(default_factory=lambda: ["*"], description="Regions/countries supported by this model.")
    created_at: Optional[datetime.datetime] = Field(None, description="Model registration / creation timestamp.")


class ResidualPrediction(BaseModel):
    """Output of a residual prediction target = (observed_yield - assimilated_WOFOST_yield)."""
    residual: float = Field(0.0, description="Predicted residual yield delta [kg/ha].")
    uncertainty: float = Field(0.0, description="Estimated residual prediction standard deviation [kg/ha].")
    model_id: str = Field(..., description="ID of the model that generated this prediction.")
    is_fallback: bool = Field(True, description="True if no validated model was used and residual is 0.")


class CorrectedYieldPrediction(BaseModel):
    """Combined output containing original assimilated yield and applied residual correction."""
    assimilated_yield_kg_ha: float = Field(..., description="Original assimilated WOFOST yield prediction [kg/ha].")
    residual_correction_kg_ha: float = Field(0.0, description="Applied residual correction delta [kg/ha].")
    corrected_yield_kg_ha: float = Field(..., description="Final corrected yield prediction = assimilated_yield + residual.")
    residual_uncertainty_kg_ha: float = Field(0.0, description="Uncertainty (std) of the residual correction [kg/ha].")
    model_id: str = Field(..., description="Model ID used for the correction.")
    model_version: str = Field(..., description="Model version string.")
    is_validated_correction: bool = Field(False, description="True if correction came from a validated model.")
