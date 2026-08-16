"""
backend/app/crops/__init__.py
==============================

Crop Configuration & Registry Package for AgriTwin
--------------------------------------------------
Exports CropConfig, CropRegistry, global_crop_registry, and schema dataclasses.
"""

from backend.app.crops.schemas import (
    CalibrationMetadata,
    CropConfig,
    ObservationMapping,
    PhenologyConfig,
    ResidualModelMetadata,
    WofostParamDefaults,
)
from backend.app.crops.registry import CropRegistry, global_crop_registry

__all__ = [
    "CropConfig",
    "CropRegistry",
    "global_crop_registry",
    "WofostParamDefaults",
    "PhenologyConfig",
    "ObservationMapping",
    "CalibrationMetadata",
    "ResidualModelMetadata",
]
