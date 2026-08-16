"""
backend/app/residual/__init__.py
================================

ResidualModel Abstraction Package for AgriTwin
----------------------------------------------
Exports ResidualModel ABC, NoResidualModel, ResidualModelRegistry, and schemas.
"""

from backend.app.residual.base import ResidualModel
from backend.app.residual.no_residual import NoResidualModel
from backend.app.residual.registry import ResidualModelRegistry, global_residual_registry
from backend.app.residual.schemas import (
    CorrectedYieldPrediction,
    ModelMetadata,
    ResidualPrediction,
)

__all__ = [
    "ResidualModel",
    "NoResidualModel",
    "ResidualModelRegistry",
    "global_residual_registry",
    "ModelMetadata",
    "ResidualPrediction",
    "CorrectedYieldPrediction",
]
