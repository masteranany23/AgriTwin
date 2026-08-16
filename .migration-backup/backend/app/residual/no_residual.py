"""
backend/app/residual/no_residual.py
===================================

Default NoResidualModel Implementation for AgriTwin
----------------------------------------------------
Acts as the identity fallback model when no validated ML residual model exists.

Guarantees:
- `is_available()` returns False.
- `predict_residual()` returns 0.0 residual correction.
- `apply_correction()` leaves the assimilated WOFOST prediction unchanged.
- Never fabricates residual corrections.
"""

from typing import Optional

from backend.app.residual.base import ResidualModel
from backend.app.residual.schemas import ModelMetadata, ResidualPrediction


class NoResidualModel(ResidualModel):
    """Default fallback residual model returning zero correction."""

    def __init__(self):
        self._metadata = ModelMetadata(
            model_id="no_residual_v1",
            name="No Residual Model (Identity Fallback)",
            version="1.0.0",
            description="Default identity fallback model returning zero residual yield correction.",
            validated=False,
            supported_crops=["*"],
            supported_regions=["*"],
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def is_applicable(self, crop: str, region: Optional[str] = None) -> bool:
        """Supported across all crops and regions as generic identity fallback."""
        return True

    def is_available(self, crop: str, region: Optional[str] = None) -> bool:
        """Returns False indicating no validated ML model artifact is loaded."""
        return False

    def predict_residual(
        self,
        feature_vector: Optional[object] = None,
        assimilated_yield: float = 0.0,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> ResidualPrediction:
        """Always returns zero residual correction."""
        return ResidualPrediction(
            residual=0.0,
            uncertainty=0.0,
            model_id=self.metadata.model_id,
            is_fallback=True,
        )

    def predict_uncertainty(
        self,
        feature_vector: Optional[object] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> float:
        """Returns zero residual uncertainty."""
        return 0.0
