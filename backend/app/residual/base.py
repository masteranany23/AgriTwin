"""
backend/app/residual/base.py
=============================

Abstract Base Class for Residual Yield Models in AgriTwin
---------------------------------------------------------
Defines the contract for yield residual models:
    residual_target = observed_yield - assimilated_WOFOST_yield

Principles:
1. Optional Abstraction: Plugs into prediction pipelines to apply post-assimilation residual adjustments.
2. Leakage & Fallback Safety: When no validated model exists, `is_available()` returns False
   and `apply_correction()` returns the assimilated WOFOST prediction unchanged.
3. No Fabricated Corrections: Zero residual correction is returned unless a validated model is active.
"""

from abc import ABC, abstractmethod
from typing import Optional

from backend.app.residual.schemas import (
    CorrectedYieldPrediction,
    ModelMetadata,
    ResidualPrediction,
)


class ResidualModel(ABC):
    """Abstract interface for optional residual yield models."""

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return model metadata, version, name, and applicability attributes."""
        ...

    @abstractmethod
    def is_applicable(self, crop: str, region: Optional[str] = None) -> bool:
        """Check whether this model supports the specified crop and region.

        Args:
            crop: Crop name string (e.g. "wheat", "rice").
            region: Optional region or country code string (e.g. "NL", "IN").

        Returns:
            True if the model is designed to support the given crop and region.
        """
        ...

    @abstractmethod
    def is_available(self, crop: str, region: Optional[str] = None) -> bool:
        """Check whether a validated model artifact is loaded and available for inference.

        Args:
            crop: Crop name string.
            region: Optional region code string.

        Returns:
            True if a validated model is active and ready for inference, False otherwise.
        """
        ...

    @abstractmethod
    def predict_residual(
        self,
        feature_vector: Optional[object] = None,
        assimilated_yield: float = 0.0,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> ResidualPrediction:
        """Predict the yield residual target = (observed_yield - assimilated_WOFOST_yield).

        Args:
            feature_vector: Optional tabular feature vector (e.g. from FeatureEngine).
            assimilated_yield: Assimilated WOFOST yield estimate [kg/ha].
            crop: Crop name string.
            region: Optional region code string.

        Returns:
            ResidualPrediction object containing predicted residual and uncertainty.
        """
        ...

    @abstractmethod
    def predict_uncertainty(
        self,
        feature_vector: Optional[object] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> float:
        """Estimate uncertainty (std) of the residual prediction [kg/ha].

        Args:
            feature_vector: Optional tabular feature vector.
            crop: Crop name string.
            region: Optional region code string.

        Returns:
            Uncertainty standard deviation in kg/ha.
        """
        ...

    def apply_correction(
        self,
        assimilated_yield: float,
        feature_vector: Optional[object] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> CorrectedYieldPrediction:
        """Apply residual correction to the assimilated WOFOST yield prediction.

        If the model is not available or not validated, returns the assimilated WOFOST
        yield prediction unchanged with residual_correction = 0.0.

        Args:
            assimilated_yield: Assimilated WOFOST yield prediction [kg/ha].
            feature_vector: Optional feature vector.
            crop: Crop name string.
            region: Optional region code string.

        Returns:
            CorrectedYieldPrediction containing original yield, correction, and corrected yield.
        """
        if not self.is_available(crop=crop or "*", region=region):
            return CorrectedYieldPrediction(
                assimilated_yield_kg_ha=assimilated_yield,
                residual_correction_kg_ha=0.0,
                corrected_yield_kg_ha=assimilated_yield,
                residual_uncertainty_kg_ha=0.0,
                model_id=self.metadata.model_id,
                model_version=self.metadata.version,
                is_validated_correction=False,
            )

        pred = self.predict_residual(
            feature_vector=feature_vector,
            assimilated_yield=assimilated_yield,
            crop=crop,
            region=region,
        )

        return CorrectedYieldPrediction(
            assimilated_yield_kg_ha=assimilated_yield,
            residual_correction_kg_ha=pred.residual,
            corrected_yield_kg_ha=assimilated_yield + pred.residual,
            residual_uncertainty_kg_ha=pred.uncertainty,
            model_id=self.metadata.model_id,
            model_version=self.metadata.version,
            is_validated_correction=self.metadata.validated and not pred.is_fallback,
        )
