"""
tests/test_residual_model.py
=============================

Unit tests for the AgriTwin ResidualModel abstraction:
- ResidualModel ABC interface contracts
- NoResidualModel identity fallback behavior
- Availability and applicability checks
- Zero residual correction guarantee when no validated model exists
- ResidualModelRegistry registration and lookup
"""

import pytest
from typing import Optional

from backend.app.residual import (
    CorrectedYieldPrediction,
    ModelMetadata,
    NoResidualModel,
    ResidualModel,
    ResidualModelRegistry,
    ResidualPrediction,
)


class MockValidatedResidualModel(ResidualModel):
    """Test concrete implementation of a validated residual model (without ML training)."""

    def __init__(self, model_id: str = "mock_wheat_nl_v1"):
        self._metadata = ModelMetadata(
            model_id=model_id,
            name="Mock Validated Wheat Residual Model",
            version="1.1.0",
            description="Mock validated model for testing residual correction pipeline.",
            validated=True,
            supported_crops=["wheat"],
            supported_regions=["NL"],
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def is_applicable(self, crop: str, region: Optional[str] = None) -> bool:
        crop_ok = crop.lower() in [c.lower() for c in self.metadata.supported_crops]
        region_ok = region is None or region.upper() in [r.upper() for r in self.metadata.supported_regions]
        return crop_ok and region_ok

    def is_available(self, crop: str, region: Optional[str] = None) -> bool:
        return self.is_applicable(crop, region)

    def predict_residual(
        self,
        feature_vector: Optional[object] = None,
        assimilated_yield: float = 0.0,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> ResidualPrediction:
        return ResidualPrediction(
            residual=250.0,
            uncertainty=35.0,
            model_id=self.metadata.model_id,
            is_fallback=False,
        )

    def predict_uncertainty(
        self,
        feature_vector: Optional[object] = None,
        crop: Optional[str] = None,
        region: Optional[str] = None,
    ) -> float:
        return 35.0


def test_residual_model_abc_cannot_be_instantiated():
    """Verify that ResidualModel ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ResidualModel()


def test_no_residual_model_fallback_behavior():
    """Verify NoResidualModel guarantees zero residual correction and returns original yield."""
    model = NoResidualModel()

    # Metadata & Versioning
    meta = model.metadata
    assert meta.model_id == "no_residual_v1"
    assert meta.validated is False
    assert meta.version == "1.0.0"

    # Applicability & Availability
    assert model.is_applicable("wheat", "NL") is True
    assert model.is_applicable("rice", "IN") is True
    assert model.is_available("wheat", "NL") is False

    # Predict Residual
    pred = model.predict_residual(assimilated_yield=4000.0, crop="wheat", region="NL")
    assert pred.residual == 0.0
    assert pred.uncertainty == 0.0
    assert pred.is_fallback is True
    assert model.predict_uncertainty() == 0.0

    # Apply Correction: Returns assimilated WOFOST yield prediction unchanged
    assimilated_yield = 4250.5
    result = model.apply_correction(assimilated_yield=assimilated_yield, crop="wheat", region="NL")

    assert isinstance(result, CorrectedYieldPrediction)
    assert result.assimilated_yield_kg_ha == pytest.approx(assimilated_yield)
    assert result.residual_correction_kg_ha == 0.0
    assert result.corrected_yield_kg_ha == pytest.approx(assimilated_yield)
    assert result.residual_uncertainty_kg_ha == 0.0
    assert result.is_validated_correction is False


def test_validated_residual_model_correction():
    """Verify that a validated ResidualModel applies non-zero corrections accurately."""
    model = MockValidatedResidualModel()
    assert model.is_applicable("wheat", "NL") is True
    assert model.is_applicable("rice", "NL") is False
    assert model.is_available("wheat", "NL") is True

    assimilated_yield = 5000.0
    result = model.apply_correction(assimilated_yield=assimilated_yield, crop="wheat", region="NL")

    assert result.assimilated_yield_kg_ha == 5000.0
    assert result.residual_correction_kg_ha == 250.0
    assert result.corrected_yield_kg_ha == 5250.0
    assert result.residual_uncertainty_kg_ha == 35.0
    assert result.is_validated_correction is True


def test_residual_model_registry_resolution():
    """Verify ResidualModelRegistry fallback and model resolution."""
    registry = ResidualModelRegistry()

    # 1. When no custom validated model is registered, fallback to NoResidualModel
    default_model = registry.get_model(crop="wheat", region="NL")
    assert isinstance(default_model, NoResidualModel)
    assert default_model.is_available("wheat", "NL") is False

    # 2. Register mock validated model for wheat / NL
    mock_model = MockValidatedResidualModel()
    registry.register_model(mock_model)

    # Lookup matching crop & region -> returns mock_model
    resolved = registry.get_model(crop="wheat", region="NL")
    assert resolved.metadata.model_id == "mock_wheat_nl_v1"

    # Lookup non-matching crop (rice) -> falls back to NoResidualModel
    fallback_for_rice = registry.get_model(crop="rice", region="NL")
    assert isinstance(fallback_for_rice, NoResidualModel)

    # Unregister model
    registry.unregister_model("mock_wheat_nl_v1")
    assert isinstance(registry.get_model(crop="wheat", region="NL"), NoResidualModel)
