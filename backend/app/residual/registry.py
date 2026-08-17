"""
backend/app/residual/registry.py
=================================

Residual Model Registry for AgriTwin
------------------------------------
Manages registration and lookup of `ResidualModel` instances.
Returns `NoResidualModel()` when no validated model matches the crop/region context.
"""

import logging
from typing import Dict, List, Optional

from backend.app.residual.base import ResidualModel
from backend.app.residual.no_residual import NoResidualModel
from backend.app.residual.schemas import ModelMetadata

logger = logging.getLogger(__name__)


class ResidualModelRegistry:
    """Registry managing available residual models."""

    def __init__(self):
        self._models: Dict[str, ResidualModel] = {}
        self._default_fallback = NoResidualModel()
        # Register default fallback model
        self.register_model(self._default_fallback)

    def register_model(self, model: ResidualModel) -> None:
        """Register a residual model instance.

        Args:
            model: Instance implementing ResidualModel interface.
        """
        model_id = model.metadata.model_id
        self._models[model_id] = model
        logger.info(f"Registered ResidualModel: {model_id} ({model.metadata.name})")

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the registry by ID."""
        if model_id in self._models and model_id != self._default_fallback.metadata.model_id:
            del self._models[model_id]
            logger.info(f"Unregistered ResidualModel: {model_id}")

    def reset(self) -> None:
        """Reset registry to default state containing only NoResidualModel."""
        self._models = {}
        self.register_model(self._default_fallback)

    def get_model(self, crop: str, region: Optional[str] = None) -> ResidualModel:
        """Retrieve the best validated residual model for the given crop and region.

        If no validated model is available, returns `NoResidualModel()`.

        Args:
            crop: Crop name string (e.g. "wheat", "rice").
            region: Optional region or country code string.

        Returns:
            A ResidualModel instance (either a validated model or NoResidualModel fallback).
        """
        for model_id, model in self._models.items():
            if model_id == self._default_fallback.metadata.model_id:
                continue
            if model.is_applicable(crop, region) and model.is_available(crop, region):
                return model

        # Fallback to identity NoResidualModel
        return self._default_fallback

    def list_models(self) -> List[ModelMetadata]:
        """List metadata of all registered residual models."""
        return [model.metadata for model in self._models.values()]


# Module-level default global registry instance
global_residual_registry = ResidualModelRegistry()
