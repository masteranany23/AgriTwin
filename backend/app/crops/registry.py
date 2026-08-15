"""
backend/app/crops/registry.py
==============================

CropRegistry Manager for AgriTwin
---------------------------------
Central registry managing crop model configurations (`CropConfig`).
Integrates directly with PCSE WOFOST parameter loading (`create_crop_provider`)
and agromanagement building (`build_agromanagement`).

Principles:
- Does not rewrite the WOFOST simulation engine.
- Supports auto-discovery for secondary crops available in WOFOST YAML parameter files.
- Guarantees 100% numerical identity with legacy simulation execution.
"""

import datetime as dt
import logging
from typing import Dict, List, Optional

from backend.app.crops.defaults import get_default_crop_configs
from backend.app.crops.schemas import (
    CalibrationMetadata,
    CropConfig,
    ObservationMapping,
    PhenologyConfig,
    ResidualModelMetadata,
    WofostParamDefaults,
)

logger = logging.getLogger(__name__)


class CropRegistry:
    """Central registry for crop model configurations."""

    def __init__(self):
        self._configs: Dict[str, CropConfig] = {}
        # Pre-populate with standard crop defaults
        for name, config in get_default_crop_configs().items():
            self.register_crop(config)

    def register_crop(self, config: CropConfig) -> None:
        """Register or update a CropConfig instance in the registry."""
        name_lower = config.crop_name.lower()
        self._configs[name_lower] = config
        logger.info("Registered CropConfig: %s (default variety: %s)", config.crop_name, config.wofost.default_variety)

    def has_crop(self, crop_name: str) -> bool:
        """Check if a crop is registered or available in the WOFOST parameter database."""
        name_lower = crop_name.lower()
        if name_lower in self._configs:
            return True

        # Check PCSE WOFOST parameter database
        try:
            from backend.app.simulation.crop_provider import list_available_crops
            available = list_available_crops()
            return name_lower in available
        except Exception:
            return False

    def get_crop(self, crop_name: str) -> CropConfig:
        """Retrieve the CropConfig for the given crop_name.

        If the crop is not explicitly pre-registered, attempts to auto-discover it
        from WOFOST parameter files and build a baseline CropConfig.

        Args:
            crop_name: Lowercase crop name string (e.g. 'wheat', 'rice', 'barley').

        Returns:
            CropConfig object.

        Raises:
            KeyError: If crop_name is not found in registry or WOFOST parameter database.
        """
        name_lower = crop_name.lower()

        if name_lower in self._configs:
            return self._configs[name_lower]

        # Auto-discovery fallback from PCSE parameter database
        try:
            from backend.app.simulation.crop_provider import list_available_crops
            available = list_available_crops()
            if name_lower in available:
                varieties = list(available[name_lower])
                default_var = varieties[0] if varieties else f"{name_lower}_default"
                
                # Check for transplanted start type
                from backend.app.simulation.agromanagement import get_crop_start_type
                start_type = get_crop_start_type(name_lower)

                discovered_config = CropConfig(
                    crop_name=name_lower,
                    display_name=name_lower.capitalize(),
                    wofost=WofostParamDefaults(
                        default_variety=default_var,
                        crop_file=f"{name_lower}.yaml",
                    ),
                    phenology=PhenologyConfig(
                        crop_start_type=start_type,
                        crop_end_type="harvest",
                    ),
                    observation_mappings=[
                        ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10),
                        ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05),
                    ],
                    calibration=CalibrationMetadata(
                        status="BASELINE",
                        source=f"Auto-discovered WOFOST parameters ({default_var})",
                    ),
                    residual_model=ResidualModelMetadata(residual_model_id="no_residual_v1"),
                )

                self.register_crop(discovered_config)
                logger.info("Auto-discovered and registered CropConfig for '%s'", name_lower)
                return discovered_config
        except Exception as e:
            logger.warning("Error auto-discovering crop '%s': %s", name_lower, e)

        raise KeyError(
            f"Crop '{crop_name}' not found in CropRegistry or WOFOST parameter database."
        )

    def list_crops(self) -> List[CropConfig]:
        """List all currently registered CropConfig instances."""
        return list(self._configs.values())


# Helper integration functions to attach to CropConfig or use directly
def create_crop_provider_for_config(
    config: CropConfig,
    variety_name: Optional[str] = None,
    crop_param_dir: Optional[str] = None,
):
    """Create PCSE YAMLCropDataProvider from a CropConfig instance."""
    from backend.app.simulation.crop_provider import create_crop_provider
    variety = variety_name or config.wofost.default_variety
    return create_crop_provider(
        crop_name=config.crop_name,
        variety_name=variety,
        crop_param_dir=crop_param_dir,
    )


def build_agromanagement_for_config(
    config: CropConfig,
    sow_date: dt.date,
    harvest_date: dt.date,
    variety_name: Optional[str] = None,
    campaign_start_date: Optional[dt.date] = None,
    crop_end_type: Optional[str] = None,
    max_duration: Optional[int] = None,
    irrigation_events: Optional[list] = None,
):
    """Build PCSE AgroManagement list using rules from a CropConfig instance."""
    from backend.app.simulation.agromanagement import build_agromanagement
    variety = variety_name or config.wofost.default_variety
    return build_agromanagement(
        crop_name=config.crop_name,
        variety_name=variety,
        sow_date=sow_date,
        harvest_date=harvest_date,
        campaign_start_date=campaign_start_date,
        crop_start_type=config.phenology.crop_start_type,
        crop_end_type=crop_end_type or config.phenology.crop_end_type,
        max_duration=max_duration or config.phenology.max_duration,
        irrigation_events=irrigation_events,
    )


# Attach helper methods to CropConfig class dynamically for seamless DX
CropConfig.create_crop_provider = create_crop_provider_for_config
CropConfig.build_agromanagement = build_agromanagement_for_config

# Global module-level CropRegistry singleton instance
global_crop_registry = CropRegistry()
