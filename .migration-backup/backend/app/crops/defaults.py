"""
backend/app/crops/defaults.py
==============================

Pre-populated default CropConfig definitions for standard WOFOST crops in AgriTwin.
Derived strictly from existing PCSE YAML crop parameter files and agromanagement rules.
"""

from typing import Dict
from backend.app.crops.schemas import (
    CalibrationMetadata,
    CropConfig,
    ObservationMapping,
    PhenologyConfig,
    ResidualModelMetadata,
    WofostParamDefaults,
)


def get_default_crop_configs() -> Dict[str, CropConfig]:
    """Build dictionary of pre-configured CropConfig objects for standard crops."""
    configs = {}

    # 1. Wheat
    configs["wheat"] = CropConfig(
        crop_name="wheat",
        display_name="Wheat",
        wofost=WofostParamDefaults(
            default_variety="Winter_wheat_101",
            crop_file="wheat.yaml",
            tsum1=1050.0,
            tsum2=1000.0,
            slatb_default=0.002,
            span=35.0,
            tdwi=210.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="sowing",
            crop_end_type="harvest",
            dvsi=0.0,
            dvs_emergence=0.0,
            dvs_anthesis=1.0,
            dvs_maturity=2.0,
            max_duration=300,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="NDVI", wofost_variable="LAI", conversion_factor=0.8, default_std=0.15, unit="-"),
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10, unit="m2/m2"),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05, unit="cm3/cm3"),
            ObservationMapping(sensor_variable="TAGP", wofost_variable="TAGP", conversion_factor=1.0, default_std=200.0, unit="kg/ha"),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Winter_wheat_101)",
            initial_state_variances={"LAI": 0.05, "SM": 0.002, "TAGP": 500.0, "TWSO": 0.0},
        ),
        residual_model=ResidualModelMetadata(
            residual_model_id="no_residual_v1",
            applicable_models=["no_residual_v1"],
            supports_residual_correction=True,
        ),
    )

    # 2. Rice
    configs["rice"] = CropConfig(
        crop_name="rice",
        display_name="Rice",
        wofost=WofostParamDefaults(
            default_variety="Rice_IR64",
            crop_file="rice.yaml",
            tsum1=850.0,
            tsum2=750.0,
            slatb_default=0.0024,
            span=30.0,
            tdwi=150.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="emergence",  # Transplanted rice starts at emergence to avoid TSUMEM=0
            crop_end_type="harvest",
            dvsi=0.16,
            dvs_emergence=0.0,
            dvs_anthesis=1.0,
            dvs_maturity=2.0,
            max_duration=210,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="NDVI", wofost_variable="LAI", conversion_factor=0.75, default_std=0.15, unit="-"),
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10, unit="m2/m2"),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.04, unit="cm3/cm3"),
            ObservationMapping(sensor_variable="TAGP", wofost_variable="TAGP", conversion_factor=1.0, default_std=250.0, unit="kg/ha"),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Rice_IR64)",
            initial_state_variances={"LAI": 0.06, "SM": 0.002, "TAGP": 600.0, "TWSO": 0.0},
        ),
        residual_model=ResidualModelMetadata(
            residual_model_id="no_residual_v1",
            applicable_models=["no_residual_v1"],
            supports_residual_correction=True,
        ),
    )

    # 3. Maize
    configs["maize"] = CropConfig(
        crop_name="maize",
        display_name="Maize",
        wofost=WofostParamDefaults(
            default_variety="Grain_maize_201",
            crop_file="maize.yaml",
            tsum1=720.0,
            tsum2=800.0,
            slatb_default=0.0016,
            span=40.0,
            tdwi=100.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="sowing",
            crop_end_type="harvest",
            dvsi=0.0,
            dvs_emergence=0.0,
            dvs_anthesis=1.0,
            dvs_maturity=2.0,
            max_duration=200,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="NDVI", wofost_variable="LAI", conversion_factor=0.85, default_std=0.15, unit="-"),
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10, unit="m2/m2"),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05, unit="cm3/cm3"),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Grain_maize_201)",
            initial_state_variances={"LAI": 0.05, "SM": 0.002, "TAGP": 500.0},
        ),
        residual_model=ResidualModelMetadata(
            residual_model_id="no_residual_v1",
            applicable_models=["no_residual_v1"],
            supports_residual_correction=True,
        ),
    )

    # 4. Barley
    configs["barley"] = CropConfig(
        crop_name="barley",
        display_name="Barley",
        wofost=WofostParamDefaults(
            default_variety="Spring_barley_301",
            crop_file="barley.yaml",
            tsum1=800.0,
            tsum2=750.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="sowing",
            crop_end_type="harvest",
            dvsi=0.0,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Spring_barley_301)",
        ),
        residual_model=ResidualModelMetadata(residual_model_id="no_residual_v1"),
    )

    # 5. Potato
    configs["potato"] = CropConfig(
        crop_name="potato",
        display_name="Potato",
        wofost=WofostParamDefaults(
            default_variety="Potato_701",
            crop_file="potato.yaml",
            tsum1=600.0,
            tsum2=900.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="emergence",
            crop_end_type="harvest",
            dvsi=0.0,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Potato_701)",
        ),
        residual_model=ResidualModelMetadata(residual_model_id="no_residual_v1"),
    )

    # 6. Soybean
    configs["soybean"] = CropConfig(
        crop_name="soybean",
        display_name="Soybean",
        wofost=WofostParamDefaults(
            default_variety="Soybean_901",
            crop_file="soybean.yaml",
            tsum1=750.0,
            tsum2=850.0,
        ),
        phenology=PhenologyConfig(
            crop_start_type="sowing",
            crop_end_type="harvest",
            dvsi=0.0,
        ),
        observation_mappings=[
            ObservationMapping(sensor_variable="LAI", wofost_variable="LAI", conversion_factor=1.0, default_std=0.10),
            ObservationMapping(sensor_variable="SM", wofost_variable="SM", conversion_factor=1.0, default_std=0.05),
        ],
        calibration=CalibrationMetadata(
            status="BASELINE",
            calibration_region="GLOBAL",
            source="WOFOST 7.2 Standard Parameters (Soybean_901)",
        ),
        residual_model=ResidualModelMetadata(residual_model_id="no_residual_v1"),
    )

    return configs
