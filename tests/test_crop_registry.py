"""
tests/test_crop_registry.py
============================

Unit and regression tests for AgriTwin CropRegistry and CropConfig:
- Verification of WOFOST parameters, phenology configs, observation mappings, calibration, and residual metadata
- Auto-discovery of secondary crops from WOFOST YAML parameter database
- Numerical regression tests comparing direct simulation vs CropRegistry execution
"""

import datetime as dt
import pytest
from pcse.models import Wofost72_WLP_FD
from pcse.base import ParameterProvider

from backend.app.crops import CropConfig, CropRegistry, global_crop_registry
from backend.app.simulation.engine import run_simulation
from backend.app.simulation.weather_provider import create_weather_provider
from backend.app.simulation.soil_provider import create_soil_params
from backend.app.simulation.site_provider import create_site_params


def test_crop_registry_prepopulated_crops():
    """Verify prepopulated crops exist and have correct metadata."""
    registry = CropRegistry()

    assert registry.has_crop("wheat") is True
    assert registry.has_crop("rice") is True
    assert registry.has_crop("maize") is True

    wheat = registry.get_crop("wheat")
    assert isinstance(wheat, CropConfig)
    assert wheat.wofost.default_variety == "Winter_wheat_101"
    assert wheat.phenology.crop_start_type == "sowing"
    assert wheat.phenology.dvsi == 0.0
    assert len(wheat.observation_mappings) > 0
    assert wheat.calibration.status == "BASELINE"
    assert wheat.residual_model.residual_model_id == "no_residual_v1"

    rice = registry.get_crop("rice")
    assert rice.wofost.default_variety == "Rice_IR64"
    assert rice.phenology.crop_start_type == "emergence"  # Transplanted mode
    assert rice.phenology.dvsi > 0.0


def test_crop_registry_auto_discovery():
    """Verify auto-discovery for crops present in PCSE YAML files but not pre-populated."""
    registry = CropRegistry()

    # 'sunflower' is available in PCSE files
    assert registry.has_crop("sunflower") is True
    sunflower = registry.get_crop("sunflower")
    assert sunflower.crop_name == "sunflower"
    assert sunflower.wofost.default_variety == "Sunflower_1101"
    assert sunflower.phenology.crop_start_type == "sowing"


def test_crop_registry_nonexistent_crop_raises_keyerror():
    """Verify querying a non-existent crop raises KeyError."""
    registry = CropRegistry()
    assert registry.has_crop("nonexistent_fake_crop") is False
    with pytest.raises(KeyError):
        registry.get_crop("nonexistent_fake_crop")


def test_simulation_numerical_regression_wheat():
    """Regression test: verify identical WOFOST outputs between direct run_simulation and CropRegistry execution."""
    sow_date = dt.date(2020, 10, 15)
    harvest_date = dt.date(2021, 7, 30)
    campaign_start = sow_date - dt.timedelta(days=14)

    # 1. Direct simulation
    direct_res = run_simulation(
        crop_name="wheat",
        variety_name="Winter_wheat_101",
        sow_date=sow_date,
        harvest_date=harvest_date,
        use_nasa_weather=False,
    )

    # 2. Registry-based execution
    wheat_config = global_crop_registry.get_crop("wheat")

    wdp = create_weather_provider(
        latitude=52.0,
        longitude=5.5,
        elevation=10.0,
        start_year=campaign_start.year,
        end_year=harvest_date.year,
        start_date=campaign_start,
        end_date=harvest_date,
        use_nasa=False,
    )
    cropd = wheat_config.create_crop_provider(variety_name="Winter_wheat_101")
    soildata = create_soil_params()
    sitedata = create_site_params(wav=10.0)

    params = ParameterProvider(cropdata=cropd, soildata=soildata, sitedata=sitedata)
    agro = wheat_config.build_agromanagement(
        sow_date=sow_date,
        harvest_date=harvest_date,
        variety_name="Winter_wheat_101",
        campaign_start_date=campaign_start,
    )

    wofost = Wofost72_WLP_FD(params, wdp, agro)
    wofost.run_till_terminate()

    registry_raw_output = wofost.get_output()

    # Compare step counts
    assert len(direct_res.raw_output) == len(registry_raw_output)

    # Compare daily numerical states
    for d1, d2 in zip(direct_res.raw_output, registry_raw_output):
        assert d1["day"] == d2["day"]
        assert d1["LAI"] == pytest.approx(d2["LAI"], abs=1e-6)
        assert d1["TAGP"] == pytest.approx(d2["TAGP"], abs=1e-6)
        assert d1["TWSO"] == pytest.approx(d2["TWSO"], abs=1e-6)
        assert d1["SM"] == pytest.approx(d2["SM"], abs=1e-6)
        assert d1["DVS"] == pytest.approx(d2["DVS"], abs=1e-6)


def test_simulation_numerical_regression_rice():
    """Regression test: verify identical WOFOST outputs for transplanted rice."""
    sow_date = dt.date(2021, 5, 1)
    harvest_date = dt.date(2021, 9, 30)
    campaign_start = sow_date - dt.timedelta(days=14)

    # 1. Direct simulation
    direct_res = run_simulation(
        crop_name="rice",
        variety_name="Rice_IR64",
        sow_date=sow_date,
        harvest_date=harvest_date,
        use_nasa_weather=False,
    )

    # 2. Registry-based execution
    rice_config = global_crop_registry.get_crop("rice")
    assert rice_config.phenology.crop_start_type == "emergence"

    wdp = create_weather_provider(
        latitude=52.0,
        longitude=5.5,
        elevation=10.0,
        start_year=campaign_start.year,
        end_year=harvest_date.year,
        start_date=campaign_start,
        end_date=harvest_date,
        use_nasa=False,
    )
    cropd = rice_config.create_crop_provider(variety_name="Rice_IR64")
    soildata = create_soil_params()
    sitedata = create_site_params(wav=10.0)

    params = ParameterProvider(cropdata=cropd, soildata=soildata, sitedata=sitedata)
    agro = rice_config.build_agromanagement(
        sow_date=sow_date,
        harvest_date=harvest_date,
        variety_name="Rice_IR64",
        campaign_start_date=campaign_start,
    )

    wofost = Wofost72_WLP_FD(params, wdp, agro)
    wofost.run_till_terminate()

    registry_raw_output = wofost.get_output()

    assert len(direct_res.raw_output) == len(registry_raw_output)

    for d1, d2 in zip(direct_res.raw_output, registry_raw_output):
        assert d1["day"] == d2["day"]
        assert d1["LAI"] == pytest.approx(d2["LAI"], abs=1e-6)
        assert d1["TAGP"] == pytest.approx(d2["TAGP"], abs=1e-6)
        assert d1["TWSO"] == pytest.approx(d2["TWSO"], abs=1e-6)
        assert d1["DVS"] == pytest.approx(d2["DVS"], abs=1e-6)
