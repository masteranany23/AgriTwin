"""
tests/test_ensemble_simulation_fixes.py — Regression tests for Campaign Window & Soil Ensemble Constraints
========================================================================================================

Tests:
  1. Enforce canonical campaign window: start = sowing_date - 14 days, end = harvest_date across
     simulation engine, weather providers, and ensemble manager.
  2. Enforce soil moisture physical constraint SMW < SMFCF < SM0 across all ensemble members.
  3. Verify fixed-seed reproducibility for ensemble parameter perturbation.
"""

import datetime as dt
import random
import pytest

from backend.app.simulation.engine import run_simulation
from backend.app.simulation.weather_provider import create_weather_provider, SyntheticWeatherProvider
from backend.app.assimilation.ensemble.ensemble_manager import EnsembleManager
from backend.app.simulation.soil_provider import create_soil_params, _validate_soil_moisture_ordering


class TestCanonicalCampaignWindow:
    """Verify that sowing_date - 14 days to harvest_date is enforced consistently."""

    def test_synthetic_weather_provider_respects_explicit_dates(self):
        sow_date = dt.date(2023, 5, 10)
        harvest_date = dt.date(2023, 10, 15)
        campaign_start = sow_date - dt.timedelta(days=14)

        wdp = SyntheticWeatherProvider(
            latitude=52.0,
            longitude=5.5,
            start_date=campaign_start,
            end_date=harvest_date,
        )
        assert wdp.first_date == campaign_start
        assert wdp.last_date == harvest_date

    def test_create_weather_provider_respects_explicit_dates(self):
        sow_date = dt.date(2022, 11, 1)
        harvest_date = dt.date(2023, 6, 30)
        campaign_start = sow_date - dt.timedelta(days=14)

        wdp = create_weather_provider(
            latitude=28.6,
            longitude=77.2,
            start_date=campaign_start,
            end_date=harvest_date,
            use_nasa=False,
        )
        assert wdp.first_date == campaign_start
        assert wdp.last_date == harvest_date

    def test_ensemble_manager_campaign_window(self):
        sow_date = dt.date(2021, 4, 1)
        harvest_date = dt.date(2021, 9, 20)
        campaign_start = sow_date - dt.timedelta(days=14)

        mgr = EnsembleManager(
            crop_name="wheat",
            variety_name="Winter_wheat_101",
            sow_date=sow_date,
            harvest_date=harvest_date,
        )
        assert mgr.base_wdp.first_date == campaign_start
        assert mgr.base_wdp.last_date == harvest_date

    def test_engine_run_simulation_uses_canonical_window(self):
        sow_date = dt.date(2020, 10, 15)
        harvest_date = dt.date(2021, 7, 30)
        campaign_start = sow_date - dt.timedelta(days=14)

        res = run_simulation(
            crop_name="wheat",
            variety_name="Winter_wheat_101",
            sow_date=sow_date,
            harvest_date=harvest_date,
        )
        assert res.daily_output[0]["date"] == campaign_start.isoformat()
        assert res.daily_output[-1]["date"] <= harvest_date.isoformat()


class TestSoilEnsemblePerturbationConstraints:
    """Verify that every perturbed ensemble member strictly satisfies SMW < SMFCF < SM0."""

    def test_all_ensemble_members_satisfy_soil_ordering(self):
        random.seed(42)
        mgr = EnsembleManager(
            crop_name="wheat",
            variety_name="Winter_wheat_101",
            sow_date=dt.date(2020, 10, 15),
            harvest_date=dt.date(2021, 7, 30),
        )
        n_members = 120
        mgr.create_ensemble(n=n_members)

        assert len(mgr.members) == n_members
        for i, member in enumerate(mgr.members):
            smw = member.perturbed_parameters["SMW"]
            smfcf = member.perturbed_parameters["SMFCF"]
            sm0 = mgr.base_soildata["SM0"]

            # Explicit inequality check
            assert 0.01 <= smw < smfcf < sm0, (
                f"Member {i} failed soil ordering constraint: SMW={smw}, SMFCF={smfcf}, SM0={sm0}"
            )
            # Standard soil validator check should pass without error
            soildata = dict(mgr.base_soildata, SMW=smw, SMFCF=smfcf)
            _validate_soil_moisture_ordering(soildata)

    def test_soil_perturbation_with_narrow_base_sm0(self):
        """Verify soil perturbation logic even when base soil has tight margins."""
        random.seed(123)
        custom_soil = {
            "SMW": 0.12,
            "SMFCF": 0.22,
            "SM0": 0.26,  # Narrow saturation limit
            "CRAIRC": 0.05,
            "RDMSOL": 100.0,
            "K0": 10.0,
            "SOPE": 10.0,
            "KSUB": 10.0,
        }
        mgr = EnsembleManager(
            crop_name="wheat",
            variety_name="Winter_wheat_101",
            soil_params=custom_soil,
        )
        mgr.create_ensemble(n=50)

        for i, member in enumerate(mgr.members):
            smw = member.perturbed_parameters["SMW"]
            smfcf = member.perturbed_parameters["SMFCF"]
            sm0 = mgr.base_soildata["SM0"]
            assert 0.01 <= smw < smfcf < sm0, (
                f"Member {i} with custom soil failed: SMW={smw}, SMFCF={smfcf}, SM0={sm0}"
            )
            soildata = dict(mgr.base_soildata, SMW=smw, SMFCF=smfcf)
            _validate_soil_moisture_ordering(soildata)

    def test_fixed_seed_reproducibility(self):
        """Verify that initializing EnsembleManager with a fixed random seed produces exact results."""
        sow = dt.date(2020, 10, 15)
        harvest = dt.date(2021, 7, 30)

        random.seed(999)
        mgr1 = EnsembleManager(crop_name="wheat", sow_date=sow, harvest_date=harvest)
        mgr1.create_ensemble(n=10)
        smw_list1 = [m.perturbed_parameters["SMW"] for m in mgr1.members]
        smfcf_list1 = [m.perturbed_parameters["SMFCF"] for m in mgr1.members]

        random.seed(999)
        mgr2 = EnsembleManager(crop_name="wheat", sow_date=sow, harvest_date=harvest)
        mgr2.create_ensemble(n=10)
        smw_list2 = [m.perturbed_parameters["SMW"] for m in mgr2.members]
        smfcf_list2 = [m.perturbed_parameters["SMFCF"] for m in mgr2.members]

        assert smw_list1 == smw_list2
        assert smfcf_list1 == smfcf_list2
