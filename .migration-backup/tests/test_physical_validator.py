"""
tests/test_physical_validator.py
==================================

Focused unit tests for the physical-feasibility validation layer prior to
WOFOST state injection.
"""

import datetime
from unittest.mock import MagicMock
import pytest

from backend.app.assimilation.state.state_vector import StateVector
from backend.app.assimilation.updater.physical_validator import validate_physical_feasibility
from backend.app.assimilation.updater.state_updater import StateUpdater, InjectionResult

TODAY = datetime.date(2024, 5, 1)


def _make_mock_wofost(sm0: float = 0.45, rdmsol: float = 120.0) -> MagicMock:
    """Return a mock WOFOST engine with configurable soil parameters."""
    wofost = MagicMock()
    wofost.params.soildata = {
        "SM0": sm0,
        "SMFCF": 0.30,
        "SMW": 0.10,
        "RDMSOL": rdmsol,
    }
    return wofost


def test_valid_posterior_state():
    """Verify that a physically consistent posterior state passes validation."""
    sv = StateVector(
        date=TODAY,
        lai=2.5,
        sm=0.28,
        tagp=2000.0,
        twso=500.0,
        twlv=800.0,
        twst=700.0,
        twrt=400.0,
        rftra=0.95,
        dvs=0.8,
        rd=50.0,
    )
    wofost = _make_mock_wofost()
    is_valid, violations = validate_physical_feasibility(sv, wofost)
    
    assert is_valid is True
    assert len(violations) == 0


def test_negative_biomass_and_lai():
    """Verify detection of negative values in LAI, biomass components, or SM."""
    sv = StateVector(
        date=TODAY,
        lai=-0.5,
        sm=-0.05,
        tagp=-100.0,
        twso=-10.0,
    )
    is_valid, violations = validate_physical_feasibility(sv)
    
    assert is_valid is False
    assert any("LAI" in v for v in violations)
    assert any("SM" in v for v in violations)
    assert any("TAGP" in v for v in violations)
    assert any("TWSO" in v for v in violations)


def test_rftra_range_validation():
    """Verify that RFTRA values outside [0.0, 1.0] trigger validation failure."""
    sv_high = StateVector(date=TODAY, rftra=1.2)
    is_valid, violations = validate_physical_feasibility(sv_high)
    assert is_valid is False
    assert any("RFTRA" in v for v in violations)

    sv_low = StateVector(date=TODAY, rftra=-0.1)
    is_valid, violations = validate_physical_feasibility(sv_low)
    assert is_valid is False
    assert any("RFTRA" in v for v in violations)


def test_soil_moisture_exceeds_saturation():
    """Verify that soil moisture exceeding soil pore space SM0 is flagged."""
    wofost = _make_mock_wofost(sm0=0.45)
    
    # SM = 0.50 exceeds saturation SM0 = 0.45
    sv = StateVector(date=TODAY, sm=0.50)
    is_valid, violations = validate_physical_feasibility(sv, wofost)
    
    assert is_valid is False
    assert any("exceeds soil saturation SM0" in v for v in violations)


def test_biomass_component_exceeds_tagp():
    """Verify that individual biomass component exceeding TAGP is flagged."""
    # TWSO = 1500 > TAGP = 1000
    sv = StateVector(
        date=TODAY,
        tagp=1000.0,
        twso=1500.0,
    )
    is_valid, violations = validate_physical_feasibility(sv)
    
    assert is_valid is False
    assert any("TWSO" in v and "exceeds" in v for v in violations)


def test_sum_of_biomass_components_exceeds_tagp():
    """Verify that sum of (TWSO + TWLV + TWST) exceeding TAGP is flagged."""
    # TWSO(500) + TWLV(400) + TWST(400) = 1300 > TAGP(1000)
    sv = StateVector(
        date=TODAY,
        tagp=1000.0,
        twso=500.0,
        twlv=400.0,
        twst=400.0,
    )
    is_valid, violations = validate_physical_feasibility(sv)
    
    assert is_valid is False
    assert any("Sum of aboveground biomass components" in v for v in violations)


def test_root_depth_exceeds_rdmsol():
    """Verify that root depth exceeding max rootable soil depth (RDMSOL) is flagged."""
    wofost = _make_mock_wofost(rdmsol=100.0)
    sv = StateVector(date=TODAY, rd=130.0)
    
    is_valid, violations = validate_physical_feasibility(sv, wofost)
    assert is_valid is False
    assert any("exceeds maximum rootable soil depth RDMSOL" in v for v in violations)


def test_pre_emergence_biomass_flagged():
    """Verify that pre-emergence stage (DVS < 0) with non-zero LAI/biomass is flagged."""
    sv = StateVector(date=TODAY, dvs=-0.2, lai=1.0, twrt=50.0)
    is_valid, violations = validate_physical_feasibility(sv)
    
    assert is_valid is False
    assert any("Pre-emergence" in v for v in violations)


def test_state_updater_integration_valid_and_invalid():
    """Verify StateUpdater.inject records physical validation status in InjectionResult."""
    updater = StateUpdater(verify=False)
    wofost = _make_mock_wofost()
    
    # Valid state
    sv_valid = StateVector(date=TODAY, lai=2.0, sm=0.25)
    res_valid = updater.inject(wofost, sv_valid)
    assert res_valid.is_physically_valid is True
    assert len(res_valid.validation_errors) == 0
    
    # Physically invalid state (negative LAI & TWSO > TAGP)
    sv_invalid = StateVector(date=TODAY, lai=-1.0, tagp=500.0, twso=800.0)
    res_invalid = updater.inject(wofost, sv_invalid)
    assert res_invalid.is_physically_valid is False
    assert len(res_invalid.validation_errors) >= 2
    # Verify defensive bounds clamping still applied for setting variables safely
    assert res_invalid.injected["lai"] == 0.0
