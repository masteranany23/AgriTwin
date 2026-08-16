"""
backend/app/assimilation/updater/physical_validator.py
=========================================================

Physical feasibility validation layer for EnKF posterior states before injection
into WOFOST.

Scientific rationale:
    EnKF linear analysis updates (x_a = x_f + K * d) can occasionally generate
    posterior states that violate fundamental physical constraints (e.g., negative
    biomass, soil moisture exceeding total soil pore space saturation, or biomass
    components exceeding total above-ground production).

    This module performs explicit, scientifically defensible physical-feasibility
    validation on state vectors prior to injection into PCSE/WOFOST.

Constraints validated:
    1. Non-negativity: LAI, TAGP, TWSO, TWLV, TWST, TWRT, SM, RD >= 0.
    2. Soil Moisture: 0 <= SM <= SM0 (soil moisture cannot exceed saturation / pore space).
    3. Relative Transpiration Factor: 0.0 <= RFTRA <= 1.0.
    4. Biomass Component Consistency:
       - Individual component weights TWSO, TWLV, TWST cannot exceed TAGP.
       - Sum of components (TWSO + TWLV + TWST) cannot exceed TAGP.
    5. Development & Root Depth:
       - 0.0 <= DVS <= 3.0.
       - 0.0 <= RD <= RDMSOL (if max root depth available).
       - Pre-emergence (DVS < 0) cannot have positive LAI or biomass.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_physical_feasibility(
    state: object,
    wofost: Optional[object] = None,
) -> tuple[bool, list[str]]:
    """Validate physical feasibility of a posterior state vector before injection.

    Args:
        state: StateVector (or duck-typed object carrying state variable attributes).
        wofost: Optional running WOFOST engine instance used to inspect active
                soil parameters (e.g., SM0 saturation, RDMSOL max root depth).

    Returns:
        tuple (is_valid: bool, violations: list[str])
        is_valid is True if zero physical violations were detected.
    """
    violations: list[str] = []

    # 1. Non-negativity checks
    non_negative_fields = {
        "lai": "LAI [m²/m²]",
        "tagp": "TAGP [kg/ha]",
        "twso": "TWSO [kg/ha]",
        "twlv": "TWLV [kg/ha]",
        "twst": "TWST [kg/ha]",
        "twrt": "TWRT [kg/ha]",
        "sm": "SM [cm³/cm³]",
        "rd": "RD [cm]",
    }

    for key, label in non_negative_fields.items():
        val = getattr(state, key, None)
        if val is not None and isinstance(val, (int, float)) and val < 0.0:
            violations.append(f"{label} cannot be negative (got {val:.4f})")

    # 2. Relative Transpiration Factor (RFTRA) range [0.0, 1.0]
    rftra = getattr(state, "rftra", None)
    if rftra is not None and isinstance(rftra, (int, float)):
        if rftra < 0.0 or rftra > 1.0:
            violations.append(f"RFTRA [-] must be in range [0.0, 1.0] (got {rftra:.4f})")

    # 3. Soil Moisture saturation check (SM <= SM0)
    sm = getattr(state, "sm", None)
    if sm is not None and isinstance(sm, (int, float)):
        sm0 = None
        # Extract SM0 from wofost instance if available
        if wofost is not None:
            try:
                soildata = None
                if hasattr(wofost, "params") and hasattr(wofost.params, "soildata"):
                    soildata = wofost.params.soildata
                elif hasattr(wofost, "soildata"):
                    soildata = wofost.soildata

                if isinstance(soildata, dict):
                    sm0_val = soildata.get("SM0")
                    if isinstance(sm0_val, (int, float)):
                        sm0 = float(sm0_val)
            except Exception:
                sm0 = None

        if sm0 is not None and sm > sm0:
            violations.append(
                f"Soil moisture SM ({sm:.4f}) exceeds soil saturation SM0 ({sm0:.4f})"
            )
        elif sm > 1.0:
            violations.append(f"Soil moisture SM ({sm:.4f}) exceeds physical upper limit 1.0")

    # 4. Biomass component consistency vs TAGP
    tagp = getattr(state, "tagp", None)
    twso = getattr(state, "twso", None)
    twlv = getattr(state, "twlv", None)
    twst = getattr(state, "twst", None)

    if tagp is not None and isinstance(tagp, (int, float)):
        # Individual component checks
        if twso is not None and isinstance(twso, (int, float)) and twso > tagp:
            violations.append(
                f"Storage organ weight TWSO ({twso:.1f}) exceeds total above-ground production TAGP ({tagp:.1f})"
            )
        if twlv is not None and isinstance(twlv, (int, float)) and twlv > tagp:
            violations.append(
                f"Leaf weight TWLV ({twlv:.1f}) exceeds total above-ground production TAGP ({tagp:.1f})"
            )
        if twst is not None and isinstance(twst, (int, float)) and twst > tagp:
            violations.append(
                f"Stem weight TWST ({twst:.1f}) exceeds total above-ground production TAGP ({tagp:.1f})"
            )

        # Sum of aboveground components check
        if (
            twso is not None and isinstance(twso, (int, float))
            and twlv is not None and isinstance(twlv, (int, float))
            and twst is not None and isinstance(twst, (int, float))
        ):
            comp_sum = twso + twlv + twst
            if comp_sum > tagp + 1e-3:
                violations.append(
                    f"Sum of aboveground biomass components TWSO+TWLV+TWST ({comp_sum:.1f}) "
                    f"exceeds TAGP ({tagp:.1f})"
                )

    # 5. Development stage & root depth bounds
    dvs = getattr(state, "dvs", None)
    if dvs is not None and isinstance(dvs, (int, float)):
        if dvs < 0.0 or dvs > 3.0:
            violations.append(f"Development stage DVS ({dvs:.3f}) out of valid range [0.0, 3.0]")

    rd = getattr(state, "rd", None)
    if rd is not None and isinstance(rd, (int, float)):
        rdmsol = None
        if wofost is not None:
            try:
                soildata = None
                if hasattr(wofost, "params") and hasattr(wofost.params, "soildata"):
                    soildata = wofost.params.soildata
                elif hasattr(wofost, "soildata"):
                    soildata = wofost.soildata

                if isinstance(soildata, dict):
                    rd_val = soildata.get("RDMSOL")
                    if isinstance(rd_val, (int, float)):
                        rdmsol = float(rd_val)
            except Exception:
                rdmsol = None

        if rdmsol is not None and rd > rdmsol:
            violations.append(
                f"Root depth RD ({rd:.1f} cm) exceeds maximum rootable soil depth RDMSOL ({rdmsol:.1f} cm)"
            )

    # Pre-emergence check (DVS < 0)
    if dvs is not None and isinstance(dvs, (int, float)) and dvs < 0.0:
        lai = getattr(state, "lai", None)
        if lai is not None and isinstance(lai, (int, float)) and lai > 0.0:
            violations.append(f"Pre-emergence state (DVS < 0) cannot have non-zero LAI ({lai:.4f})")
        twrt = getattr(state, "twrt", None)
        if twrt is not None and isinstance(twrt, (int, float)) and twrt > 0.0:
            violations.append(f"Pre-emergence state (DVS < 0) cannot have non-zero root biomass TWRT ({twrt:.1f})")

    is_valid = len(violations) == 0
    return is_valid, violations
