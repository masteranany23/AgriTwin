"""backend/app/services/quality_control_service.py — Centralized Quality Control Service
=====================================================================================

Centralizes observation quality control, physical bounds checking, satellite cloud masking,
quality score thresholds, and statistical Z-score outlier detection relative to model forecasts.

Returns explicit ObservationStatus values:
- VALID: Passed all quality control checks.
- OUTLIER: Failed physical bounds validation or statistical Z-score gate vs forecast ensemble.
- MISSING: Observation value is None/NaN or data missing.
- REJECTED: Failed quality score, cloud cover threshold, explicit DB rejection, or source inclusion check.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Union
import numpy as np

from backend.app.assimilation.models.observation import Observation, ObservationSource, ObservationStatus
from backend.app.assimilation.state.state_vector import STATE_INDEX, STATE_VARIABLES

logger = logging.getLogger(__name__)

# Default physical bounds for common crop & environmental variables
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "LAI":   (0.0, 8.0),      # Leaf Area Index [m2/m2]
    "SM":    (0.0, 0.60),     # Volumetric Soil Moisture [cm3/cm3]
    "TAGP":  (0.0, 50000.0),  # Total Aboveground Production [kg/ha]
    "TWSO":  (0.0, 50000.0),  # Storage Organ Weight [kg/ha]
    "TWLV":  (0.0, 50000.0),  # Leaf Weight [kg/ha]
    "TWST":  (0.0, 50000.0),  # Stem Weight [kg/ha]
    "TWRT":  (0.0, 50000.0),  # Root Weight [kg/ha]
    "DVS":   (0.0, 3.0),      # Development Stage [-]
    "RD":    (0.0, 300.0),    # Root Depth [cm]
    "RFTRA": (0.0, 5.0),      # Relative Transpiration [-]
}


@dataclass
class QCConfig:
    """Quality Control configuration parameters."""
    min_quality_score: Optional[int]   = 60     # Minimum quality score threshold (0-100)
    max_cloud_cover:   Optional[float] = 0.20   # Maximum satellite cloud cover fraction (0.0-1.0)
    max_z_score:       float           = 3.0    # Statistical Z-score outlier gate vs ensemble forecast
    include_sources:   list[str]       = field(
        default_factory=lambda: [s.value for s in ObservationSource]
    )
    custom_bounds:     dict[str, tuple[float, float]] = field(default_factory=dict)

    def get_bounds(self, variable_name: str) -> Optional[tuple[float, float]]:
        """Get physical bounds for a variable, prioritizing custom_bounds."""
        var_upper = variable_name.upper()
        if var_upper in self.custom_bounds:
            return self.custom_bounds[var_upper]
        return PHYSICAL_BOUNDS.get(var_upper)


@dataclass
class QCResult:
    """Explicit evaluation result from QualityControlService."""
    status: ObservationStatus
    passed: bool
    reason: Optional[str] = None
    z_score: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        return self.status == ObservationStatus.VALID and self.passed


class QualityControlService:
    """Centralized Quality Control Service for observation validation, filtering, and gating."""

    def __init__(self, config: Optional[QCConfig] = None) -> None:
        self.config = config or QCConfig()

    def check_physical_bounds(
        self, variable_name: str, value: float, config: Optional[QCConfig] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if a numeric value lies within physical bounds for a variable."""
        if value is None or math.isnan(value):
            return False, "Value is None or NaN"

        cfg = config or self.config
        bounds = cfg.get_bounds(variable_name)
        if bounds is not None:
            min_val, max_val = bounds
            if not (min_val <= value <= max_val):
                return False, f"Value {value:.4f} outside physical bounds [{min_val}, {max_val}] for {variable_name}"

        return True, None

    def check_cloud_cover(
        self, cloud_cover: Optional[float], max_cloud_cover: Optional[float] = None
    ) -> tuple[bool, Optional[str]]:
        """Check if satellite cloud cover is within allowed threshold."""
        max_cc = max_cloud_cover if max_cloud_cover is not None else self.config.max_cloud_cover
        if max_cc is not None and cloud_cover is not None:
            # Handle percentage scale (0-100) vs fraction (0.0-1.0)
            cc_frac = cloud_cover / 100.0 if cloud_cover > 1.0 else cloud_cover
            max_cc_frac = max_cc / 100.0 if max_cc > 1.0 else max_cc
            if cc_frac > max_cc_frac:
                return False, f"Cloud cover {cc_frac:.2f} exceeds threshold {max_cc_frac:.2f}"
        return True, None

    def evaluate_observation(
        self,
        obs: Observation,
        ens_mean: Optional[float] = None,
        ens_std: Optional[float] = None,
        config: Optional[QCConfig] = None,
    ) -> QCResult:
        """Evaluate a single Observation instance against all QC rules.
        
        Returns explicit QCResult with ObservationStatus (VALID, OUTLIER, MISSING, REJECTED).
        """
        cfg = config or self.config

        # 1. Check if observation object or value is missing/NaN
        if obs is None:
            return QCResult(ObservationStatus.MISSING, False, "Observation is None")

        val = getattr(obs, "value", None)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return QCResult(ObservationStatus.MISSING, False, "Observation value is None or NaN")

        # 2. Check DB status override
        current_status = getattr(obs, "status", None)
        if current_status == ObservationStatus.REJECTED:
            return QCResult(ObservationStatus.REJECTED, False, "Explicitly marked REJECTED in database")
        if current_status == ObservationStatus.MISSING:
            return QCResult(ObservationStatus.MISSING, False, "Explicitly marked MISSING in database")

        # 3. Source inclusion filter
        src_obj = getattr(obs, "source", None)
        src_str = src_obj.value if hasattr(src_obj, "value") else str(src_obj)
        if src_str not in cfg.include_sources:
            return QCResult(ObservationStatus.REJECTED, False, f"Source '{src_str}' not in allowed sources {cfg.include_sources}")

        # 4. Quality score filter
        quality_score = getattr(obs, "quality_score", None)
        if cfg.min_quality_score is not None and quality_score is not None:
            if quality_score < cfg.min_quality_score:
                return QCResult(ObservationStatus.REJECTED, False, f"Quality score {quality_score} < threshold {cfg.min_quality_score}")

        # 5. Satellite cloud cover filter
        if src_str == "SATELLITE":
            cloud_cover = getattr(obs, "cloud_cover", None)
            passed_cc, cc_reason = self.check_cloud_cover(cloud_cover, cfg.max_cloud_cover)
            if not passed_cc:
                return QCResult(ObservationStatus.REJECTED, False, cc_reason)

        # 6. Physical bounds check
        var_name = getattr(obs, "variable_name", "") or getattr(obs, "variable", "")
        passed_bounds, bounds_reason = self.check_physical_bounds(var_name, val, cfg)
        if not passed_bounds:
            return QCResult(ObservationStatus.OUTLIER, False, bounds_reason)

        # 7. Outlier Z-score gate vs forecast ensemble
        calculated_z = None
        if ens_mean is not None and ens_std is not None and not math.isnan(ens_mean) and ens_std > 0:
            calculated_z = abs(val - ens_mean) / ens_std
            if calculated_z > cfg.max_z_score:
                return QCResult(
                    ObservationStatus.OUTLIER,
                    False,
                    f"Outlier Z-score {calculated_z:.2f} > threshold {cfg.max_z_score:.2f}",
                    z_score=calculated_z,
                )

        return QCResult(ObservationStatus.VALID, True, z_score=calculated_z)

    def filter_observations(
        self,
        observations: list[Observation],
        X_f: Optional[np.ndarray] = None,
        x_mean_f: Optional[np.ndarray] = None,
        config: Optional[QCConfig] = None,
    ) -> list[Observation]:
        """Filter a list of observations, returning only those passing QC (status == VALID).
        
        Optional X_f and x_mean_f provide state ensemble matrix and mean for Z-score gating.
        """
        cfg = config or self.config
        passed: list[Observation] = []

        # Import _OBS_VAR_TO_SV mapping lazily to avoid circular imports
        from backend.app.assimilation.services.assimilation_service import _OBS_VAR_TO_SV

        for obs in observations:
            ens_mean = None
            ens_std = None

            if X_f is not None and x_mean_f is not None:
                var_name = getattr(obs, "variable_name", "") or getattr(obs, "variable", "")
                sv_key = _OBS_VAR_TO_SV.get(var_name.upper())
                if sv_key is not None and sv_key in STATE_INDEX:
                    idx = STATE_INDEX[sv_key]
                    if idx < len(x_mean_f) and idx < X_f.shape[0]:
                        ens_mean = float(x_mean_f[idx])
                        ens_std = float(np.nanstd(X_f[idx, :]))

            res = self.evaluate_observation(obs, ens_mean=ens_mean, ens_std=ens_std, config=cfg)
            if res.is_valid:
                passed.append(obs)
            else:
                logger.debug("QC filtered obs %s: status=%s, reason=%s", getattr(obs, "id", obs), res.status.value, res.reason)

        return passed
