"""
Multi-source Fusion Service
===========================

Fuses optical (Sentinel-2), radar (Sentinel-1 SAR), and ground-truth (farmer photos)
into a single, best-estimate LAI/SM value.

Research Alignment:
- Option C: Sentinel-2 fails during cloudy monsoon seasons (>70% cloud cover).
- Sentinel-1 SAR (RVI) works through clouds via radar.
- Option D: Farmer photos (GRVI) provide a reality check (30% weight).
- Combines sources based on dynamic cloud cover and confidence scores.
"""

import logging
import numpy as np




from backend.app.api.schemas.fusion import (
    FusionRequest,
    FusionResponse,
    ConfidenceRequest,
    ObservationSource
)
from backend.app.services.confidence_estimator import ConfidenceEstimator

logger = logging.getLogger(__name__)


class MultiSourceFusionService:
    """
    Fuses multiple LAI/SM sources into a single optimal estimate.
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self.confidence_estimator = ConfidenceEstimator()
    
    def _rvi_to_lai(self, rvi: float, crop_type: str = "wheat") -> float:
        """
        Convert Sentinel-1 Radar Vegetation Index (RVI) to LAI.
        
        RVI = (4 * σ_vh) / (σ_vv + σ_vh)
        LAI = a * RVI + b (calibrated for Indian crops)
        
        Research: Empirical calibration for rice/wheat in Indo-Gangetic plains.
        """
        # Calibration coefficients (simplified, based on published literature)
        coefficients = {
            "wheat": {"a": 3.5, "b": 0.2},
            "rice": {"a": 4.0, "b": 0.1},
            "maize": {"a": 3.0, "b": 0.3},
            "default": {"a": 3.0, "b": 0.2}
        }
        coeff = coefficients.get(crop_type.lower(), coefficients["default"])
        lai = coeff["a"] * rvi + coeff["b"]
        return max(0.0, min(8.0, lai))
    
    def _grvi_to_lai(self, grvi: float) -> float:
        """
        Convert GRVI (from smartphone photos) to LAI.
        
        GRVI = (G - R) / (G + R)
        LAI = a * GRVI^2 + b * GRVI + c (non-linear relationship)
        
        Research: Plants (2024) - GRVI correlates with SPAD (R² > 0.85).
        """
        # Polynomial calibration from research papers
        # LAI ≈ 1.5 * GRVI^2 + 3.0 * GRVI + 0.5
        lai = 1.5 * (grvi ** 2) + 3.0 * grvi + 0.5
        return max(0.0, min(8.0, lai))
    
    def fuse_lai(self, request: FusionRequest) -> FusionResponse:
        """
        Fuse LAI observations using dynamic weights based on cloud cover and confidence.
        """
        cloud_cover = request.cloud_cover
        crop_type = request.crop_type or "default"
        
        # Extract observations by source
        obs_dict = {obs.get("source"): obs for obs in request.observations}
        
        # --- Step 1: Determine weights based on cloud cover ---
        if cloud_cover < 40:
            # Clear sky: Trust Sentinel-2 (Optical)
            weights = {
                ObservationSource.SENTINEL2.value: 0.70,
                ObservationSource.SENTINEL1_SAR.value: 0.10,
                ObservationSource.SMARTPHONE_GRVI.value: 0.20,
            }
            quality_flag = "HIGH"
            primary_source = "SENTINEL2"
            
        elif 40 <= cloud_cover < 70:
            # Partly cloudy: Weighted blend
            s2_weight = 1.0 - (cloud_cover / 100.0)  # 0.3 to 0.6
            s1_weight = 1.0 - s2_weight
            weights = {
                ObservationSource.SENTINEL2.value: s2_weight * 0.7,
                ObservationSource.SENTINEL1_SAR.value: s1_weight * 0.7,
                ObservationSource.SMARTPHONE_GRVI.value: 0.30,  # Fixed 30% for reality check
            }
            quality_flag = "MEDIUM"
            primary_source = "FUSED_S2_S1"
            
        else:
            # >70% Cloud cover (Monsoon): Use Sentinel-1 SAR exclusively
            weights = {
                ObservationSource.SENTINEL1_SAR.value: 0.70,
                ObservationSource.SMARTPHONE_GRVI.value: 0.30,
                ObservationSource.SENTINEL2.value: 0.0,
            }
            quality_flag = "LOW"
            primary_source = "SENTINEL1_SAR"
        
        # --- Step 2: Normalize weights so they sum to 1 ---
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # --- Step 3: Compute weighted LAI using inverse-variance uncertainty ---
        fused_lai = 0.0
        contributing_sources = []
        source_weights = {}
        source_confidences = {}
        
        for source, weight in weights.items():
            if weight <= 0:
                continue
            
            obs = obs_dict.get(source)
            if not obs:
                continue
            
            raw_value = obs.get("value")
            if raw_value is None:
                continue

            source_enum = ObservationSource(source)
            
            # Convert to LAI if necessary
            if source == ObservationSource.SENTINEL1_SAR.value:
                # Assume raw_value is RVI
                lai_value = self._rvi_to_lai(raw_value, crop_type)
            elif source == ObservationSource.SMARTPHONE_GRVI.value:
                # Assume raw_value is GRVI
                lai_value = self._grvi_to_lai(raw_value)
            else:
                lai_value = raw_value
            
            # --- Extract Observation Uncertainty R = variance = std_dev^2 ---
            # Check for explicit uncertainty metrics in observation item
            r_val = None
            if "variance" in obs and obs["variance"] is not None and float(obs["variance"]) > 0:
                r_val = float(obs["variance"])
            elif "observation_error_r" in obs and obs["observation_error_r"] is not None and float(obs["observation_error_r"]) > 0:
                r_val = float(obs["observation_error_r"])
            elif "uncertainty" in obs and obs["uncertainty"] is not None and float(obs["uncertainty"]) > 0:
                r_val = float(obs["uncertainty"]) ** 2
            elif "std_dev" in obs and obs["std_dev"] is not None and float(obs["std_dev"]) > 0:
                r_val = float(obs["std_dev"]) ** 2

            # Compute confidence score and fallback R
            conf_req = ConfidenceRequest(
                source=source_enum,
                value=lai_value,
                cloud_cover=cloud_cover if source == ObservationSource.SENTINEL2.value else 0,
                viewing_angle=obs.get("viewing_angle", 0),
                sensor_health=obs.get("sensor_health", 1.0),
                days_since_observation=obs.get("days_since", 0)
            )
            conf_resp = self.confidence_estimator.compute_confidence(conf_req)

            # If no explicit uncertainty field provided, use confidence-derived error variance R
            if r_val is None:
                r_val = float(conf_resp.observation_error_r)

            # Safeguard lower bound on R to prevent division by zero
            r_val = max(1e-6, r_val)

            # Inverse-variance weighting modulated by cloud-cover source eligibility prior
            # w_i \propto base_weight / R_i
            effective_weight = weight / r_val
            fused_lai += effective_weight * lai_value
            source_weights[source] = effective_weight
            source_confidences[source] = conf_resp.confidence_score
            contributing_sources.append(source)
        
        # --- Step 4: Normalize final fused value and weights ---
        total_eff_weight = sum(source_weights.values())
        if total_eff_weight > 0:
            fused_lai = fused_lai / total_eff_weight
            normalized_weights = {k: v / total_eff_weight for k, v in source_weights.items()}
        else:
            # If no valid observations, fallback to no data
            return FusionResponse(
                field_id=request.field_id,
                date=request.date,
                fused_lai=0.0,
                fused_confidence=0.0,
                source_weights={},
                contributing_sources=[],
                quality_flag="NO_DATA",
                message="No valid observations to fuse."
            )
        
        # --- Step 5: Compute overall confidence of the fused product ---
        weighted_confidence = sum(
            normalized_weights[s] * source_confidences[s] for s in contributing_sources
        )
        fused_confidence = min(0.95, max(0.10, weighted_confidence))
        
        # Clamp LAI to physical bounds
        fused_lai = max(0.0, min(8.0, fused_lai))
        
        return FusionResponse(
            field_id=request.field_id,
            date=request.date,
            fused_lai=round(fused_lai, 3),
            fused_confidence=round(fused_confidence, 3),
            source_weights={k: round(v, 3) for k, v in normalized_weights.items()},
            contributing_sources=contributing_sources,
            quality_flag=quality_flag,
            message=f"Fused LAI using {len(contributing_sources)} sources with uncertainty weighting. Primary: {primary_source}"
        )
