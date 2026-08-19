"""
Confidence Estimation Service
=============================

Assigns a confidence score (0-1) and corresponding observation error (R) to each data source.

Research Alignment:
- Provides the "R" (observation error) for the EnKF's Kalman Gain.
- High confidence = Low R = High Gain (model trusts observation).
- Low confidence = High R = Low Gain (model trusts itself).
- Specific handling for farmer photos (R=0.3) as per Option D.
- Cloud cover penalty for optical satellites (monsoon season).
"""

import logging



from backend.app.api.schemas.fusion import ConfidenceRequest, ConfidenceResponse, ObservationSource

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """
    Computes confidence scores for observations based on source, quality, and environmental factors.
    """
    
    # Base confidence (0-1) for each source type
    BASE_CONFIDENCE = {
        ObservationSource.SENTINEL2: 0.90,
        ObservationSource.SENTINEL1_SAR: 0.75,
        ObservationSource.MODIS: 0.60,
        ObservationSource.SMARTPHONE_GRVI: 0.60,
        ObservationSource.ERA5_LAND: 0.80,
        ObservationSource.FUSED: 0.85,
    }
    
    # Default observation error (R) for EnKF if confidence is 1.0
    # Lower R = more trust. This maps to the research's R values.
    BASE_OBSERVATION_ERROR = {
        ObservationSource.SENTINEL2: 0.10,
        ObservationSource.SENTINEL1_SAR: 0.20,
        ObservationSource.MODIS: 0.25,
        ObservationSource.SMARTPHONE_GRVI: 0.30,  # RESEARCH OPTION D: Gentle nudge
        ObservationSource.ERA5_LAND: 0.15,
        ObservationSource.FUSED: 0.12,
    }
    
    def compute_confidence(self, request: ConfidenceRequest) -> ConfidenceResponse:
        """
        Compute overall confidence score and the R value for EnKF.
        """
        source = request.source
        base_conf = self.BASE_CONFIDENCE.get(source, 0.50)
        base_r = self.BASE_OBSERVATION_ERROR.get(source, 0.25)
        
        factors = {}
        adjusted_conf = base_conf
        
        # --- 1. Cloud Cover Penalty (Optical only) ---
        if source in [ObservationSource.SENTINEL2, ObservationSource.MODIS]:
            cloud_penalty = 1.0 - (request.cloud_cover / 100.0)
            factors["cloud_cover"] = cloud_penalty
            adjusted_conf *= cloud_penalty
        
        # --- 2. Viewing Angle Penalty ---
        # Off-nadir angles (> 15 degrees) degrade quality
        angle_penalty = max(0.0, 1.0 - (request.viewing_angle / 45.0))
        factors["viewing_angle"] = angle_penalty
        adjusted_conf *= angle_penalty
        
        # --- 3. Sensor Health ---
        factors["sensor_health"] = request.sensor_health
        adjusted_conf *= request.sensor_health
        
        # --- 4. Data Age Decay ---
        # Older observations are less relevant
        if request.days_since_observation > 5:
            age_decay = max(0.5, 1.0 - (request.days_since_observation - 5) * 0.02)
            factors["age_decay"] = age_decay
            adjusted_conf *= age_decay
        else:
            factors["age_decay"] = 1.0
        
        # --- 5. Value Physical Plausibility ---
        # If the value is physically unrealistic, reduce confidence
        if request.source == ObservationSource.SMARTPHONE_GRVI:
            # GRVI should be between -1 and 1
            if not (-1.0 <= request.value <= 1.0):
                adjusted_conf *= 0.5
                factors["value_plausibility"] = 0.5
            else:
                factors["value_plausibility"] = 1.0
        elif request.source in [ObservationSource.SENTINEL2, ObservationSource.SENTINEL1_SAR]:
            # LAI should be between 0 and 8
            if not (0.0 <= request.value <= 8.0):
                adjusted_conf *= 0.5
                factors["value_plausibility"] = 0.5
            else:
                factors["value_plausibility"] = 1.0
        
        # --- Clamp and Finalize ---
        final_conf = max(0.10, min(1.0, adjusted_conf))
        
        # Convert confidence to R (Observation Error)
        # R = Base_R + (1 - Confidence) * (1 - Base_R)
        # If confidence is high (1.0), R = Base_R.
        # If confidence is low (0.0), R approaches 1.0 (complete distrust).
        observation_error_r = base_r + (1.0 - final_conf) * (1.0 - base_r)
        observation_error_r = max(0.01, min(0.95, observation_error_r))
        
        return ConfidenceResponse(
            source=source,
            original_value=request.value,
            confidence_score=round(final_conf, 3),
            observation_error_r=round(observation_error_r, 3),
            factors=factors,
            message=f"Confidence computed. R={observation_error_r:.3f} for EnKF."
        )
    
    def batch_compute(self, observations: list) -> list:
        """Compute confidence for multiple observations."""
        results = []
        for obs in observations:
            req = ConfidenceRequest(**obs)
            results.append(self.compute_confidence(req))
        return results
