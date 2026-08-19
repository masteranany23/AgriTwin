"""
Main correction model orchestrator supporting multiple modes.
"""
import logging

from typing import Dict, Optional
import json

import numpy as np
import pandas as pd

from ..api.schemas import PredictionRequest, PredictionResponse, ConfidenceInterval
from ..api.config import Config, get_model_path, get_gp_model_path, get_ics_path
from .ensemble import StackedEnsemble
from .gp_correction import DeepGPCorrection


logger = logging.getLogger(__name__)


class CorrectionModel:
    """
    Unified correction model supporting multiple modes:
    - constant: ICS ratio multiplication
    - xgboost: Single XGBoost fallback
    - ensemble: Stacked ensemble
    - ensemble_gp: Ensemble + GP spatial correction
    """
    
    def __init__(self, config: Config):
        """
        Initialize correction model.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.model_type = config.model["type"]
        self.version = config.model["version"]
        
        self.ensemble: Optional[StackedEnsemble] = None
        self.gp_model: Optional[DeepGPCorrection] = None
        self.ics_ratios: Optional[Dict] = None
        
        logger.info(f"Initialized CorrectionModel (type={self.model_type})")
    
    def load(self):
        """Load model artifacts based on type."""
        logger.info(f"Loading model artifacts for {self.model_type}...")
        
        # Load ICS ratios (always needed for fallback)
        self._load_ics_ratios()
        
        # Load ensemble if needed
        if self.model_type in ["ensemble", "ensemble_gp", "xgboost"]:
            model_path = get_model_path(self.config)
            if model_path.exists():
                if self.model_type in ["ensemble", "ensemble_gp"]:
                    self.ensemble = StackedEnsemble(self.config.model.get("ensemble", {}))
                    self.ensemble.load(model_path)
                    logger.info("Ensemble loaded")
                else:
                    # For xgboost-only mode, create minimal ensemble
                    self.ensemble = StackedEnsemble(self.config.model.get("ensemble", {}))
                    self.ensemble.load(model_path)
                    logger.info("XGBoost model loaded")
            else:
                logger.warning(f"Model not found at {model_path}, using ICS fallback")
        
        # Load GP if enabled
        if self.model_type == "ensemble_gp" and self.config.model.get("enable_gp", False):
            gp_path = get_gp_model_path(self.config)
            if gp_path.exists():
                self.gp_model = DeepGPCorrection(self.config.model.get("gp", {}))
                self.gp_model.load(gp_path)
                logger.info("GP correction loaded")
            else:
                logger.warning(f"GP model not found at {gp_path}, disabling GP")
    
    def _load_ics_ratios(self):
        """Load ICS ratio lookup table."""
        ics_dir = get_ics_path(self.config)
        ics_files = self.config.data.get("ics_files", [])
        
        self.ics_ratios = {}
        
        for filename in ics_files:
            filepath = ics_dir / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    data = json.load(f)
                    self.ics_ratios.update(data)
                logger.info(f"Loaded ICS ratios from {filename}")
            else:
                logger.warning(f"ICS file not found: {filepath}")
        
        if self.ics_ratios:
            logger.info(f"Total ICS ratios loaded: {len(self.ics_ratios)}")
        else:
            logger.warning("No ICS ratios loaded")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate bias-corrected prediction.
        
        Args:
            request: Prediction request with WOFOST yield and features.
            
        Returns:
            PredictionResponse with corrected yield.
        """
        warnings = []
        original_yield = request.wofost_yield
        
        # Build feature vector
        features = self._build_features(request)
        
        # Mode-specific correction
        if self.model_type == "constant":
            corrected_yield, correction_factor, ics_ratio, ci = self._constant_correction(
                request, original_yield
            )
            if ics_ratio is None:
                warnings.append("No ICS ratio available, using 1.0")
        
        elif self.model_type in ["xgboost", "ensemble"]:
            if self.ensemble is not None:
                corrected_yield, correction_factor, ci = self._ensemble_correction(
                    features, original_yield
                )
                ics_ratio = None
            else:
                # Fallback to ICS
                warnings.append("Model not loaded, using ICS fallback")
                corrected_yield, correction_factor, ics_ratio, ci = self._constant_correction(
                    request, original_yield
                )
        
        elif self.model_type == "ensemble_gp":
            if self.ensemble is not None:
                corrected_yield, correction_factor, ci = self._ensemble_gp_correction(
                    features, original_yield, request
                )
                ics_ratio = None
            else:
                warnings.append("Model not loaded, using ICS fallback")
                corrected_yield, correction_factor, ics_ratio, ci = self._constant_correction(
                    request, original_yield
                )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Ensure positive yield
        if corrected_yield < 0:
            warnings.append("Negative prediction clipped to 0")
            corrected_yield = 0.0
        
        return PredictionResponse(
            original_yield=original_yield,
            corrected_yield=corrected_yield,
            correction_factor=correction_factor,
            ics_ratio=ics_ratio,
            model_version=self.version,
            confidence_interval=ci,
            warnings=warnings
        )
    
    def _constant_correction(
        self,
        request: PredictionRequest,
        original_yield: float
    ) -> tuple:
        """Apply ICS ratio correction."""
        # Lookup key: state_district_crop_year
        key = f"{request.state}_{request.district}_{request.crop_key.value}_{request.year}"
        
        ics_ratio = self.ics_ratios.get(key)
        
        if ics_ratio is None:
            # Try without year
            key_no_year = f"{request.state}_{request.district}_{request.crop_key.value}"
            ics_ratio = self.ics_ratios.get(key_no_year)
        
        if ics_ratio is None:
            # Fallback: no correction
            ics_ratio = 1.0
        
        corrected_yield = original_yield * ics_ratio
        correction_factor = ics_ratio
        
        # Simple CI: ±10% of corrected yield
        ci = ConfidenceInterval(
            lower=corrected_yield * 0.9,
            upper=corrected_yield * 1.1,
            mean=corrected_yield
        )
        
        return corrected_yield, correction_factor, ics_ratio, ci
    
    def _ensemble_correction(
        self,
        features: pd.DataFrame,
        original_yield: float
    ) -> tuple:
        """Apply ensemble correction."""
        # Predict with ensemble
        pred = self.ensemble.predict(features)[0]
        corrected_yield = float(pred)
        correction_factor = corrected_yield / (original_yield + 1e-6)
        
        # Confidence interval from ensemble variance (simplified)
        ci = ConfidenceInterval(
            lower=corrected_yield * 0.85,
            upper=corrected_yield * 1.15,
            mean=corrected_yield
        )
        
        return corrected_yield, correction_factor, ci
    
    def _ensemble_gp_correction(
        self,
        features: pd.DataFrame,
        original_yield: float,
        request: PredictionRequest
    ) -> tuple:
        """Apply ensemble + GP correction."""
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict(features)[0]
        
        # GP correction if available
        if self.gp_model is not None:
            lat = np.array([request.latitude])
            lon = np.array([request.longitude])
            year = np.array([request.year])
            
            corrected_yield, variance = self.gp_model.predict(
                ensemble_pred.reshape(-1),
                lat, lon, year,
                return_variance=True
            )
            corrected_yield = float(corrected_yield[0])
            
            # Confidence interval from GP variance
            std = np.sqrt(variance[0])
            ci = ConfidenceInterval(
                lower=corrected_yield - 1.28 * std,  # 80% CI
                upper=corrected_yield + 1.28 * std,
                mean=corrected_yield
            )
        else:
            corrected_yield = float(ensemble_pred)
            ci = ConfidenceInterval(
                lower=corrected_yield * 0.85,
                upper=corrected_yield * 1.15,
                mean=corrected_yield
            )
        
        correction_factor = corrected_yield / (original_yield + 1e-6)
        
        return corrected_yield, correction_factor, ci
    
    def _build_features(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Build feature vector from request.
        
        Args:
            request: Prediction request.
            
        Returns:
            Feature DataFrame with single row.
        """
        features = {
            "wofost_yield": request.wofost_yield,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "year": request.year,
            "lai_mean": request.lai_mean or 0.0,
            "ndvi_mean": request.ndvi_mean or 0.0,
            "ndre_mean": request.ndre_mean or 0.0,
            "rainfall_total": request.rainfall_total or 0.0,
            "temperature_mean": request.temperature_mean or 0.0,
            "soil_moisture_mean": request.soil_moisture_mean or 0.0,
        }
        
        # Add derived features
        features["year_norm"] = (request.year - 2010) / 20.0
        
        return pd.DataFrame([features])
