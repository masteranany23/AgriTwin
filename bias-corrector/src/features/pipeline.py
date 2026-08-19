"""
Feature engineering pipeline for training.
"""
import logging
from typing import List, Dict


import pandas as pd


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for bias correction model.
    
    Transforms raw inputs into model-ready features.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.feature_names: List[str] = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit pipeline and transform training data.
        
        Args:
            df: Raw training data.
            
        Returns:
            Transformed feature DataFrame.
        """
        logger.info("Fitting feature pipeline...")
        
        features_df = df.copy()
        
        # Core features
        core_features = [
            "wofost_yield",
            "latitude",
            "longitude",
            "year"
        ]
        
        # Weather/Climate features
        weather_features = [
            "rainfall_total",
            "temperature_mean",
            "soil_moisture_mean"
        ]
        
        # Satellite features
        satellite_features = [
            "lai_mean",
            "ndvi_mean",
            "ndre_mean"
        ]
        
        # Fill missing values
        for col in weather_features + satellite_features:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0
        
        # Derived features
        features_df["year_norm"] = (features_df["year"] - features_df["year"].min()) / (
            features_df["year"].max() - features_df["year"].min() + 1e-6
        )
        
        # Spatial features
        features_df["lat_lon_interact"] = features_df["latitude"] * features_df["longitude"]
        
        # Temporal features
        features_df["year_squared"] = features_df["year_norm"] ** 2
        
        # Climate-yield interaction
        if "rainfall_total" in features_df.columns and "temperature_mean" in features_df.columns:
            features_df["climate_stress"] = (
                features_df["temperature_mean"] / (features_df["rainfall_total"] + 1e-6)
            )
        
        # Vegetation features
        if "lai_mean" in features_df.columns and "ndvi_mean" in features_df.columns:
            features_df["veg_health"] = features_df["lai_mean"] * features_df["ndvi_mean"]
        
        # Select final features
        self.feature_names = (
            core_features +
            weather_features +
            satellite_features +
            ["year_norm", "lat_lon_interact", "year_squared", "climate_stress", "veg_health"]
        )
        
        # Ensure all features exist
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        result = features_df[self.feature_names]
        
        logger.info(f"Feature pipeline fitted with {len(self.feature_names)} features")
        logger.info(f"Features: {self.feature_names}")
        
        return result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: Raw data.
            
        Returns:
            Transformed features.
        """
        if not self.feature_names:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        # Apply same transformations as fit_transform
        return self.fit_transform(df)
