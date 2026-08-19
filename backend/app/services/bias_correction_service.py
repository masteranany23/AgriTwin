"""
Bias Correction Service for AgriTwin v3.0 Multi-Model Averaging.

This service applies stacked ensemble corrections to WOFOST outputs.
It loads the trained stacked ensemble model and applies bias corrections.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default project root is 3 levels up: backend/app/services -> backend/app -> backend -> root
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]

class BiasCorrectionService:
    """Service for applying bias corrections using stacked ensemble."""
    
    def __init__(self, project_root: Optional[Path | str] = None):
        self.project_root = Path(project_root) if project_root else DEFAULT_PROJECT_ROOT
        self.models_dir = self.project_root / "models"
        
        # Load the stacked ensemble model
        self.ensemble_model = None
        self.model_loaded = False
        self._load_model()
        
        # Feature names expected by the model
        self.expected_features = [
            'lai_mean', 'lai_std', 'lai_trend',
            'sm_layer1_mean', 'sm_layer1_std',
            'sm_layer2_mean', 'sm_layer2_std',
            'sm_layer3_mean', 'sm_layer3_std',
            'sm_layer4_mean', 'sm_layer4_std',
            'heat_strain_hours', 'dvs_phase'
        ]
    
    def _load_model(self):
        """Load the stacked ensemble model from disk."""
        model_file = self.models_dir / "stacked_ensemble.pkl"
        
        if not model_file.exists():
            logger.warning(f"Stacked ensemble model not found: {model_file}")
            logger.info("Will attempt to load fallback XGBoost model if available")
            
            # Try to load fallback XGBoost model
            xgb_model_file = self.models_dir / "xgboost_model.pkl"
            if xgb_model_file.exists():
                try:
                    with open(xgb_model_file, 'rb') as f:
                        self.ensemble_model = pickle.load(f)
                    self.model_loaded = True
                    logger.info(f"Loaded fallback XGBoost model from {xgb_model_file}")
                except Exception as e:
                    logger.error(f"Failed to load fallback model: {e}")
            return
        
        try:
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.ensemble_model = saved_data.get('pipeline')
            self.model_loaded = True
            self.model_metadata = saved_data.get('training_history', {})
            
            logger.info(f"Loaded stacked ensemble model from {model_file}")
            logger.info(f"Model version: {saved_data.get('version', 'unknown')}")
            
            # Log model performance if available
            if 'evaluation' in self.model_metadata:
                eval_data = self.model_metadata['evaluation']
                logger.info(f"Model performance - RMSE: {eval_data.get('rmse', 'N/A'):.2f}, "
                           f"R²: {eval_data.get('r2', 'N/A'):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to load stacked ensemble model: {e}")
            self.model_loaded = False
    
    def is_model_available(self) -> bool:
        """Check if a bias correction model is available."""
        return self.model_loaded and self.ensemble_model is not None
    
    def extract_features_from_daily_data(self, daily_data: pd.DataFrame, 
                                        window_size: int = 30) -> pd.DataFrame:
        """
        Extract features from daily WOFOST data for bias correction.
        
        Args:
            daily_data: DataFrame with daily WOFOST outputs
            window_size: Size of sliding window (default: 30 days)
            
        Returns:
            DataFrame with extracted features
        """
        logger.info(f"Extracting features from {len(daily_data)} daily records")
        
        features = []
        
        # Process each day in the data
        for i in range(len(daily_data)):
            # Get window for current day (looking back window_size days)
            start_idx = max(0, i - window_size + 1)
            window_data = daily_data.iloc[start_idx:i + 1]
            
            # Skip if window is too small
            if len(window_data) < 7:  # Minimum 7 days for meaningful statistics
                continue
            
            # Extract LAI features
            if 'lai' in window_data.columns:
                lai_values = window_data['lai'].values
                lai_mean = np.mean(lai_values)
                lai_std = np.std(lai_values)
                
                # Calculate trend if we have enough points
                if len(lai_values) > 1:
                    try:
                        lai_trend = np.polyfit(range(len(lai_values)), lai_values, 1)[0]
                    except:
                        lai_trend = 0.0
                else:
                    lai_trend = 0.0
            else:
                lai_mean = lai_std = lai_trend = np.nan
            
            # Extract soil moisture features (4 layers)
            sm_features = {}
            for layer_idx in range(1, 5):
                sm_key = f'sm_layer{layer_idx}'
                if sm_key in window_data.columns:
                    sm_values = window_data[sm_key].values
                    sm_features[f'sm_layer{layer_idx}_mean'] = np.mean(sm_values)
                    sm_features[f'sm_layer{layer_idx}_std'] = np.std(sm_values)
                else:
                    # Try alternative column names
                    alt_keys = [f'sm{layer_idx}', f'SM_L{layer_idx}', f'soil_moisture_{layer_idx}']
                    found = False
                    for alt_key in alt_keys:
                        if alt_key in window_data.columns:
                            sm_values = window_data[alt_key].values
                            sm_features[f'sm_layer{layer_idx}_mean'] = np.mean(sm_values)
                            sm_features[f'sm_layer{layer_idx}_std'] = np.std(sm_values)
                            found = True
                            break
                    
                    if not found:
                        sm_features[f'sm_layer{layer_idx}_mean'] = np.nan
                        sm_features[f'sm_layer{layer_idx}_std'] = np.nan
            
            # Heat strain hours (estimate if temperature data available)
            if 'temperature' in window_data.columns:
                heat_strain_hours = np.sum(window_data['temperature'] > 34)
            elif 'tmax' in window_data.columns:
                heat_strain_hours = np.sum(window_data['tmax'] > 34)
            else:
                # Estimate based on season if date available
                heat_strain_hours = 0
            
            # DVS phase (0=Vegetative, 1=Reproductive, 2=Grain fill)
            if 'dvs' in daily_data.columns and i < len(daily_data):
                current_dvs = daily_data.iloc[i]['dvs']
                if current_dvs < 0.65:
                    dvs_phase = 0  # Vegetative
                elif current_dvs < 1.3:
                    dvs_phase = 1  # Reproductive
                else:
                    dvs_phase = 2  # Grain fill
            else:
                dvs_phase = 0
            
            # Get current date if available
            if 'date' in daily_data.columns and i < len(daily_data):
                current_date = daily_data.iloc[i]['date']
            else:
                current_date = None
            
            feature_row = {
                'date': current_date,
                'lai_mean': lai_mean,
                'lai_std': lai_std,
                'lai_trend': lai_trend,
                'heat_strain_hours': heat_strain_hours,
                'dvs_phase': dvs_phase,
                **sm_features
            }
            
            features.append(feature_row)
        
        if not features:
            logger.warning("No features extracted from daily data")
            return pd.DataFrame(columns=self.expected_features)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted {len(features_df)} feature samples")
        
        return features_df
    
    def apply_bias_correction(self, daily_data: pd.DataFrame, 
                             wofost_final_yield: float) -> Dict[str, Any]:
        """
        Apply bias correction to WOFOST outputs using stacked ensemble.
        
        Args:
            daily_data: DataFrame with daily WOFOST outputs
            wofost_final_yield: Final yield predicted by WOFOST (kg/ha)
            
        Returns:
            Dictionary with correction results
        """
        if not self.is_model_available():
            logger.warning("No bias correction model available")
            return {
                'correction_applied': False,
                'final_yield': wofost_final_yield,
                'correction_amount': 0.0,
                'confidence_interval': [wofost_final_yield * 0.9, wofost_final_yield * 1.1],
                'message': 'No bias correction model available'
            }
        
        try:
            # Extract features from daily data
            features_df = self.extract_features_from_daily_data(daily_data)
            
            if features_df.empty:
                logger.warning("No features extracted, cannot apply correction")
                return {
                    'correction_applied': False,
                    'final_yield': wofost_final_yield,
                    'correction_amount': 0.0,
                    'confidence_interval': [wofost_final_yield * 0.9, wofost_final_yield * 1.1],
                    'message': 'No features extracted from daily data'
                }
            
            # Prepare features for prediction
            X = features_df.copy()
            
            # Ensure we have all expected features (fill missing with NaN)
            for feature in self.expected_features:
                if feature not in X.columns:
                    X[feature] = np.nan
            
            # Keep only expected features in correct order
            X = X[self.expected_features]
            
            # Handle missing values
            if X.isna().any().any():
                logger.info("Handling missing values in features...")
                # Fill numeric columns with their mean
                X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype.kind in 'biufc' else col)
            
            # Make predictions for all windows
            y_pred = self.ensemble_model.predict(X)
            
            # Average predictions to get overall correction
            avg_correction = np.mean(y_pred) if len(y_pred) > 0 else 0.0
            
            # Calculate confidence interval from prediction distribution
            if len(y_pred) > 1:
                std_correction = np.std(y_pred)
                confidence_interval = [
                    wofost_final_yield + avg_correction - 1.96 * std_correction,
                    wofost_final_yield + avg_correction + 1.96 * std_correction
                ]
            else:
                # Default 20% confidence interval if we don't have enough predictions
                confidence_interval = [
                    (wofost_final_yield + avg_correction) * 0.8,
                    (wofost_final_yield + avg_correction) * 1.2
                ]
            
            # Apply correction to final yield
            corrected_yield = wofost_final_yield + avg_correction
            
            # Ensure yield is positive
            corrected_yield = max(0, corrected_yield)
            
            logger.info(f"Applied bias correction: {avg_correction:.2f} kg/ha")
            logger.info(f"Original yield: {wofost_final_yield:.2f} kg/ha")
            logger.info(f"Corrected yield: {corrected_yield:.2f} kg/ha")
            logger.info(f"80% CI: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
            
            return {
                'correction_applied': True,
                'final_yield': corrected_yield,
                'correction_amount': avg_correction,
                'confidence_interval': confidence_interval,
                'original_yield': wofost_final_yield,
                'model_used': 'stacked_ensemble' if 'stacking' in str(type(self.ensemble_model)) else 'xgboost',
                'num_predictions': len(y_pred),
                'correction_std': np.std(y_pred) if len(y_pred) > 1 else 0.0,
                'message': f'Applied {self.get_model_type()} bias correction'
            }
            
        except Exception as e:
            logger.error(f"Error applying bias correction: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'correction_applied': False,
                'final_yield': wofost_final_yield,
                'correction_amount': 0.0,
                'confidence_interval': [wofost_final_yield * 0.9, wofost_final_yield * 1.1],
                'message': f'Error applying bias correction: {str(e)}'
            }
    
    def get_model_type(self) -> str:
        """Get the type of model being used."""
        if not self.ensemble_model:
            return "none"
        
        model_str = str(type(self.ensemble_model)).lower()
        if 'stacking' in model_str or 'ensemble' in model_str:
            return "stacked_ensemble"
        elif 'xgboost' in model_str:
            return "xgboost"
        elif 'pipeline' in model_str:
            # Check if it's a pipeline containing stacking
            if hasattr(self.ensemble_model, 'named_steps'):
                for name, step in self.ensemble_model.named_steps.items():
                    if 'stacking' in str(type(step)).lower():
                        return "stacked_ensemble"
        return "unknown"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_model_available():
            return {'available': False, 'message': 'No model loaded'}
        
        info = {
            'available': True,
            'model_type': self.get_model_type(),
            'loaded': self.model_loaded,
        }
        
        # Add metadata if available
        if hasattr(self, 'model_metadata'):
            info['metadata'] = self.model_metadata
        
        # Add model performance if available
        if hasattr(self, 'model_metadata') and 'evaluation' in self.model_metadata:
            eval_data = self.model_metadata['evaluation']
            info['performance'] = {
                'rmse': eval_data.get('rmse'),
                'r2': eval_data.get('r2'),
                'improvement_pct': eval_data.get('improvement_over_baseline_pct')
            }
        
        return info

# Singleton instance for easy access
_bias_correction_service = None

def get_bias_correction_service() -> BiasCorrectionService:
    """Get or create the bias correction service singleton."""
    global _bias_correction_service
    if _bias_correction_service is None:
        _bias_correction_service = BiasCorrectionService()
    return _bias_correction_service