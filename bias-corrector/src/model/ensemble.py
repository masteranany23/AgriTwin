"""
Stacked ensemble model for yield correction.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb


logger = logging.getLogger(__name__)


class StackedEnsemble:
    """
    Stacked ensemble combining XGBoost, Random Forest, and LightGBM.
    
    Uses TimeSeriesSplit to prevent temporal leakage and Ridge meta-learner
    to combine base model predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize stacked ensemble.
        
        Args:
            config: Configuration dictionary with ensemble parameters.
        """
        self.config = config
        self.n_splits = config.get("n_splits", 5)
        self.random_state = config.get("random_state", 42)
        
        # Base estimators
        self.base_models: Dict[str, Any] = {}
        self.meta_model: Optional[Ridge] = None
        
        # Training history
        self.cv_scores: Dict[str, List[float]] = {}
        self.feature_names: Optional[List[str]] = None
        
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base estimators."""
        logger.info("Initializing base models")
        
        # XGBoost
        self.base_models["xgboost"] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Random Forest
        self.base_models["random_forest"] = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # LightGBM
        self.base_models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # Meta-learner
        self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        years: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train stacked ensemble with temporal cross-validation.
        
        Args:
            X: Feature matrix (N, F).
            y: Target values (N,).
            years: Year for each sample (for proper temporal split).
            
        Returns:
            Dict with final validation metrics.
        """
        logger.info(f"Training ensemble on {len(X)} samples with {X.shape[1]} features")
        
        self.feature_names = list(X.columns)
        
        # Convert to numpy for efficiency
        X_np = X.values
        y_np = y.values
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Store out-of-fold predictions for meta-learner
        oof_preds = np.zeros((len(X), len(self.base_models)))
        
        # Train base models with CV
        for model_name, model in self.base_models.items():
            logger.info(f"Training {model_name}...")
            
            fold_scores = []
            fold_preds = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np), 1):
                X_train, X_val = X_np[train_idx], X_np[val_idx]
                y_train, y_val = y_np[train_idx], y_np[val_idx]
                
                # Clone and train model
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)
                
                # Validate
                y_pred = model_clone.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                fold_scores.append({"rmse": rmse, "r2": r2})
                fold_preds.append((val_idx, y_pred))
                
                logger.info(f"  Fold {fold}/{self.n_splits}: RMSE={rmse:.2f}, R²={r2:.3f}")
            
            # Store CV scores
            self.cv_scores[model_name] = fold_scores
            
            # Collect out-of-fold predictions
            for val_idx, y_pred in fold_preds:
                oof_preds[val_idx, list(self.base_models.keys()).index(model_name)] = y_pred
            
            # Train final model on full data
            logger.info(f"Training final {model_name} on full dataset")
            model.fit(X_np, y_np)
            
            # Log feature importance for tree models
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_features = sorted(
                    zip(self.feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.info(f"  Top 5 features: {top_features}")
        
        # Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner (Ridge)...")
        self.meta_model.fit(oof_preds, y_np)
        
        # Final evaluation on OOF predictions
        final_pred = self.meta_model.predict(oof_preds)
        final_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_np, final_pred))),
            "r2": float(r2_score(y_np, final_pred)),
            "mape": float(mean_absolute_percentage_error(y_np, final_pred) * 100)
        }
        
        logger.info(f"Final ensemble metrics: RMSE={final_metrics['rmse']:.2f}, "
                   f"R²={final_metrics['r2']:.3f}, MAPE={final_metrics['mape']:.2f}%")
        
        return final_metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        return_base_preds: bool = False
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix (N, F).
            return_base_preds: If True, also return base model predictions.
            
        Returns:
            Predictions (N,) or tuple (predictions, base_predictions).
        """
        if not self.feature_names:
            raise ValueError("Model not trained")
        
        # Ensure feature order matches training
        X = X[self.feature_names]
        X_np = X.values
        
        # Get base model predictions
        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (model_name, model) in enumerate(self.base_models.items()):
            base_preds[:, i] = model.predict(X_np)
        
        # Meta-learner prediction
        predictions = self.meta_model.predict(base_preds)
        
        if return_base_preds:
            return predictions, base_preds
        return predictions
    
    def save(self, path: Path):
        """
        Save ensemble to disk.
        
        Args:
            path: Path to save model (will be pickled).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "config": self.config,
            "base_models": self.base_models,
            "meta_model": self.meta_model,
            "feature_names": self.feature_names,
            "cv_scores": self.cv_scores
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: Path):
        """
        Load ensemble from disk.
        
        Args:
            path: Path to saved model.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.config = model_data["config"]
        self.base_models = model_data["base_models"]
        self.meta_model = model_data["meta_model"]
        self.feature_names = model_data["feature_names"]
        self.cv_scores = model_data["cv_scores"]
        
        logger.info(f"Ensemble loaded from {path}")
    
    def _clone_model(self, model):
        """Clone a model with same parameters."""
        if isinstance(model, xgb.XGBRegressor):
            return xgb.XGBRegressor(**model.get_params())
        elif isinstance(model, RandomForestRegressor):
            return RandomForestRegressor(**model.get_params())
        elif isinstance(model, lgb.LGBMRegressor):
            return lgb.LGBMRegressor(**model.get_params())
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
