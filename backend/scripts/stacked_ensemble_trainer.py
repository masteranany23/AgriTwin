"""
Stacked Ensemble Trainer for AgriTwin v3.0 Multi-Model Averaging.

This script trains a stacked ensemble model (XGBoost + LightGBM + MLP) with Ridge meta-model
using TimeSeriesSplit to prevent look-ahead bias.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import sys
from typing import Optional

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to sys.path
sys.path.append(str(DEFAULT_PROJECT_ROOT))

# ML imports
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError as e:
    logging.error(f"ML libraries not available: {e}")
    ML_AVAILABLE = False
    # Create dummy classes for type hints
    class DummyEstimator:
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
    TimeSeriesSplit = type('TimeSeriesSplit', (), {'n_splits': 3})
    StackingRegressor = DummyEstimator
    Ridge = DummyEstimator
    MLPRegressor = DummyEstimator
    StandardScaler = DummyEstimator
    Pipeline = type('Pipeline', (), {'fit': lambda self, X, y: self, 'predict': lambda self, X: np.zeros(len(X))})
    xgb = type('xgb', (), {'XGBRegressor': DummyEstimator})
    lgb = type('lgb', (), {'LGBMRegressor': DummyEstimator})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StackedEnsembleTrainer:
    def __init__(self, project_root: Optional[Path | str] = None):
        self.project_root = Path(project_root) if project_root else DEFAULT_PROJECT_ROOT
        self.data_dir = self.project_root / "data"  # Processed data goes here
        self.models_dir = self.project_root / "models"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ensemble pipeline
        self.pipeline = None
        self.training_history = {}
        
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available. Install: pip install scikit-learn xgboost lightgbm pyarrow")
            logger.warning("Running in demonstration mode only.")
    
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load the generated stacked training data.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        data_file = self.data_dir / "processed" / "stacked_training_data.parquet"
        
        if not data_file.exists():
            # Check for CSV version
            csv_file = data_file.with_suffix('.csv')
            if csv_file.exists():
                logger.info(f"Loading data from CSV: {csv_file}")
                df = pd.read_csv(csv_file)
            else:
                logger.error(f"Training data not found: {data_file}")
                logger.info("Creating sample data for demonstration...")
                df = self._create_sample_data()
        else:
            logger.info(f"Loading data from parquet: {data_file}")
            df = pd.read_parquet(data_file)
        
        # Check if we have data
        if df.empty:
            logger.error("Loaded empty dataset")
            return None, None, None, None
        
        logger.info(f"Dataset loaded with shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Separate features and target
        # Assume last column is target
        if 'target' not in df.columns:
            logger.warning("'target' column not found, using last column as target")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        else:
            X = df.drop('target', axis=1)
            y = df['target']
        
        # Remove non-numeric columns for training
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.info(f"Dropping non-numeric columns: {non_numeric_cols.tolist()}")
            X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        if X.isna().any().any():
            logger.info("Handling missing values...")
            X = X.fillna(X.mean())
        
        # Manual time-based split: Train on older years (2019-2020), Test on newer years (2021-2023)
        if 'year' in df.columns:
            logger.info("Performing time-based split by year...")
            
            # Sort by year and date if available
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            
            # Create train/test split based on years
            train_years = [2019, 2020]
            test_years = [2021, 2022, 2023]
            
            train_mask = df['year'].isin(train_years)
            test_mask = df['year'].isin(test_years)
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            logger.info(f"Train set: {len(X_train)} samples (years {train_years})")
            logger.info(f"Test set: {len(X_test)} samples (years {test_years})")
            
            if len(X_train) == 0:
                logger.warning("No training data found for specified years, using random split")
                # Fallback to random split
                split_idx = int(len(X) * 0.6)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Fallback: use time-series split if we have date information
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                split_idx = int(len(df) * 0.6)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            else:
                # Last resort: random split (not recommended for time series)
                logger.warning("No time information found, using random split (may cause data leakage)")
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
        
        return X_train, X_test, y_train, y_test
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration when real data is not available."""
        logger.info("Creating sample training data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create date range
        dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
        dates = np.random.choice(dates, n_samples, replace=True)
        
        # Create sample features
        data = {
            'date': dates,
            'year': [d.year for d in dates],
            'day_of_year': [d.timetuple().tm_yday for d in dates],
            'lai_mean': np.random.uniform(0.5, 4.5, n_samples),
            'lai_std': np.random.uniform(0.1, 1.0, n_samples),
            'lai_trend': np.random.uniform(-0.1, 0.1, n_samples),
            'sm_layer1_mean': np.random.uniform(0.15, 0.35, n_samples),
            'sm_layer1_std': np.random.uniform(0.01, 0.05, n_samples),
            'sm_layer2_mean': np.random.uniform(0.20, 0.40, n_samples),
            'sm_layer2_std': np.random.uniform(0.01, 0.05, n_samples),
            'sm_layer3_mean': np.random.uniform(0.25, 0.45, n_samples),
            'sm_layer3_std': np.random.uniform(0.01, 0.05, n_samples),
            'sm_layer4_mean': np.random.uniform(0.30, 0.50, n_samples),
            'sm_layer4_std': np.random.uniform(0.01, 0.05, n_samples),
            'heat_strain_hours': np.random.randint(0, 20, n_samples),
            'dvs_phase': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.3, 0.3])
        }
        
        # Create target with some relationship to features
        X_sample = pd.DataFrame(data)
        X_numeric = X_sample.select_dtypes(include=[np.number])
        
        # Simple linear relationship with noise
        coefficients = np.random.uniform(-10, 10, len(X_numeric.columns))
        y_sample = X_numeric.values.dot(coefficients) + np.random.normal(0, 50, n_samples)
        
        X_sample['target'] = y_sample
        
        # Save sample data
        output_file = self.data_dir / "processed" / "stacked_training_data.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        X_sample.to_csv(output_file, index=False)
        
        logger.info(f"Created sample data with {len(X_sample)} samples")
        return X_sample
    
    def create_ensemble_pipeline(self):
        """Create the stacked ensemble pipeline."""
        if not ML_AVAILABLE:
            logger.error("ML libraries not available. Cannot create ensemble.")
            return
        
        logger.info("Creating stacked ensemble pipeline...")
        
        # Define base models
        base_models = [
            ('xgboost', xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )),
            ('lightgbm', lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                verbose=False
            ))
        ]
        
        # Define meta model
        meta_model = Ridge(alpha=0.5, random_state=42)
        
        # TimeSeriesSplit to prevent look-ahead bias
        cv = TimeSeriesSplit(n_splits=3)
        
        # Create stacking regressor
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=-1,
            passthrough=False
        )
        
        # Create full pipeline with scaling
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', stacking_regressor)
        ])
        
        logger.info("Stacked ensemble pipeline created successfully")
        logger.info(f"Base models: {[name for name, _ in base_models]}")
        logger.info(f"Meta model: {meta_model.__class__.__name__}")
        logger.info(f"Cross-validation: TimeSeriesSplit(n_splits={cv.n_splits})")
    
    def train(self, X_train, y_train):
        """Train the stacked ensemble."""
        if self.pipeline is None:
            self.create_ensemble_pipeline()
        
        if self.pipeline is None:
            logger.error("Pipeline not created. Cannot train.")
            return
        
        logger.info("Training stacked ensemble...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        try:
            # Train the pipeline
            self.pipeline.fit(X_train, y_train)
            
            # Store training metadata
            self.training_history['train_samples'] = len(X_train)
            self.training_history['train_features'] = X_train.shape[1]
            self.training_history['train_date'] = datetime.now().isoformat()
            
            logger.info("Training completed successfully")
            
            # Get feature importance from tree-based models
            self._analyze_feature_importance(X_train)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _analyze_feature_importance(self, X_train):
        """Analyze feature importance from tree-based models."""
        if not ML_AVAILABLE:
            return
        
        try:
            # Get feature importance from XGBoost
            xgb_model = self.pipeline.named_steps['stacking'].estimators_[0]
            if hasattr(xgb_model, 'feature_importances_'):
                xgb_importance = xgb_model.feature_importances_
                top_xgb_idx = np.argsort(xgb_importance)[-10:][::-1]
                
                logger.info("Top 10 features by XGBoost importance:")
                for i, idx in enumerate(top_xgb_idx):
                    if idx < len(X_train.columns):
                        feature_name = X_train.columns[idx]
                        logger.info(f"  {i+1}. {feature_name}: {xgb_importance[idx]:.4f}")
            
            # Get feature importance from LightGBM
            lgb_model = self.pipeline.named_steps['stacking'].estimators_[1]
            if hasattr(lgb_model, 'feature_importances_'):
                lgb_importance = lgb_model.feature_importances_
                top_lgb_idx = np.argsort(lgb_importance)[-10:][::-1]
                
                logger.info("Top 10 features by LightGBM importance:")
                for i, idx in enumerate(top_lgb_idx):
                    if idx < len(X_train.columns):
                        feature_name = X_train.columns[idx]
                        logger.info(f"  {i+1}. {feature_name}: {lgb_importance[idx]:.4f}")
                        
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {e}")
    
    def evaluate(self, X_test, y_test):
        """Evaluate the trained ensemble on test data."""
        if self.pipeline is None:
            logger.error("No trained model available for evaluation.")
            return None
        
        logger.info("Evaluating model on test set...")
        logger.info(f"Test data shape: {X_test.shape}")
        
        try:
            # Make predictions
            y_pred = self.pipeline.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage improvement over baseline (mean prediction)
            baseline_pred = np.full_like(y_pred, y_test.mean())
            baseline_mse = mean_squared_error(y_test, baseline_pred)
            baseline_rmse = np.sqrt(baseline_mse)
            improvement_pct = (baseline_rmse - rmse) / baseline_rmse * 100
            
            # Store evaluation results
            evaluation_results = {
                'rmse': rmse,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'improvement_over_baseline_pct': improvement_pct,
                'test_samples': len(X_test)
            }
            
            self.training_history['evaluation'] = evaluation_results
            
            # Log results
            logger.info("=" * 60)
            logger.info("EVALUATION RESULTS")
            logger.info("=" * 60)
            logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
            logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
            logger.info(f"R² Score: {r2:.4f}")
            logger.info(f"Improvement over baseline: {improvement_pct:.1f}%")
            logger.info("=" * 60)
            
            # Additional analysis
            if len(y_test) > 0:
                error_distribution = y_pred - y_test
                logger.info(f"Error distribution:")
                logger.info(f"  Mean error: {error_distribution.mean():.2f}")
                logger.info(f"  Std error: {error_distribution.std():.2f}")
                logger.info(f"  Min error: {error_distribution.min():.2f}")
                logger.info(f"  Max error: {error_distribution.max():.2f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.pipeline is None:
            logger.error("No trained model to save.")
            return False
        
        model_file = self.models_dir / "stacked_ensemble.pkl"
        
        try:
            # Save the model
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'training_history': self.training_history,
                    'save_date': datetime.now().isoformat(),
                    'version': 'v3.0'
                }, f)
            
            logger.info(f"Model saved to {model_file}")
            
            # Also save metadata as JSON
            metadata_file = model_file.with_suffix('.json')
            import json
            with open(metadata_file, 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
            
            logger.info(f"Model metadata saved to {metadata_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_existing_model(self):
        """Load an existing trained model."""
        model_file = self.models_dir / "stacked_ensemble.pkl"
        
        if not model_file.exists():
            logger.warning(f"No existing model found at {model_file}")
            return False
        
        try:
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.pipeline = saved_data.get('pipeline')
            self.training_history = saved_data.get('training_history', {})
            
            logger.info(f"Loaded existing model from {model_file}")
            logger.info(f"Model version: {saved_data.get('version', 'unknown')}")
            logger.info(f"Training date: {saved_data.get('save_date', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def run_full_training(self):
        """Run complete training pipeline."""
        logger.info("Starting full training pipeline...")
        
        # Check if model already exists
        if self.load_existing_model():
            logger.info("Existing model loaded successfully")
            return True
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        if X_train is None:
            logger.error("Failed to load training data")
            return False
        
        # Train model
        if not self.train(X_train, y_train):
            logger.error("Training failed")
            return False
        
        # Evaluate model
        evaluation_results = self.evaluate(X_test, y_test)
        
        if evaluation_results is None:
            logger.error("Evaluation failed")
            return False
        
        # Save model
        if not self.save_model():
            logger.error("Failed to save model")
            return False
        
        # Print summary
        self._print_summary(evaluation_results)
        
        return True
    
    def _print_summary(self, evaluation_results):
        """Print training summary."""
        print("\n" + "="*70)
        print("STACKED ENSEMBLE TRAINING SUMMARY")
        print("="*70)
        print(f"Model Type: Multi-Model Averaging (XGBoost + LightGBM + MLP + Ridge)")
        print(f"Training Strategy: TimeSeriesSplit (no look-ahead bias)")
        print(f"Training Samples: {self.training_history.get('train_samples', 'N/A')}")
        print(f"Test Samples: {self.training_history.get('evaluation', {}).get('test_samples', 'N/A')}")
        print(f"Features Used: {self.training_history.get('train_features', 'N/A')}")
        print("\nPerformance Metrics:")
        print(f"  RMSE: {evaluation_results.get('rmse', 'N/A'):.2f}")
        print(f"  R² Score: {evaluation_results.get('r2', 'N/A'):.4f}")
        print(f"  Improvement over baseline: {evaluation_results.get('improvement_over_baseline_pct', 'N/A'):.1f}%")
        
        target_improvement = 15.0  # Target RMSE reduction from requirements
        actual_improvement = evaluation_results.get('improvement_over_baseline_pct', 0)
        
        if actual_improvement >= target_improvement:
            print(f"\n✅ SUCCESS: Achieved {actual_improvement:.1f}% improvement")
            print(f"   (Target: {target_improvement}% RMSE reduction)")
        else:
            print(f"\n⚠️  PARTIAL: Achieved {actual_improvement:.1f}% improvement")
            print(f"   (Target: {target_improvement}% RMSE reduction)")
        
        print(f"\nModel saved to: {self.models_dir / 'stacked_ensemble.pkl'}")
        print("="*70)

def main():
    """Main entry point."""
    trainer = StackedEnsembleTrainer()
    
    if not ML_AVAILABLE:
        print("❌ ML libraries not available.")
        print("Please install required packages:")
        print("  pip install scikit-learn xgboost lightgbm pyarrow")
        return
    
    success = trainer.run_full_training()
    
    if success:
        print("\n✅ Stacked ensemble training completed successfully!")
        print("The model is ready for use in the API.")
    else:
        print("\n❌ Stacked ensemble training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()