"""
Deep Gaussian Process correction with spatial-temporal kernel.
"""
import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood


logger = logging.getLogger(__name__)


class SpatialTemporalKernel(gpytorch.kernels.Kernel):
    """
    Custom kernel: RBF(lat, lon) × RBF(year) for spatial-temporal smoothing.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Spatial kernel (lat, lon)
        self.spatial_kernel = ScaleKernel(
            RBFKernel(ard_num_dims=2, active_dims=[0, 1])
        )
        
        # Temporal kernel (year)
        self.temporal_kernel = ScaleKernel(
            RBFKernel(ard_num_dims=1, active_dims=[2])
        )
    
    def forward(self, x1, x2, diag=False, **params):
        """Compute kernel: k_spatial * k_temporal."""
        k_spatial = self.spatial_kernel(x1, x2, diag=diag, **params)
        k_temporal = self.temporal_kernel(x1, x2, diag=diag, **params)
        
        if diag:
            return k_spatial * k_temporal
        return k_spatial.mul(k_temporal)


class GPCorrectionModel(ExactGP):
    """
    Gaussian Process model for yield correction.
    
    Learns residual correction on top of ensemble predictions using
    spatial (lat/lon) and temporal (year) features.
    """
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = SpatialTemporalKernel()
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DeepGPCorrection:
    """
    Deep GP correction layer for ensemble predictions.
    
    Takes ensemble predictions and learns spatial-temporal residuals.
    """
    
    def __init__(self, config: dict):
        """
        Initialize GP correction.
        
        Args:
            config: Configuration with GP parameters.
        """
        self.config = config
        self.noise_variance = config.get("noise_variance", 0.1)
        self.training_iterations = config.get("training_iterations", 50)
        self.use_gpu = config.get("use_gpu", False) and torch.cuda.is_available()
        
        self.model: Optional[GPCorrectionModel] = None
        self.likelihood: Optional[GaussianLikelihood] = None
        
        # Normalization parameters
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None
        
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        logger.info(f"GP using device: {self.device}")
    
    def train(
        self,
        ensemble_preds: np.ndarray,
        true_yields: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        years: np.ndarray
    ):
        """
        Train GP on ensemble residuals.
        
        Args:
            ensemble_preds: Ensemble predictions (N,).
            true_yields: True yield values (N,).
            latitudes: Latitude for each sample (N,).
            longitudes: Longitude for each sample (N,).
            years: Year for each sample (N,).
        """
        logger.info(f"Training GP on {len(ensemble_preds)} samples")
        
        # Compute residuals
        residuals = true_yields - ensemble_preds
        
        # Prepare features: [lat, lon, year_normalized]
        year_norm = (years - years.min()) / (years.max() - years.min() + 1e-6)
        X = np.column_stack([latitudes, longitudes, year_norm])
        
        # Normalize inputs
        self.x_mean = torch.tensor(X.mean(axis=0), dtype=torch.float32)
        self.x_std = torch.tensor(X.std(axis=0) + 1e-6, dtype=torch.float32)
        X_norm = (X - self.x_mean.numpy()) / self.x_std.numpy()
        
        # Normalize targets
        self.y_mean = float(residuals.mean())
        self.y_std = float(residuals.std() + 1e-6)
        y_norm = (residuals - self.y_mean) / self.y_std
        
        # Convert to tensors
        train_x = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(y_norm, dtype=torch.float32).to(self.device)
        
        # Initialize model and likelihood
        self.likelihood = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        ).to(self.device)
        self.likelihood.noise = self.noise_variance
        
        self.model = GPCorrectionModel(train_x, train_y, self.likelihood).to(self.device)
        
        # Training mode
        self.model.train()
        self.likelihood.train()
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop
        logger.info("Optimizing GP hyperparameters...")
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Iteration {i+1}/{self.training_iterations}, Loss: {loss.item():.3f}")
        
        # Set to eval mode
        self.model.eval()
        self.likelihood.eval()
        
        logger.info("GP training complete")
        
        # Log learned lengthscales
        spatial_ls = self.model.covar_module.spatial_kernel.base_kernel.lengthscale.detach().cpu().numpy()
        temporal_ls = self.model.covar_module.temporal_kernel.base_kernel.lengthscale.detach().cpu().numpy()
        logger.info(f"Learned lengthscales - Spatial: {spatial_ls}, Temporal: {temporal_ls}")
    
    def predict(
        self,
        ensemble_preds: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        years: np.ndarray,
        return_variance: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict GP correction and apply to ensemble predictions.
        
        Args:
            ensemble_preds: Ensemble predictions (N,).
            latitudes: Latitude for each sample (N,).
            longitudes: Longitude for each sample (N,).
            years: Year for each sample (N,).
            return_variance: If True, return prediction variance.
            
        Returns:
            Corrected predictions (N,) and optionally variances (N,).
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Prepare features
        year_norm = (years - years.min()) / (years.max() - years.min() + 1e-6)
        X = np.column_stack([latitudes, longitudes, year_norm])
        
        # Normalize
        X_norm = (X - self.x_mean.numpy()) / self.x_std.numpy()
        test_x = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean.cpu().numpy()
            variance = predictions.variance.cpu().numpy() if return_variance else None
        
        # Denormalize
        mean_denorm = mean * self.y_std + self.y_mean
        
        # Apply correction
        corrected_preds = ensemble_preds + mean_denorm
        
        if return_variance:
            variance_denorm = variance * (self.y_std ** 2)
            return corrected_preds, variance_denorm
        
        return corrected_preds, None
    
    def save(self, path: Path):
        """
        Save GP model to disk.
        
        Args:
            path: Path to save model.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "config": self.config,
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        logger.info(f"GP model saved to {path}")
    
    def load(self, path: Path):
        """
        Load GP model from disk.
        
        Args:
            path: Path to saved model.
        """
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.config = model_data["config"]
        self.x_mean = model_data["x_mean"]
        self.x_std = model_data["x_std"]
        self.y_mean = model_data["y_mean"]
        self.y_std = model_data["y_std"]
        
        # Reconstruct model (need dummy data)
        dummy_x = torch.zeros(1, 3).to(self.device)
        dummy_y = torch.zeros(1).to(self.device)
        
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPCorrectionModel(dummy_x, dummy_y, self.likelihood).to(self.device)
        
        self.model.load_state_dict(model_data["model_state"])
        self.likelihood.load_state_dict(model_data["likelihood_state"])
        
        self.model.eval()
        self.likelihood.eval()
        
        logger.info(f"GP model loaded from {path}")
