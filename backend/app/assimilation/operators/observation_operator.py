"""
backend/app/assimilation/operators/observation_operator.py
============================================================

Explicit observation operator abstraction for EnKF data assimilation.

Mathematical foundation:
    In data assimilation, the observation equation relates the true (or forecast)
    state x in R^n to an observation vector y in R^m via:

        y = h(x) + v,   v ~ N(0, R)

    where:
        x: State vector of dimension n (or ensemble matrix X_f of shape n x N)
        h: Observation operator mapping state space R^n -> observation space R^m
        y: Observation vector in R^m
        v: Observation error noise with covariance matrix R in R^(m x m)

    For direct observations of state variables (e.g. LAI from Sentinel-2 or Soil
    Moisture from sensors), h(x) is linear:
        h(x) = H * x
    where H in R^(m x n) is a selection matrix with H_{i, j} = 1 for the state
    variable index j corresponding to observation i.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from backend.app.assimilation.state.state_vector import STATE_DIM, STATE_INDEX, STATE_VARIABLES


class BaseObservationOperator(ABC):
    """Abstract Base Class for observation operators h(x)."""

    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply observation operator h(x) to state vector x or ensemble X.

        Args:
            x: 1D array of shape (n,) or 2D ensemble matrix of shape (n, N).

        Returns:
            Observed prediction: 1D array of shape (m,) or 2D matrix of shape (m, N).
        """
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Shorthand alias for apply(x)."""
        return self.apply(x)

    def observe_with_noise(
        self,
        x: np.ndarray,
        R: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample synthetic observation y = h(x) + v, where v ~ N(0, R).

        Args:
            x: State vector (n,) or ensemble (n, N).
            R: Observation error covariance matrix of shape (m, m).
            seed: Optional random seed for reproducibility.

        Returns:
            Observation y of shape (m,) or perturbed observation matrix (m, N).
        """
        hx = self.apply(x)
        m = R.shape[0]
        rng = np.random.default_rng(seed)

        if hx.ndim == 1:
            v = rng.multivariate_normal(np.zeros(m), R)
            return hx + v
        elif hx.ndim == 2:
            N = hx.shape[1]
            V = rng.multivariate_normal(np.zeros(m), R, size=N).T
            return hx + V
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {hx.shape}")

    @property
    def matrix(self) -> Optional[np.ndarray]:
        """Return explicit linear matrix H (m, n) if operator is linear, else None."""
        return None


class DirectObservationOperator(BaseObservationOperator):
    """Observation operator for direct state variable measurements (h(x) = H * x).

    Maps selected state vector indices to observation space via linear matrix H.
    Used for direct LAI, SM, TAGP, etc. observations.

    Args:
        observed_indices: List or 1D array of state variable indices [0..n-1]
                          corresponding to observations in vector y.
        state_dim: Full state space dimension n (default STATE_DIM=10).
    """

    def __init__(
        self,
        observed_indices: Union[list[int], np.ndarray],
        state_dim: int = STATE_DIM,
    ) -> None:
        self.observed_indices = np.array(observed_indices, dtype=int)
        self.state_dim = state_dim
        self.m = len(self.observed_indices)

        # Construct H matrix of shape (m, n)
        H = np.zeros((self.m, self.state_dim), dtype=np.float64)
        for i, idx in enumerate(self.observed_indices):
            if 0 <= idx < self.state_dim:
                H[i, idx] = 1.0
            else:
                raise ValueError(
                    f"Index {idx} out of range for state_dim={self.state_dim}."
                )
        self._H = H

    @classmethod
    def from_variable_names(
        cls,
        variables: list[str],
        state_dim: int = STATE_DIM,
    ) -> "DirectObservationOperator":
        """Construct DirectObservationOperator from variable names (e.g. ['lai', 'sm'])."""
        indices = []
        for var in variables:
            var_lower = var.lower()
            if var_lower not in STATE_INDEX:
                raise KeyError(f"Variable {var!r} not in STATE_VARIABLES: {STATE_VARIABLES}")
            indices.append(STATE_INDEX[var_lower])
        return cls(observed_indices=indices, state_dim=state_dim)

    @property
    def matrix(self) -> np.ndarray:
        """Return the linear selection matrix H of shape (m, n)."""
        return self._H

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Compute h(x) = H * x or H * X_f.

        Args:
            x: 1D state vector (n,) or 2D ensemble matrix (n, N).

        Returns:
            Observation projection of shape (m,) or (m, N).
        """
        if x.ndim == 1:
            if x.shape[0] != self.state_dim:
                raise ValueError(
                    f"State vector length {x.shape[0]} does not match operator state_dim={self.state_dim}."
                )
            return self._H @ x
        elif x.ndim == 2:
            if x.shape[0] != self.state_dim:
                raise ValueError(
                    f"State ensemble rows {x.shape[0]} do not match operator state_dim={self.state_dim}."
                )
            return self._H @ x
        else:
            raise ValueError(f"Expected 1D or 2D array for x, got {x.ndim}D.")


class ObservationModel:
    """Encapsulates the complete observation equation y = h(x) + v, v ~ N(0, R).

    Attributes:
        operator: BaseObservationOperator mapping state space to observation space.
        R: Observation error covariance matrix of shape (m, m).
    """

    def __init__(
        self,
        operator: BaseObservationOperator,
        R: np.ndarray,
    ) -> None:
        self.operator = operator
        self.R = np.ascontiguousarray(R, dtype=np.float64)

        if self.R.ndim != 2 or self.R.shape[0] != self.R.shape[1]:
            raise ValueError(f"R must be a square 2D matrix, got shape {self.R.shape}")

        if self.operator.matrix is not None:
            m = self.operator.matrix.shape[0]
            if self.R.shape[0] != m:
                raise ValueError(
                    f"R dimension ({self.R.shape[0]}) does not match operator observation dimension ({m})."
                )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Compute noiseless observation prediction y_hat = h(x)."""
        return self.operator.apply(x)

    def sample_observations(self, x: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Sample synthetic observations y = h(x) + v."""
        return self.operator.observe_with_noise(x, self.R, seed=seed)
