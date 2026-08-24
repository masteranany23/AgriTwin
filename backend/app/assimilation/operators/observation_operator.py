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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from backend.app.assimilation.state.state_vector import STATE_DIM, STATE_INDEX, STATE_VARIABLES


class UnsupportedObservationError(ValueError):
    """Raised when an observation cannot be mapped to the model state vector due to missing vertical/hydrological model information."""
    pass


class BaseObservationOperator(ABC):
    """Abstract Base Class for observation operators h(x).

    Attributes:
        observation_depth: Sensing depth or vertical profile range (e.g., "0-5 cm", "0-100 cm").
        observation_support: Physical/spatial support type (e.g., "surface_skin", "root_zone", "direct_state").
        model_target_variable: Targeted WOFOST state variable (e.g., "SM", "LAI").
        operator_type: Identifier of operator implementation class.
        uncertainty: Standard deviation of observation error.
    """

    def __init__(
        self,
        observation_depth: str = "unspecified",
        observation_support: str = "unspecified",
        model_target_variable: str = "unspecified",
        operator_type: str = "BaseObservationOperator",
        uncertainty: Optional[float] = None,
    ) -> None:
        self.observation_depth = observation_depth
        self.observation_support = observation_support
        self.model_target_variable = model_target_variable
        self.operator_type = operator_type
        self.uncertainty = uncertainty

    def get_metadata(self) -> dict[str, Any]:
        """Return comprehensive metadata for observation provenance audit."""
        return {
            "observation_depth": self.observation_depth,
            "observation_support": self.observation_support,
            "model_target_variable": self.model_target_variable,
            "operator_type": self.operator_type,
            "uncertainty": self.uncertainty,
        }

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
        """Sample synthetic observation y = h(x) + v, where v ~ N(0, R)."""
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
    Used for direct LAI, root-zone SM, TAGP, etc. observations where the sensor
    directly measures the target model variable.
    """

    def __init__(
        self,
        observed_indices: Union[list[int], np.ndarray],
        state_dim: int = STATE_DIM,
        observation_depth: str = "0-100 cm",
        observation_support: str = "root_zone",
        model_target_variable: str = "direct_state",
        uncertainty: Optional[float] = None,
    ) -> None:
        super().__init__(
            observation_depth=observation_depth,
            observation_support=observation_support,
            model_target_variable=model_target_variable,
            operator_type="DirectObservationOperator",
            uncertainty=uncertainty,
        )
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
        uncertainty: Optional[float] = None,
    ) -> "DirectObservationOperator":
        """Construct DirectObservationOperator from variable names (e.g. ['lai', 'sm'])."""
        indices = []
        for var in variables:
            var_lower = var.lower()
            if var_lower not in STATE_INDEX:
                raise KeyError(f"Variable {var!r} not in STATE_VARIABLES: {STATE_VARIABLES}")
            indices.append(STATE_INDEX[var_lower])
        return cls(
            observed_indices=indices,
            state_dim=state_dim,
            model_target_variable=",".join(variables),
            uncertainty=uncertainty,
        )

    @property
    def matrix(self) -> np.ndarray:
        """Return the linear selection matrix H of shape (m, n)."""
        return self._H

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Compute h(x) = H * x or H * X_f."""
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


class SurfaceSoilMoistureObservationOperator(BaseObservationOperator):
    """Observation operator for surface-sensitive remote sensing soil moisture (h(x) = h_surf(x)).

    Used for satellite microwave radiometry/radar (e.g. SMAP, Sentinel-1 SAR) that senses only the
    top surface skin layer (0-5 cm).

    Scientific Invariant:
    - Does NOT assume surface SM equals root-zone SM.
    - Does NOT invent arbitrary conversion coefficients or fabricate synthetic observations.
    - Does NOT modify EnKF mathematics.
    - If vertical hydrology / 1D soil layer model information is unconfigured/unavailable, fails explicitly
      by raising UnsupportedObservationError rather than silently performing an invalid direct mapping.
    """

    def __init__(
        self,
        observed_indices: Union[list[int], np.ndarray],
        state_dim: int = STATE_DIM,
        observation_depth: str = "0-5 cm",
        observation_support: str = "surface_skin",
        model_target_variable: str = "SM",
        uncertainty: Optional[float] = 0.04,
        hydrology_model: Optional[Any] = None,
    ) -> None:
        super().__init__(
            observation_depth=observation_depth,
            observation_support=observation_support,
            model_target_variable=model_target_variable,
            operator_type="SurfaceSoilMoistureObservationOperator",
            uncertainty=uncertainty,
        )
        self.observed_indices = np.array(observed_indices, dtype=int)
        self.state_dim = state_dim
        self.m = len(self.observed_indices)
        self.hydrology_model = hydrology_model

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply surface observation operator.

        Raises UnsupportedObservationError when no vertical hydrology model is present,
        preventing silent direct mapping of 0-5 cm surface observations to WOFOST root-zone SM.
        """
        if self.hydrology_model is not None and hasattr(self.hydrology_model, "get_surface_moisture"):
            return self.hydrology_model.get_surface_moisture(x)

        raise UnsupportedObservationError(
            "Surface-to-root-zone soil moisture transformation is unsupported because vertical "
            "hydrological/soil layer discretization is not configured. Direct mapping of 0-5 cm "
            "surface remote sensing observations to WOFOST root-zone SM is non-equivalent and rejected."
        )


def get_observation_operator(
    variable_name: str,
    source: Optional[str] = None,
    observed_indices: Optional[Union[list[int], np.ndarray]] = None,
    uncertainty: Optional[float] = None,
    hydrology_model: Optional[Any] = None,
    state_dim: int = STATE_DIM,
) -> BaseObservationOperator:
    """Factory function selecting the appropriate observation operator based on variable and source provenance.

    Args:
        variable_name: Name of observed variable (e.g., 'SM', 'LAI').
        source: Provenance source string (e.g., 'SATELLITE', 'SENTINEL1_SAR', 'SENSOR', 'IOT_SENSOR').
        observed_indices: State vector indices observed.
        uncertainty: Observation uncertainty standard deviation.
        hydrology_model: Optional vertical soil hydrology transformation module.
        state_dim: Dimension of state vector.

    Returns:
        BaseObservationOperator (DirectObservationOperator or SurfaceSoilMoistureObservationOperator).
    """
    var_upper = str(variable_name).upper()
    src_upper = str(source).upper() if source else ""

    if observed_indices is None:
        if var_upper in ("ROOT_ZONE_SOIL_MOISTURE", "ROOT_ZONE_SM", "SURFACE_SOIL_MOISTURE", "SURFACE_SM"):
            idx = STATE_INDEX.get("sm")
        else:
            idx = STATE_INDEX.get(var_upper.lower())
        observed_indices = [idx] if idx is not None else [1]

    # Identify remote sensing or explicit surface soil moisture
    is_surface_sm = (
        var_upper in ("SURFACE_SOIL_MOISTURE", "SURFACE_SM") or
        (var_upper == "SM" and src_upper in ("SATELLITE", "SENTINEL1", "SENTINEL1_SAR", "SENTINEL-1", "SMAP"))
    )

    if is_surface_sm:
        return SurfaceSoilMoistureObservationOperator(
            observed_indices=observed_indices,
            state_dim=state_dim,
            observation_depth="0-5 cm",
            observation_support="surface_skin",
            model_target_variable="SM",
            uncertainty=uncertainty,
            hydrology_model=hydrology_model,
        )

    depth = "0-100 cm" if var_upper in ("SM", "ROOT_ZONE_SOIL_MOISTURE", "ROOT_ZONE_SM") else "canopy/crop"
    support = "root_zone" if var_upper in ("SM", "ROOT_ZONE_SOIL_MOISTURE", "ROOT_ZONE_SM") else "direct_state"

    return DirectObservationOperator(
        observed_indices=observed_indices,
        state_dim=state_dim,
        observation_depth=depth,
        observation_support=support,
        model_target_variable=var_upper,
        uncertainty=uncertainty,
    )


class ObservationModel:
    """Encapsulates the complete observation equation y = h(x) + v, v ~ N(0, R)."""

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

