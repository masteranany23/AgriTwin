"""
tests/test_observation_operator.py
===================================

Unit tests for the observation-operator abstraction layer (BaseObservationOperator,
DirectObservationOperator, ObservationModel) and verification of EnKF equivalence.
"""

import numpy as np
import pytest

from backend.app.assimilation.filters.enkf import enkf_update
from backend.app.assimilation.operators.observation_operator import (
    BaseObservationOperator,
    DirectObservationOperator,
    ObservationModel,
)
from backend.app.assimilation.state.state_vector import STATE_DIM, STATE_INDEX


def test_direct_observation_operator_eval():
    """Verify DirectObservationOperator correctly maps state vectors and ensembles."""
    # Observe LAI (idx 0) and SM (idx 1)
    op = DirectObservationOperator(observed_indices=[0, 1], state_dim=STATE_DIM)
    
    assert op.m == 2
    assert op.matrix.shape == (2, STATE_DIM)
    assert op.matrix[0, 0] == 1.0
    assert op.matrix[1, 1] == 1.0

    # 1D test vector
    x = np.array([2.5, 0.3, 100.0, 50.0, 0.9, 30.0, 20.0, 10.0, 0.5, 25.0])
    y_pred = op.apply(x)
    np.testing.assert_array_almost_equal(y_pred, [2.5, 0.3])

    # 2D ensemble matrix (10, 5)
    rng = np.random.default_rng(42)
    X_f = rng.standard_normal((STATE_DIM, 5))
    Y_pred = op.apply(X_f)
    assert Y_pred.shape == (2, 5)
    np.testing.assert_array_almost_equal(Y_pred[0, :], X_f[0, :])
    np.testing.assert_array_almost_equal(Y_pred[1, :], X_f[1, :])


def test_direct_observation_operator_from_variable_names():
    """Verify factory construction from variable names."""
    op = DirectObservationOperator.from_variable_names(["lai", "rftra"])
    assert op.observed_indices.tolist() == [STATE_INDEX["lai"], STATE_INDEX["rftra"]]


def test_observation_model_sampling():
    """Verify ObservationModel prediction and noise sampling."""
    op = DirectObservationOperator([0, 1], state_dim=STATE_DIM)
    R = np.diag([0.04, 0.0025])  # std 0.2 and 0.05
    model = ObservationModel(operator=op, R=R)

    x = np.array([3.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    y_pred = model.predict(x)
    np.testing.assert_array_almost_equal(y_pred, [3.0, 0.25])

    y_noisy = model.sample_observations(x, seed=123)
    assert y_noisy.shape == (2,)
    # Sampled noise should be close to mean
    assert abs(y_noisy[0] - 3.0) < 1.0
    assert abs(y_noisy[1] - 0.25) < 0.2


def test_enkf_update_equivalence_explicit_vs_default():
    """Prove that enkf_update with explicit DirectObservationOperator yields bit-for-bit identical results."""
    n, N = STATE_DIM, 20
    rng = np.random.default_rng(999)

    # Forecast ensemble
    X_f = rng.normal(loc=2.0, scale=0.5, size=(n, N))

    # Partial observation vector y (LAI=2.8, SM=0.30)
    y = np.full(n, np.nan)
    y[0] = 2.8
    y[1] = 0.30

    # Observation error covariance matrix R
    R = np.eye(n) * 0.1

    # 1. Update without explicit H_operator (default fallback)
    # Set seed to ensure identical stochastic perturbation
    np.random.seed(42)
    X_a_default, d_default, K_default = enkf_update(X_f, y, R, H_operator=None)

    # 2. Update with explicit DirectObservationOperator
    np.random.seed(42)
    op = DirectObservationOperator(observed_indices=[0, 1], state_dim=n)
    X_a_explicit, d_explicit, K_explicit = enkf_update(X_f, y, R, H_operator=op)

    # Results must be identical to machine precision
    np.testing.assert_array_almost_equal(X_a_default, X_a_explicit, decimal=14)
    np.testing.assert_array_almost_equal(d_default, d_explicit, decimal=14)
    np.testing.assert_array_almost_equal(K_default, K_explicit, decimal=14)
