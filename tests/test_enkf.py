"""
tests/test_enkf.py
==================

Unit tests for the stochastic Ensemble Kalman Filter mathematical implementation.
"""

import numpy as np


from backend.app.assimilation.filters.enkf import enkf_update


def test_enkf_update_full_observation():
    n = 3  # state dimension
    N = 1000  # large ensemble to test statistical properties
    
    # Generate ensemble
    X_f = np.random.randn(n, N) * 2.0  # Forecast variance ~ 4.0
    
    # True state
    x_true = np.array([5.0, -2.0, 10.0])
    
    # Observation with noise
    R = np.eye(n) * 1.0  # Obs variance = 1.0
    y = x_true + np.random.multivariate_normal(np.zeros(n), R)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # Shape checks
    assert X_a.shape == (n, N)
    assert d.shape == (n,)
    assert K.shape == (n, n)
    
    # Since R is much smaller than forecast variance, analysis mean should pull strongly toward y
    x_a_mean = np.mean(X_a, axis=1)
    
    # Check that x_a_mean is closer to y than the original forecast mean (~0.0)
    dist_forecast_to_y = np.linalg.norm(np.mean(X_f, axis=1) - y)
    dist_analysis_to_y = np.linalg.norm(x_a_mean - y)
    assert dist_analysis_to_y < dist_forecast_to_y


def test_enkf_update_partial_observation():
    n = 3
    N = 100
    X_f = np.random.randn(n, N)
    
    # Only observe the first variable
    y = np.array([5.0, np.nan, np.nan])
    R = np.eye(n)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # Innovation should only exist for the observed variable
    assert not np.isnan(d[0])
    assert np.isnan(d[1])
    assert np.isnan(d[2])
    
    # K should only be non-zero in the first column
    assert np.any(K[:, 0] != 0.0)
    assert np.all(K[:, 1] == 0.0)
    assert np.all(K[:, 2] == 0.0)


def test_enkf_update_with_missing_state():
    n = 3
    N = 100
    X_f = np.random.randn(n, N)
    # The third variable is entirely missing (e.g. TWSO before emergence)
    X_f[2, :] = np.nan
    
    y = np.array([1.0, 2.0, 3.0])
    R = np.eye(n)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # The valid observations should only be 0 and 1, because X_f[2] is NaN
    assert not np.isnan(d[0])
    assert not np.isnan(d[1])
    assert np.isnan(d[2])  # Despite y having a value, it shouldn't be assimilated
    
    # The third variable in X_a should remain NaN
    assert np.all(np.isnan(X_a[2, :]))


def test_enkf_update_no_valid_observations():
    n = 3
    N = 10
    X_f = np.random.randn(n, N)
    y = np.array([np.nan, np.nan, np.nan])
    R = np.eye(n)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # Analysis should be exactly forecast
    np.testing.assert_array_equal(X_a, X_f)
    assert np.all(np.isnan(d))
    assert np.all(K == 0.0)


def test_enkf_update_singular_R_observation_covariance():
    """Verify stability when observation error matrix R is singular (rank-deficient)."""
    n = 3
    N = 50
    X_f = np.random.randn(n, N)
    y = np.array([2.0, 4.0, 6.0])
    
    # Zero observation error covariance (singular R)
    R_singular = np.zeros((n, n))
    
    X_a, d, K = enkf_update(X_f, y, R_singular)
    
    assert X_a.shape == (n, N)
    assert not np.any(np.isnan(X_a))
    assert not np.any(np.isnan(K))
    assert not np.any(np.isnan(d))


def test_enkf_update_ill_conditioned_collinear_ensemble():
    """Verify numeric stability when forecast ensemble matrix is zero-variance or collinear."""
    n = 3
    N = 50
    # Completely identical ensemble members -> HA @ HA.T is zero matrix
    X_f = np.ones((n, N)) * 5.0
    y = np.array([6.0, 7.0, 8.0])
    R = np.eye(n) * 0.1
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    assert X_a.shape == (n, N)
    assert not np.any(np.isnan(X_a))
    assert not np.any(np.isnan(K))


def test_enkf_update_nan_isolation():
    """Verify that NaNs in one state variable do NOT corrupt valid state variables or outputs."""
    n = 4
    N = 30
    np.random.seed(42)
    X_f = np.random.randn(n, N)
    
    # Introduce NaNs into unobserved state variable 3
    X_f[3, :] = np.nan
    
    # Observe variable 0 and 1
    y = np.array([3.0, 1.5, np.nan, 4.0])
    R = np.eye(n)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # State rows 0, 1, 2 must have ZERO NaNs in X_a
    assert not np.any(np.isnan(X_a[0, :]))
    assert not np.any(np.isnan(X_a[1, :]))
    assert not np.any(np.isnan(X_a[2, :]))
    
    # State row 3 must remain all NaNs in X_a
    assert np.all(np.isnan(X_a[3, :]))
    
    # Innovation vector d must be finite for obs 0 and 1, NaN for 2 and 3
    assert not np.isnan(d[0])
    assert not np.isnan(d[1])
    assert np.isnan(d[2])
    assert np.isnan(d[3])
    
    # Kalman gain K for row 3 must be zero
    assert np.all(K[3, :] == 0.0)


def test_enkf_update_inf_values_handled_safely():
    """Verify that inf / -inf values in observations or state variables are rejected as invalid."""
    n = 3
    N = 20
    X_f = np.random.randn(n, N)
    # Put inf in y[1]
    y = np.array([2.0, np.inf, 3.0])
    R = np.eye(n)
    
    X_a, d, K = enkf_update(X_f, y, R)
    
    # Variable 1 should be treated as unobserved
    assert not np.isnan(d[0])
    assert np.isnan(d[1])
    assert not np.isnan(d[2])
    assert np.all(K[:, 1] == 0.0)


def test_enkf_update_invalid_input_dimensions():
    """Verify that invalid matrix shapes and ensemble sizes raise informative ValueErrors."""
    X_f = np.random.randn(3, 10)
    y = np.array([1.0, 2.0, 3.0])
    R = np.eye(3)
    
    # 1D X_f
    with pytest.raises(ValueError, match="2D matrix"):
        enkf_update(X_f[:, 0], y, R)
        
    # N <= 1
    with pytest.raises(ValueError, match="N must be > 1"):
        enkf_update(X_f[:, :1], y, R)
        
    # Mismatched y length
    with pytest.raises(ValueError, match="vector of length n"):
        enkf_update(X_f, y[:2], R)
        
    # Mismatched R shape
    with pytest.raises(ValueError, match="covariance matrix of shape"):
        enkf_update(X_f, y, np.eye(2))
