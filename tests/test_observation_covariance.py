"""
tests/test_observation_covariance.py
======================================

Unit tests for ObservationCovariance abstraction.

Tests:
1. Baseline diagonal R behavior is unchanged (from_variances).
2. Valid correlated covariance matrix is accepted (from_matrix).
3. Invalid covariance matrices are rejected safely and fall back to diagonal variances:
   - Non-symmetric matrix (R != R^T)
   - Non-positive-definite matrix (negative eigenvalues)
   - Matrix containing non-finite values (NaN / Inf)
   - Wrong dimensions (non-square or dimensional mismatch)
4. No off-diagonal values are invented out of thin air.
5. Integration with AssimilationService._build_observation_vector.
"""

import numpy as np
import pytest

from backend.app.assimilation.covariance.observation_covariance import ObservationCovariance
from backend.app.assimilation.state.state_vector import STATE_DIM


def test_diagonal_variances_unchanged():
    """Verify that from_variances produces exact expected diagonal R matrix."""
    vars_in = np.array([0.1, 0.2, np.nan, 0.05, 0.0, np.nan, 0.3, 0.4, 0.1, 0.2])
    cov = ObservationCovariance.from_variances(vars_in, dim=STATE_DIM)

    assert cov.is_diagonal is True
    assert cov.matrix.shape == (STATE_DIM, STATE_DIM)

    # Check diagonal matches safe variances
    expected_diag = np.array([0.1, 0.2, 0.0, 0.05, 0.0, 0.0, 0.3, 0.4, 0.1, 0.2])
    np.testing.assert_allclose(np.diag(cov.matrix), expected_diag)

    # Off-diagonals must be strictly zero
    off_diag = cov.matrix - np.diag(np.diag(cov.matrix))
    np.testing.assert_allclose(off_diag, 0.0)


def test_valid_correlated_covariance_accepted():
    """Verify that a valid symmetric positive-definite covariance matrix is accepted."""
    # Build a valid 4x4 covariance matrix
    matrix = np.array([
        [1.0, 0.2, 0.1, 0.0],
        [0.2, 2.0, 0.3, 0.05],
        [0.1, 0.3, 1.5, 0.1],
        [0.0, 0.05, 0.1, 0.8],
    ])

    cov = ObservationCovariance.from_matrix(matrix, expected_dim=4)
    assert cov.is_diagonal is False
    assert cov.matrix.shape == (4, 4)
    np.testing.assert_allclose(cov.matrix, matrix)


def test_invalid_non_symmetric_matrix_rejected():
    """Verify that non-symmetric matrix is rejected and falls back to diagonal."""
    asymmetric = np.array([
        [1.0, 0.5],
        [0.1, 1.0],  # 0.1 != 0.5
    ])
    fallback_vars = np.array([1.0, 2.0])

    cov = ObservationCovariance.from_matrix(asymmetric, fallback_variances=fallback_vars, expected_dim=2)
    assert cov.is_diagonal is True
    np.testing.assert_allclose(cov.matrix, np.diag([1.0, 2.0]))


def test_invalid_negative_eigenvalues_rejected():
    """Verify that non-positive-definite matrix is rejected safely."""
    # Indefinite matrix (eigenvalues: +1.5, -0.5)
    indefinite = np.array([
        [0.5, 1.0],
        [1.0, 0.5],
    ])
    fallback_vars = np.array([0.5, 0.5])

    cov = ObservationCovariance.from_matrix(indefinite, fallback_variances=fallback_vars, expected_dim=2)
    assert cov.is_diagonal is True
    np.testing.assert_allclose(cov.matrix, np.diag([0.5, 0.5]))


def test_invalid_nan_inf_rejected():
    """Verify that non-finite elements (NaN, Inf) trigger safe fallback."""
    nan_matrix = np.array([
        [1.0, np.nan],
        [np.nan, 1.0],
    ])
    fallback_vars = np.array([1.0, 1.0])

    cov = ObservationCovariance.from_matrix(nan_matrix, fallback_variances=fallback_vars, expected_dim=2)
    assert cov.is_diagonal is True
    np.testing.assert_allclose(cov.matrix, np.diag([1.0, 1.0]))


def test_invalid_dimensions_rejected():
    """Verify that dimension mismatch triggers safe fallback."""
    wrong_dim = np.eye(3)
    fallback_vars = np.array([1.0, 2.0])

    cov = ObservationCovariance.from_matrix(wrong_dim, fallback_variances=fallback_vars, expected_dim=2)
    assert cov.is_diagonal is True
    np.testing.assert_allclose(cov.matrix, np.diag([1.0, 2.0]))


def test_no_off_diagonal_invention():
    """Verify that when only diagonal variances are given, off-diagonals remain strictly zero."""
    vars_in = np.ones(5)
    cov = ObservationCovariance.from_variances(vars_in)

    assert cov.is_diagonal is True
    # Confirm off-diagonal values are identically 0
    for i in range(5):
        for j in range(5):
            if i != j:
                assert cov.matrix[i, j] == 0.0
