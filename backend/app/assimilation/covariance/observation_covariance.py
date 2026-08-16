"""
backend/app/assimilation/covariance/observation_covariance.py
================================================================

Observation Error Covariance Abstraction for EnKF Data Assimilation
-------------------------------------------------------------------

Provides a mathematically validated abstraction for observation error covariance matrices (R).

Scientific Rationale & Independence Guarantees:
1. Diagonal Independence Assumption (Default):
   By default, observation errors across distinct state variables or distinct sensor streams
   are assumed to be independent (uncorrelated). The observation error covariance matrix R
   is diagonal, with non-zero diagonal entries representing individual observation variances:
   r_var = r_std^2.

2. Support for Validated Correlated Covariances:
   When an explicit covariance matrix is provided by a trusted external component (e.g. multi-source
   data fusion, atmospheric radiative transfer retrieval, or multi-band satellite sensor model),
   it is validated strictly before use in EnKF assimilation.

3. Strict Mathematical Validation Rules:
   - Dimensions: Must be a 2D square matrix matching the expected state or observation dimension.
   - Finiteness: All elements must be finite numbers (no NaN or Inf).
   - Symmetry: Matrix must be symmetric (R = R^T within tolerance atol=1e-6).
   - Positive Semi-Definiteness: All eigenvalues must be non-negative (eig >= -1e-10).

4. Zero Off-Diagonal Invention & Safe Fallback:
   Off-diagonal covariance terms are NEVER fabricated or synthesized. If no explicit covariance
   is supplied, or if a supplied matrix fails validation, the system safely defaults to or falls
   back to the diagonal independent observation covariance built from baseline variance estimates.
"""

import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ObservationCovariance:
    """Validated observation error covariance matrix abstraction for EnKF data assimilation."""

    def __init__(self, matrix: np.ndarray, is_diagonal: bool = True) -> None:
        """Initialise ObservationCovariance instance with a validated 2D matrix.

        Args:
            matrix: 2D square numpy array representing the covariance matrix R.
            is_diagonal: Boolean flag indicating if the matrix is strictly diagonal.
        """
        self._matrix = np.array(matrix, dtype=np.float64)
        self._is_diagonal = is_diagonal

    @property
    def matrix(self) -> np.ndarray:
        """Return the underlying covariance matrix R."""
        return self._matrix

    @property
    def is_diagonal(self) -> bool:
        """Return True if the covariance matrix is diagonal (uncorrelated observations)."""
        return self._is_diagonal

    @classmethod
    def from_variances(
        cls,
        variances: np.ndarray,
        dim: Optional[int] = None,
    ) -> "ObservationCovariance":
        """Build standard diagonal covariance matrix from variance vector.

        Missing / NaN variances are safely set to 0.0.
        Preserves exact baseline diagonal R matrix generation.

        Args:
            variances: 1D numpy array of observation variances r_var = r_std^2.
            dim: Optional target dimension. If provided, variances length must match dim.

        Returns:
            ObservationCovariance with diagonal R matrix.
        """
        var_arr = np.asarray(variances, dtype=np.float64)
        if var_arr.ndim != 1:
            raise ValueError(f"Variances must be a 1D array, got shape {var_arr.shape}.")

        n = dim if dim is not None else len(var_arr)
        if len(var_arr) != n:
            raise ValueError(f"Variances length ({len(var_arr)}) does not match expected dimension ({n}).")

        safe_vars = np.where(np.isnan(var_arr), 0.0, var_arr)
        safe_vars = np.maximum(0.0, safe_vars)
        R = np.diag(safe_vars)
        return cls(R, is_diagonal=True)

    @classmethod
    def validate_matrix(
        cls,
        matrix: np.ndarray,
        expected_dim: Optional[int] = None,
        atol: float = 1e-6,
    ) -> Tuple[bool, Optional[str]]:
        """Validate a proposed covariance matrix for finiteness, square dimensions, symmetry, and positive semi-definiteness.

        Args:
            matrix: Proposed covariance matrix to validate.
            expected_dim: Optional expected dimension (number of rows/cols).
            atol: Absolute tolerance for numerical symmetry check.

        Returns:
            Tuple of (is_valid: bool, reason: Optional[str]).
        """
        if not isinstance(matrix, np.ndarray):
            return False, "Input matrix must be a numpy ndarray."

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return False, f"Covariance matrix must be 2D square, got shape {matrix.shape}."

        if expected_dim is not None and matrix.shape[0] != expected_dim:
            return False, f"Covariance matrix dimension {matrix.shape[0]} does not match expected {expected_dim}."

        if not np.all(np.isfinite(matrix)):
            return False, "Covariance matrix contains non-finite values (NaN or Inf)."

        if not np.allclose(matrix, matrix.T, atol=atol):
            return False, "Covariance matrix is not symmetric (R != R^T)."

        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals < -1e-10):
            min_eig = float(np.min(eigvals))
            return False, f"Covariance matrix is not positive semi-definite (minimum eigenvalue = {min_eig:.6e} < 0)."

        return True, None

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        fallback_variances: Optional[np.ndarray] = None,
        expected_dim: Optional[int] = None,
        atol: float = 1e-6,
    ) -> "ObservationCovariance":
        """Build ObservationCovariance from an explicit full matrix, with strict validation and safe fallback.

        Args:
            matrix: Explicit covariance matrix supplied by a trusted component.
            fallback_variances: Optional 1D variance array to use if validation fails.
            expected_dim: Optional expected matrix dimension.
            atol: Tolerance for symmetry check.

        Returns:
            ObservationCovariance instance. If matrix is valid, uses matrix (symmetrized).
            If invalid, falls back safely to diagonal covariance.
        """
        arr = np.asarray(matrix, dtype=np.float64)
        is_valid, reason = cls.validate_matrix(arr, expected_dim=expected_dim, atol=atol)

        if is_valid:
            # Enforce exact mathematical symmetry
            sym_matrix = 0.5 * (arr + arr.T)
            off_diag_mask = ~np.eye(sym_matrix.shape[0], dtype=bool)
            is_diag = bool(np.allclose(sym_matrix[off_diag_mask], 0.0, atol=atol))
            return cls(sym_matrix, is_diagonal=is_diag)

        logger.warning(
            "ObservationCovariance: Explicit matrix rejected (%s). Falling back to diagonal covariance.",
            reason,
        )

        if fallback_variances is not None:
            return cls.from_variances(fallback_variances, dim=expected_dim)

        diag_vars = np.diag(arr) if arr.ndim == 2 and arr.shape[0] == arr.shape[1] else np.zeros(expected_dim or 10)
        return cls.from_variances(diag_vars, dim=expected_dim)
