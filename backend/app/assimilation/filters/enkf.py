"""
backend/app/assimilation/filters/enkf.py
========================================

Core mathematical implementation of the Ensemble Kalman Filter (EnKF).
"""

from typing import Optional

import numpy as np

from backend.app.assimilation.operators.observation_operator import (
    BaseObservationOperator,
    DirectObservationOperator,
)


def enkf_update(
    X_f: np.ndarray,
    y: np.ndarray,
    R: np.ndarray,
    H_operator: Optional[BaseObservationOperator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute the stochastic EnKF update step.
    
    Handles partial observations by dynamically constructing or utilizing an explicit
    observation operator H based on non-NaN values in observation vector `y` and
    forecast ensemble `X_f`.
    
    Hardened numerical implementation featuring:
    - Explicit observation operator abstraction y = h(x) + noise.
    - Stable linear solve replacing explicit matrix inversion (`np.linalg.inv`).
    - Exact covariance matrix symmetry enforcement & adaptive diagonal regularization.
    - Clean NaN protection preventing uninitialized state variables from corrupting
      valid ensemble states or output matrices.
    - Strict input shape/dimension validation.
    
    Args:
        X_f:        Forecast ensemble matrix of shape (n, N).
                    n = state dimension, N = ensemble size.
                    NaN values indicate uninitialized state variables.
        y:          Observation vector of shape (n,).
                    Missing observations must be set to np.nan.
        R:          Observation error covariance matrix of shape (n, n).
        H_operator: Optional explicit observation operator h(x). If None, a
                    DirectObservationOperator is dynamically constructed for valid obs.
              
    Returns:
        X_a: Updated analysis ensemble matrix of shape (n, N).
        d:   Innovation vector (y - H*x_mean) of shape (n,). Unobserved variables are NaN.
        K:   Kalman Gain matrix mapped to full state space shape (n, n).
    """
    # 0. Validate input shapes and dimensions
    if X_f.ndim != 2:
        raise ValueError(f"X_f must be a 2D matrix of shape (n, N), got {X_f.ndim}D.")
    n, N = X_f.shape
    if N <= 1:
        raise ValueError(f"Ensemble size N must be > 1 to compute covariance, got N={N}.")
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"y must be a 1D vector of length n={n}, got shape {y.shape}.")
    if R.ndim != 2 or R.shape != (n, n):
        raise ValueError(f"R must be a 2D covariance matrix of shape ({n}, {n}), got {R.shape}.")

    # 1. Identify valid observation indices
    # Observation y must be finite, and corresponding row in X_f must be fully finite across all members N
    valid_y = np.isfinite(y)
    valid_X = np.all(np.isfinite(X_f), axis=1)
    
    obs_idx = np.where(valid_y & valid_X)[0]
    
    if len(obs_idx) == 0:
        # No valid observations to assimilate; return forecast as analysis
        return X_f.copy(), np.full(n, np.nan), np.zeros((n, n))
        
    m = len(obs_idx)
    
    # Construct explicit observation operator if not provided
    if H_operator is None:
        H_op = DirectObservationOperator(obs_idx, state_dim=n)
    else:
        H_op = H_operator

    # 2. Extract reduced observation vector and covariance with guaranteed symmetry
    y_red = y[obs_idx]
    R_red = R[np.ix_(obs_idx, obs_idx)]
    R_red = 0.5 * (R_red + R_red.T)
    
    # 3. Identify finite/valid state variable rows to prevent NaN contamination
    valid_state_mask = np.all(np.isfinite(X_f), axis=1)
    valid_state_idx = np.where(valid_state_mask)[0]
    
    # Compute forecast mean and anomalies ONLY on valid state variables
    x_mean_valid = np.mean(X_f[valid_state_idx, :], axis=1)
    A_valid = X_f[valid_state_idx, :] - x_mean_valid[:, np.newaxis]
    
    # Evaluate observation operator projection h(X) on valid forecast states
    x_mean_full = np.zeros(n)
    x_mean_full[valid_state_idx] = x_mean_valid
    Hx_mean = H_op.apply(x_mean_full)
    if Hx_mean.shape[0] != m:
        # Fallback if H_op is defined on full state rather than reduced obs_idx
        Hx_mean = H_op.apply(x_mean_full)[obs_idx]
    
    obs_in_valid_pos = [np.where(valid_state_idx == idx)[0][0] for idx in obs_idx]
    HA = A_valid[obs_in_valid_pos, :]        # shape (m, N)
    
    # 4. Perturb observations (Stochastic EnKF requirement)
    try:
        V = np.random.multivariate_normal(np.zeros(m), R_red, size=N).T
    except ValueError:
        # Fallback if R_red has non-positive-definite numerical issues
        eigvals, eigvecs = np.linalg.eigh(R_red)
        eigvals = np.maximum(eigvals, 0.0)
        R_red_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        V = np.random.multivariate_normal(np.zeros(m), R_red_fixed, size=N).T
        
    Y = y_red[:, np.newaxis] + V
    
    # 5. Compute Innovation matrix
    HX_f = X_f[obs_idx, :]
    D = Y - HX_f
    
    # Compute mean innovation for tracking
    d_red = y_red - Hx_mean
    d_full = np.full(n, np.nan)
    d_full[obs_idx] = d_red
    
    # 6. Compute covariances with guaranteed symmetry & adaptive regularization
    PHt_valid = (1.0 / (N - 1)) * (A_valid @ HA.T)  # shape (n_valid, m)
    S = (1.0 / (N - 1)) * (HA @ HA.T) + R_red      # shape (m, m)
    S = 0.5 * (S + S.T)                            # Enforce numerical symmetry
    
    # Add adaptive diagonal regularization to ensure S is well-conditioned
    s_trace = np.trace(S)
    reg = max(1e-12, 1e-12 * (s_trace / m)) if s_trace > 0 else 1e-12
    S += reg * np.eye(m)
    
    # 7. Numerically stable linear solve for Kalman Gain (replaces np.linalg.inv(S))
    # Solve S * K_red_T = PHt_valid.T  =>  K_red = K_red_T.T
    try:
        K_red_T = np.linalg.solve(S, PHt_valid.T)
    except np.linalg.LinAlgError:
        K_red_T, _, _, _ = np.linalg.lstsq(S, PHt_valid.T, rcond=1e-10)
        
    K_red = K_red_T.T  # shape (n_valid, m)
    
    # Map Kalman Gain to full state space (n, n)
    K_full = np.zeros((n, n))
    K_full[np.ix_(valid_state_idx, obs_idx)] = K_red
    
    # 8. Update ensemble matrix
    X_a = X_f.copy()
    X_a[valid_state_idx, :] = X_f[valid_state_idx, :] + K_red @ D
    
    return X_a, d_full, K_full
