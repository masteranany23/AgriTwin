"""
backend/app/benchmarking/enkf_diagnostics.py
=============================================

Extracts and summarizes compact EnKF diagnostics:
- Innovation (y - H*x)
- Ensemble spread (prior & posterior standard deviations)
- Valid / rejected observation counts
- State update magnitude (|x_post - x_prior|)
"""

import datetime
import uuid
from typing import Dict, List, Optional
import numpy as np

from backend.app.assimilation.models.assimilation_state import AssimilationState
from backend.app.assimilation.state.state_vector import STATE_VARIABLES
from backend.app.benchmarking.schemas import CompactCycleDiagnostics, RunDiagnosticsSummary


class EnKFDiagnosticsExtractor:
    """Extractor for compact EnKF diagnostics from matrix updates or stored AssimilationState DB records."""

    @staticmethod
    def extract_from_matrices(
        cycle_date: datetime.date,
        X_f: np.ndarray,
        X_a: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        raw_obs_count: int,
        qc_obs_count: int,
        state_vars: Optional[List[str]] = None,
    ) -> CompactCycleDiagnostics:
        """Extract compact EnKF diagnostics directly from full ensemble matrices during an assimilation cycle.

        Args:
            cycle_date: Date of the assimilation cycle.
            X_f: Forecast ensemble matrix of shape (n, N).
            X_a: Analysis ensemble matrix of shape (n, N).
            y: Observation vector of shape (n,).
            d: Innovation vector of shape (n,).
            raw_obs_count: Number of retrieved observations before QC.
            qc_obs_count: Number of observations passing QC.
            state_vars: List of variable names corresponding to state dimension n.

        Returns:
            CompactCycleDiagnostics instance.
        """
        if state_vars is None:
            state_vars = list(STATE_VARIABLES)

        n = X_f.shape[0]

        # Prior and posterior means
        x_mean_f = np.nanmean(X_f, axis=1)
        x_mean_a = np.nanmean(X_a, axis=1)

        # Prior and posterior spreads (standard deviations across members)
        std_f = np.nanstd(X_f, axis=1, ddof=1) if X_f.shape[1] > 1 else np.zeros(n)
        std_a = np.nanstd(X_a, axis=1, ddof=1) if X_a.shape[1] > 1 else np.zeros(n)

        # State update magnitude
        update_mag = np.abs(x_mean_a - x_mean_f)

        innov_dict: Dict[str, Optional[float]] = {}
        prior_spread_dict: Dict[str, Optional[float]] = {}
        post_spread_dict: Dict[str, Optional[float]] = {}
        update_mag_dict: Dict[str, Optional[float]] = {}
        variables_updated: List[str] = []

        for i in range(min(n, len(state_vars))):
            v = state_vars[i].lower()
            d_val = float(d[i]) if not np.isnan(d[i]) else None
            innov_dict[v] = d_val
            prior_spread_dict[v] = float(std_f[i]) if not np.isnan(std_f[i]) else 0.0
            post_spread_dict[v] = float(std_a[i]) if not np.isnan(std_a[i]) else 0.0
            update_mag_dict[v] = float(update_mag[i]) if not np.isnan(update_mag[i]) else 0.0

            if d_val is not None:
                variables_updated.append(v.upper())

        rejected_count = max(0, raw_obs_count - qc_obs_count)

        return CompactCycleDiagnostics(
            cycle_date=cycle_date,
            variables_updated=variables_updated,
            valid_obs_count=qc_obs_count,
            rejected_obs_count=rejected_count,
            innovation=innov_dict,
            ensemble_spread_prior=prior_spread_dict,
            posterior_spread=post_spread_dict,
            state_update_magnitude=update_mag_dict,
        )

    @staticmethod
    def extract_from_db_state(state: AssimilationState, raw_obs_count: Optional[int] = None) -> CompactCycleDiagnostics:
        """Extract compact EnKF diagnostics from a stored DB AssimilationState record."""
        cycle_date = state.assimilation_time.date()
        prior = state.forecast_state_vector or {}
        post = state.updated_state_vector or {}
        obs_vec = state.observation_vector or {}
        innov_vec = state.innovation_vector or {}
        cov_matrix = state.ensemble_covariance or {}

        innov_dict: Dict[str, Optional[float]] = {}
        update_mag_dict: Dict[str, Optional[float]] = {}
        prior_spread_dict: Dict[str, Optional[float]] = {}
        post_spread_dict: Dict[str, Optional[float]] = {}
        variables_updated: List[str] = []

        all_keys = set(prior.keys()) | set(post.keys()) | set(innov_vec.keys())

        for k in all_keys:
            k_lower = k.lower()
            innov_val = innov_vec.get(k) or innov_vec.get(k_lower)
            innov_dict[k_lower] = float(innov_val) if innov_val is not None else None

            pr_val = prior.get(k) or prior.get(k_lower)
            po_val = post.get(k) or post.get(k_lower)

            if pr_val is not None and po_val is not None:
                update_mag_dict[k_lower] = float(abs(po_val - pr_val))
            else:
                update_mag_dict[k_lower] = 0.0

            # Estimate spread from stored covariance if present
            var_cov = cov_matrix.get(k) or cov_matrix.get(k_lower)
            if isinstance(var_cov, (int, float)) and var_cov >= 0:
                post_spread_dict[k_lower] = float(np.sqrt(var_cov))
                prior_spread_dict[k_lower] = float(np.sqrt(var_cov))
            else:
                post_spread_dict[k_lower] = 0.0
                prior_spread_dict[k_lower] = 0.0

            if innov_val is not None:
                variables_updated.append(k.upper())

        valid_count = state.observation_count
        raw_count = raw_obs_count if raw_obs_count is not None else valid_count
        rejected_count = max(0, raw_count - valid_count)

        return CompactCycleDiagnostics(
            cycle_date=cycle_date,
            variables_updated=variables_updated,
            valid_obs_count=valid_count,
            rejected_obs_count=rejected_count,
            innovation=innov_dict,
            ensemble_spread_prior=prior_spread_dict,
            posterior_spread=post_spread_dict,
            state_update_magnitude=update_mag_dict,
        )

    @classmethod
    def summarize_run(
        cls,
        simulation_id: uuid.UUID,
        assimilation_run_id: Optional[uuid.UUID],
        cycles: List[CompactCycleDiagnostics],
    ) -> RunDiagnosticsSummary:
        """Summarize compact EnKF diagnostics across an entire assimilation run."""
        total_cycles = len(cycles)
        executed_cycles = sum(1 for c in cycles if c.valid_obs_count > 0 or any(v is not None for v in c.innovation.values()))
        total_valid = sum(c.valid_obs_count for c in cycles)
        total_rejected = sum(c.rejected_obs_count for c in cycles)

        # Aggregate averages per state variable
        var_keys = set()
        for c in cycles:
            var_keys.update(c.innovation.keys())

        avg_update_mag: Dict[str, Optional[float]] = {}
        avg_innov: Dict[str, Optional[float]] = {}
        mean_prior_spread: Dict[str, Optional[float]] = {}
        mean_post_spread: Dict[str, Optional[float]] = {}

        for k in var_keys:
            mags = [c.state_update_magnitude[k] for c in cycles if c.state_update_magnitude.get(k) is not None]
            innovs = [abs(c.innovation[k]) for c in cycles if c.innovation.get(k) is not None]
            pr_spreads = [c.ensemble_spread_prior[k] for c in cycles if c.ensemble_spread_prior.get(k) is not None]
            po_spreads = [c.posterior_spread[k] for c in cycles if c.posterior_spread.get(k) is not None]

            avg_update_mag[k] = float(np.mean(mags)) if mags else None
            avg_innov[k] = float(np.mean(innovs)) if innovs else None
            mean_prior_spread[k] = float(np.mean(pr_spreads)) if pr_spreads else None
            mean_post_spread[k] = float(np.mean(po_spreads)) if po_spreads else None

        return RunDiagnosticsSummary(
            simulation_id=simulation_id,
            assimilation_run_id=assimilation_run_id,
            total_cycles=total_cycles,
            executed_cycles=executed_cycles,
            total_valid_obs=total_valid,
            total_rejected_obs=total_rejected,
            avg_state_update_magnitude=avg_update_mag,
            avg_innovation=avg_innov,
            mean_prior_spread=mean_prior_spread,
            mean_posterior_spread=mean_post_spread,
            cycles=cycles,
        )
