"""
backend/app/benchmarking/evaluator.py
======================================

Benchmark evaluator that compares open-loop WOFOST predictions vs.
assimilated WOFOST predictions against REAL external ground truth observations.

Enforces zero-fabrication of ground truth: evaluation occurs ONLY when
real observations are supplied.
"""

import datetime
import uuid
from typing import Dict, List, Optional
import numpy as np

from backend.app.benchmarking.metrics import compute_all_metrics
from backend.app.benchmarking.schemas import (
    BenchmarkComparisonResult,
    BenchmarkEvaluateRequest,
    GroundTruthPoint,
    MetricSet,
)


class BenchmarkEvaluator:
    """Evaluates prediction accuracy of open-loop vs. assimilated crop simulations against real ground truth."""

    def evaluate(
        self,
        simulation_id: uuid.UUID,
        variable: str,
        open_loop_series: Dict[datetime.date, float],
        assimilated_series: Dict[datetime.date, float],
        ground_truth_points: List[GroundTruthPoint],
        ensemble_bounds: Optional[Dict[datetime.date, tuple[float, float]]] = None,
    ) -> BenchmarkComparisonResult:
        """Evaluate open-loop vs assimilated predictions against supplied external ground truth.

        Args:
            simulation_id: UUID of the baseline simulation.
            variable: Target variable evaluated (e.g., "LAI", "SM", "TWSO").
            open_loop_series: Dictionary mapping date -> open-loop value.
            assimilated_series: Dictionary mapping date -> assimilated value.
            ground_truth_points: List of real external GroundTruthPoint objects.
            ensemble_bounds: Optional dictionary mapping date -> (lower_bound, upper_bound).

        Returns:
            BenchmarkComparisonResult comparing open-loop and assimilated metrics.
        """
        # Filter ground truth points matching the requested variable
        var_upper = variable.upper()
        matching_gt = [
            pt for pt in ground_truth_points
            if pt.variable.upper() == var_upper and pt.value is not None
        ]

        if not matching_gt:
            return BenchmarkComparisonResult(
                simulation_id=simulation_id,
                variable=variable,
                has_ground_truth=False,
                message="No real ground truth observations supplied for target variable. Evaluation skipped without fabrication.",
                open_loop=MetricSet(sample_size=0),
                assimilated=MetricSet(sample_size=0),
                rmse_reduction=None,
                rmse_improvement_pct=None,
            )

        # Align series by matching dates
        dates: List[datetime.date] = []
        ol_vals: List[float] = []
        ass_vals: List[float] = []
        gt_vals: List[float] = []
        ens_lowers: List[float] = []
        ens_uppers: List[float] = []
        has_ensemble_bounds = False

        for pt in matching_gt:
            d = pt.date
            if d in open_loop_series and d in assimilated_series:
                ol_val = open_loop_series[d]
                ass_val = assimilated_series[d]
                if ol_val is not None and ass_val is not None:
                    dates.append(d)
                    gt_vals.append(pt.value)
                    ol_vals.append(ol_val)
                    ass_vals.append(ass_val)

                    if ensemble_bounds and d in ensemble_bounds:
                        lower, upper = ensemble_bounds[d]
                        ens_lowers.append(lower)
                        ens_uppers.append(upper)
                        has_ensemble_bounds = True

        if len(gt_vals) == 0:
            return BenchmarkComparisonResult(
                simulation_id=simulation_id,
                variable=variable,
                has_ground_truth=False,
                message="Ground truth observations were supplied, but no dates overlapped with simulation outputs.",
                open_loop=MetricSet(sample_size=0),
                assimilated=MetricSet(sample_size=0),
                rmse_reduction=None,
                rmse_improvement_pct=None,
            )

        gt_arr = np.array(gt_vals, dtype=float)
        ol_arr = np.array(ol_vals, dtype=float)
        ass_arr = np.array(ass_vals, dtype=float)

        ol_metrics = compute_all_metrics(ol_arr, gt_arr)

        if has_ensemble_bounds and len(ens_lowers) == len(ass_arr):
            lower_arr = np.array(ens_lowers, dtype=float)
            upper_arr = np.array(ens_uppers, dtype=float)
            ass_metrics = compute_all_metrics(ass_arr, gt_arr, ensemble_lower=lower_arr, ensemble_upper=upper_arr)
        else:
            ass_metrics = compute_all_metrics(ass_arr, gt_arr)

        rmse_red = None
        rmse_imp_pct = None
        if ol_metrics.rmse is not None and ass_metrics.rmse is not None:
            rmse_red = ol_metrics.rmse - ass_metrics.rmse
            if ol_metrics.rmse > 0:
                rmse_imp_pct = (rmse_red / ol_metrics.rmse) * 100.0

        return BenchmarkComparisonResult(
            simulation_id=simulation_id,
            variable=variable,
            has_ground_truth=True,
            message=f"Successfully evaluated across {len(gt_vals)} aligned ground truth observations.",
            open_loop=ol_metrics,
            assimilated=ass_metrics,
            rmse_reduction=rmse_red,
            rmse_improvement_pct=rmse_imp_pct,
        )
