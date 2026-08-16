"""
tests/test_benchmarking_metrics.py
===================================

Unit tests for scientific validation metrics (MAE, RMSE, Bias, R², Ensemble Interval Coverage)
and the BenchmarkEvaluator.
"""

import datetime
import uuid
import numpy as np
import pytest

from backend.app.benchmarking.metrics import (
    compute_all_metrics,
    compute_bias,
    compute_ensemble_coverage,
    compute_mae,
    compute_r2,
    compute_rmse,
)
from backend.app.benchmarking.evaluator import BenchmarkEvaluator
from backend.app.benchmarking.schemas import GroundTruthPoint


def test_metric_calculations_exact():
    """Verify MAE, RMSE, Bias, R² on synthetic vectors."""
    pred = np.array([2.0, 4.0, 6.0], dtype=float)
    gt = np.array([1.0, 5.0, 6.0], dtype=float)

    # Errors: [1.0, -1.0, 0.0]
    # Abs errors: [1.0, 1.0, 0.0] -> MAE = 2/3
    # Sq errors: [1.0, 1.0, 0.0] -> RMSE = sqrt(2/3)
    # Bias: (1 - 1 + 0)/3 = 0.0
    mae = compute_mae(pred, gt)
    rmse = compute_rmse(pred, gt)
    bias = compute_bias(pred, gt)
    r2 = compute_r2(pred, gt)

    assert mae == pytest.approx(2.0 / 3.0)
    assert rmse == pytest.approx(np.sqrt(2.0 / 3.0))
    assert bias == pytest.approx(0.0)
    assert r2 is not None and r2 < 1.0


def test_ensemble_coverage_calculation():
    """Verify ensemble confidence interval coverage rate computation."""
    lowers = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    uppers = np.array([3.0, 4.0, 5.0, 6.0], dtype=float)

    # GT inside: 2.0 (in [1,3]), 4.0 (in [2,4]), 6.0 (in [3,5] - False), 5.0 (in [4,6])
    gt = np.array([2.0, 4.0, 6.0, 5.0], dtype=float)

    coverage = compute_ensemble_coverage(lowers, uppers, gt)
    # 3 out of 4 are inside -> 0.75
    assert coverage == pytest.approx(0.75)


def test_empty_or_mismatched_inputs_return_none():
    """Verify metric safety on empty or invalid inputs."""
    pred = np.array([], dtype=float)
    gt = np.array([], dtype=float)

    assert compute_mae(pred, gt) is None
    assert compute_rmse(pred, gt) is None
    assert compute_bias(pred, gt) is None
    assert compute_r2(pred, gt) is None
    assert compute_ensemble_coverage(pred, gt, pred) is None

    metrics = compute_all_metrics(pred, gt)
    assert metrics.sample_size == 0
    assert metrics.mae is None


def test_benchmark_evaluator_no_ground_truth():
    """Verify evaluator explicitly refrains from fabricating ground truth when empty."""
    evaluator = BenchmarkEvaluator()
    sim_id = uuid.uuid4()

    ol_series = {datetime.date(2025, 5, 1): 2.0}
    ass_series = {datetime.date(2025, 5, 1): 2.5}
    ground_truth = []  # Empty GT

    res = evaluator.evaluate(
        simulation_id=sim_id,
        variable="LAI",
        open_loop_series=ol_series,
        assimilated_series=ass_series,
        ground_truth_points=ground_truth,
    )

    assert res.has_ground_truth is False
    assert res.open_loop.sample_size == 0
    assert res.assimilated.sample_size == 0
    assert "No real ground truth observations supplied" in res.message


def test_benchmark_evaluator_with_real_ground_truth():
    """Verify evaluator compares open-loop vs assimilated when real ground truth is supplied."""
    evaluator = BenchmarkEvaluator()
    sim_id = uuid.uuid4()

    d1 = datetime.date(2025, 5, 1)
    d2 = datetime.date(2025, 5, 10)

    ol_series = {d1: 1.0, d2: 2.0}
    ass_series = {d1: 1.9, d2: 2.9}

    # Real ground truth is close to assimilated: [2.0, 3.0]
    gt_points = [
        GroundTruthPoint(date=d1, variable="LAI", value=2.0),
        GroundTruthPoint(date=d2, variable="LAI", value=3.0),
    ]

    res = evaluator.evaluate(
        simulation_id=sim_id,
        variable="LAI",
        open_loop_series=ol_series,
        assimilated_series=ass_series,
        ground_truth_points=gt_points,
    )

    assert res.has_ground_truth is True
    assert res.open_loop.sample_size == 2
    assert res.assimilated.sample_size == 2

    # Open-loop errors vs GT [2.0, 3.0]: [1.0, 1.0] -> RMSE = 1.0
    # Assimilated errors vs GT [2.0, 3.0]: [0.1, 0.1] -> RMSE = 0.1
    assert res.open_loop.rmse == pytest.approx(1.0)
    assert res.assimilated.rmse == pytest.approx(0.1)
    assert res.rmse_reduction == pytest.approx(0.9)
    assert res.rmse_improvement_pct == pytest.approx(90.0)
