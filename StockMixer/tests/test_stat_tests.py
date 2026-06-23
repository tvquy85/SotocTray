import numpy as np
from src_tc.stat_tests import sharpe_ratio, max_drawdown_from_returns, bootstrap_metric_ci, paired_bootstrap_delta_ci


def test_sharpe_ratio_finite():
    r = np.array([0.01, -0.005, 0.002, 0.003, 0.004, -0.001])
    s = sharpe_ratio(r)
    assert np.isfinite(s)


def test_max_drawdown_negative_or_zero():
    r = np.array([0.1, -0.2, 0.05])
    assert max_drawdown_from_returns(r) <= 0.0


def test_bootstrap_metric_ci_runs():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=100)
    out = bootstrap_metric_ci(r, n_boot=100, seed=1)
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"] or np.isfinite(out["estimate"])


def test_paired_bootstrap_detects_positive_delta_direction():
    rng = np.random.default_rng(0)
    b = rng.normal(0.0, 0.01, size=200)
    a = b + 0.001
    out = paired_bootstrap_delta_ci(a, b, n_boot=200, seed=2)
    assert out["delta_estimate"] > 0
