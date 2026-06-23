import numpy as np
from src_tc.simple_baselines import equal_weight_all, random_topk_mean


def test_equal_weight_all_runs():
    gt = np.array([[0.01, 0.02], [0.03, -0.01]])
    mask = np.ones_like(gt)
    stats = equal_weight_all(gt, mask)
    assert "net_sharpe" in stats
    assert stats["n_days"] == 2


def test_random_topk_mean_runs():
    gt = np.random.default_rng(0).normal(0.0, 0.01, size=(5, 20))
    mask = np.ones_like(gt)
    stats = random_topk_mean(gt, mask, topk=2, n_random=5, cost_bps=10.0, seed=1)
    assert "net_sharpe" in stats
