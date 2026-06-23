import numpy as np
import torch
from src_tc.backtest import softmax_weights, backtest_softmax
from src_tc.losses_v2 import masked_softmax


def test_softmax_weights_sum_to_one_and_ignore_invalid_assets():
    scores = np.array([1.0, 2.0, 100.0, 0.5])
    valid = np.array([True, True, False, True])
    w = softmax_weights(scores, valid, tau=0.1)
    assert np.isclose(w.sum(), 1.0)
    assert w[2] == 0.0
    assert np.all(w >= 0.0)


def test_numpy_softmax_matches_torch_masked_softmax():
    scores = np.array([0.2, 0.7, -0.1, 0.0], dtype=np.float64)
    valid = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
    w_np = softmax_weights(scores, valid > 0.5, tau=0.25)
    w_t = masked_softmax(
        torch.tensor(scores, dtype=torch.float32).view(-1, 1),
        torch.tensor(valid, dtype=torch.float32).view(-1, 1),
        tau=0.25,
    ).detach().numpy()
    assert np.allclose(w_np, w_t, atol=1e-6)


def test_backtest_softmax_cost_reduces_mean_return():
    pred = np.array([[1.0, 0.0], [0.0, 1.0]])
    gt = np.array([[0.01, -0.01], [-0.01, 0.01]])
    mask = np.ones_like(pred)
    a = backtest_softmax(pred, gt, mask, tau=0.01, cost_bps=0.0)
    b = backtest_softmax(pred, gt, mask, tau=0.01, cost_bps=100.0)
    assert b['net_mean_daily'] <= a['net_mean_daily']
