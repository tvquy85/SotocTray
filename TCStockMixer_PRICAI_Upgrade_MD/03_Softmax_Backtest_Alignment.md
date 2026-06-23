# 03 — Align Differentiable Training Objective with Backtest Evaluation

## Objective

Training tối ưu softmax differentiable portfolio, nhưng evaluation hiện dùng Top-K equal-weight. Thêm `backtest_softmax` làm primary evaluation và early-stopping metric. Top-K vẫn giữ làm secondary robustness check.

## Allowed edit scope

- Modify: `StockMixer/src_tc/backtest.py`
- Modify: `StockMixer/src_tc/trainer_v2.py`
- Modify: `StockMixer/src_tc/config.py`
- Modify: `StockMixer/src_tc/run_experiment_v2.py`
- Add: `StockMixer/tests/test_softmax_backtest.py`

## Step 1 — Add softmax backtest

Append to `StockMixer/src_tc/backtest.py`:

```python
def softmax_weights(scores, valid, tau=0.1):
    w = np.zeros_like(scores, dtype=np.float64)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return w
    z = scores[valid_idx].astype(np.float64) / max(float(tau), 1e-12)
    z = z - np.max(z)
    ez = np.exp(z)
    w[valid_idx] = ez / (np.sum(ez) + 1e-12)
    return w


def backtest_softmax(prediction, ground_truth, mask, tau=0.1, cost_bps=10.0):
    assert prediction.shape == ground_truth.shape == mask.shape
    num_assets, num_dates = prediction.shape
    cost_rate = cost_bps / 10000.0
    prev_w = np.zeros(num_assets, dtype=np.float64)
    gross_returns, net_returns, turnovers, holdings = [], [], [], []

    for t in range(num_dates):
        valid = mask[:, t] > 0.5
        w = softmax_weights(prediction[:, t], valid, tau=tau)
        gross = float(np.sum(w * ground_truth[:, t]))
        turnover = float(np.sum(np.abs(w - prev_w)))
        net = gross - cost_rate * turnover
        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)
        holdings.append(int(np.sum(w > 1e-8)))
        prev_w = w

    return summarize_returns(
        np.array(gross_returns, dtype=np.float64),
        np.array(net_returns, dtype=np.float64),
        np.array(turnovers, dtype=np.float64),
        holdings,
    )
```

## Step 2 — Config defaults

In `ExperimentConfig`, add:

```python
selection_backtest: str = "softmax"
selection_metric: str = "net_sharpe"
```

If dataclass construction fails for missing fields, add to YAML:

```yaml
selection_backtest: softmax
selection_metric: net_sharpe
```

## Step 3 — Update `evaluate_model`

In `trainer_v2.py`, import:

```python
from src_tc.backtest import backtest_topk, backtest_softmax
```

Replace current return block with:

```python
bt_softmax = backtest_softmax(pred, gt, mask, tau=cfg.tau, cost_bps=cfg.eval_cost_bps)
bt5 = backtest_topk(pred, gt, mask, topk=5, cost_bps=cfg.eval_cost_bps)
bt10 = backtest_topk(pred, gt, mask, topk=10, cost_bps=cfg.eval_cost_bps)
return pred, gt, mask, pred_metrics, {'softmax': bt_softmax, 'top5': bt5, 'top10': bt10}
```

## Step 4 — Update early stopping

In `run_experiment_v2.py`, replace:

```python
valid_score = valid_bt['top5']['net_sharpe']
```

with:

```python
selection_backtest = getattr(cfg, 'selection_backtest', 'softmax')
selection_metric = getattr(cfg, 'selection_metric', 'net_sharpe')
valid_score = valid_bt[selection_backtest][selection_metric]
```

Add to training log row:

```python
'valid_softmax_net_sharpe': valid_bt['softmax']['net_sharpe'],
'valid_softmax_avg_turnover': valid_bt['softmax']['avg_turnover'],
'valid_softmax_mdd': valid_bt['softmax']['max_drawdown'],
```

## Step 5 — Tests

Create `StockMixer/tests/test_softmax_backtest.py`:

```python
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
```

Run:

```bash
cd StockMixer
python -m pytest tests/test_softmax_backtest.py -q
python src_tc/run_experiment_v2.py --config configs/crypto_tc_v2.yaml --seed 12345678
python - <<'PYTEST'
import json
from pathlib import Path
m = json.loads(Path('results/crypto_tc_v2_seed12345678/metrics.json').read_text())
assert 'softmax' in m['valid_backtest']
assert 'softmax' in m['test_backtest']
print('SOFTMAX_BACKTEST_PASS')
PYTEST
```
