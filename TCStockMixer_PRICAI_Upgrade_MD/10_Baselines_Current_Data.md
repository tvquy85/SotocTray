# 10 — Minimal Baselines on the Current Dataset

## Objective
Add simple, transparent baselines that can be evaluated on the current saved `gt` and `mask` arrays. These baselines are not the final PRICAI comparison set, but they prevent the paper from relying only on internal ablations.

## Baselines in this micro-task

Implement exactly:

1. `equal_weight_all`: daily equal weight across all valid assets.
2. `random_topk_mean`: average performance of random Top-K over multiple random seeds.
3. `score_topk`: existing model Top-K backtest for comparison.

Do not implement DRL or Markowitz in this task. Those should be separate tasks after the current-data upgrade is stable.

## Files to add

```text
StockMixer/src_tc/simple_baselines.py
StockMixer/scripts/run_simple_baselines.py
StockMixer/tests/test_simple_baselines.py
```

## Step 1 — Implement simple baselines

Create `src_tc/simple_baselines.py`.

```python
import numpy as np
from src_tc.backtest import backtest_topk

TRADING_DAYS = 252.0


def summarize_returns(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / (peak + 1e-12) - 1.0
    return {
        "ann_return": float(equity[-1] ** (TRADING_DAYS / max(len(r), 1)) - 1.0),
        "net_sharpe": float(np.sqrt(TRADING_DAYS) * r.mean() / (r.std(ddof=1) + 1e-12)) if len(r) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(r) else 0.0,
        "avg_turnover": 0.0,
        "n_days": int(len(r)),
    }


def equal_weight_all(gt, mask):
    gt = np.asarray(gt, dtype=float)
    mask = np.asarray(mask, dtype=float)
    valid_count = mask.sum(axis=0)
    daily = np.where(valid_count > 0, (gt * mask).sum(axis=0) / np.maximum(valid_count, 1.0), 0.0)
    return summarize_returns(daily)


def random_topk_mean(gt, mask, topk=5, n_random=100, cost_bps=15.0, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_random):
        random_score = rng.normal(size=np.asarray(gt).shape)
        stats = backtest_topk(random_score, gt, mask, topk=topk, cost_bps=cost_bps)
        results.append({k: v for k, v in stats.items() if isinstance(v, (int, float, np.floating))})
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}
```

## Step 2 — Add script

Create `scripts/run_simple_baselines.py`.

```python
import argparse
import json
import os
import numpy as np
from src_tc.simple_baselines import equal_weight_all, random_topk_mean
from src_tc.backtest import backtest_topk


def scalarize(stats):
    return {k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cost_bps", type=float, default=15.0)
    parser.add_argument("--n_random", type=int, default=100)
    args = parser.parse_args()

    data = np.load(args.pred_npz)
    pred, gt, mask = data["pred"], data["gt"], data["mask"]
    os.makedirs(args.out, exist_ok=True)

    result = {
        "score_topk": scalarize(backtest_topk(pred, gt, mask, topk=args.topk, cost_bps=args.cost_bps)),
        "equal_weight_all": equal_weight_all(gt, mask),
        "random_topk_mean": random_topk_mean(gt, mask, topk=args.topk, n_random=args.n_random, cost_bps=args.cost_bps, seed=0),
    }

    with open(os.path.join(args.out, "simple_baselines.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(os.path.join(args.out, "simple_baselines.json"))


if __name__ == "__main__":
    main()
```

## Step 3 — Add tests

Create `tests/test_simple_baselines.py`.

```python
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
```

## Step 4 — Run commands

```bash
cd StockMixer
pytest -q tests/test_simple_baselines.py
python scripts/run_simple_baselines.py \
  --pred_npz results/crypto_tc_v2_seed0/test_predictions.npz \
  --out reports/simple_baselines_crypto_seed0 \
  --topk 5 \
  --cost_bps 15 \
  --n_random 100
```

## Pass conditions

This step passes only if:

1. `simple_baselines.json` exists.
2. Proposed `score_topk` is compared against both no-learning baselines.
3. `random_topk_mean` uses at least 100 random portfolios for final reporting.
4. No final paper claim says “SOTA” based only on these baselines.

## Required paper wording

Use cautious wording:

```text
We include transparent no-learning baselines to verify that the strategy is not merely exploiting average market drift or random Top-K selection. Comprehensive comparison against DRL and classical portfolio optimization methods is reported separately.
```

If comprehensive baselines are not completed, write:

```text
This version does not claim state-of-the-art performance across all portfolio optimization methods.
```

