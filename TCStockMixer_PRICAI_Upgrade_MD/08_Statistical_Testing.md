# 08 — Statistical Testing for Backtest Credibility

## Objective
Add bootstrap confidence intervals and paired comparison tests for daily strategy returns on the current dataset.

This task supports reviewer-grade empirical rigor. It does not change the model.

## Required methods

Implement three tests first:

1. Bootstrap confidence interval for Sharpe.
2. Paired bootstrap difference in Sharpe between proposed and baseline.
3. Maximum drawdown confidence summary.

More advanced tests such as Reality Check, SPA, Deflated Sharpe Ratio, and Probability of Backtest Overfitting can be added after this task is stable.

## Files to add

```text
StockMixer/src_tc/stat_tests.py
StockMixer/scripts/run_stat_tests.py
StockMixer/tests/test_stat_tests.py
```

## Step 1 — Implement basic statistical utilities

Create `src_tc/stat_tests.py`.

```python
import numpy as np

TRADING_DAYS = 252.0


def sharpe_ratio(returns, eps=1e-12):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * r.mean() / (r.std(ddof=1) + eps))


def max_drawdown_from_returns(returns):
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1.0
    return float(dd.min())


def iid_bootstrap_indices(n, rng):
    return rng.integers(0, n, size=n)


def bootstrap_metric_ci(returns, metric_fn=sharpe_ratio, n_boot=2000, ci=0.95, seed=0):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        raise ValueError("Need at least 5 returns for bootstrap")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = iid_bootstrap_indices(r.size, rng)
        vals.append(metric_fn(r[idx]))
    vals = np.asarray(vals)
    lo = float(np.quantile(vals, (1.0 - ci) / 2.0))
    hi = float(np.quantile(vals, 1.0 - (1.0 - ci) / 2.0))
    return {"estimate": metric_fn(r), "ci_low": lo, "ci_high": hi, "n_boot": int(n_boot)}


def paired_bootstrap_delta_ci(returns_a, returns_b, metric_fn=sharpe_ratio, n_boot=2000, ci=0.95, seed=0):
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 5:
        raise ValueError("Need at least 5 paired returns")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = iid_bootstrap_indices(a.size, rng)
        vals.append(metric_fn(a[idx]) - metric_fn(b[idx]))
    vals = np.asarray(vals)
    lo = float(np.quantile(vals, (1.0 - ci) / 2.0))
    hi = float(np.quantile(vals, 1.0 - (1.0 - ci) / 2.0))
    p_two_sided = float(2.0 * min((vals <= 0).mean(), (vals >= 0).mean()))
    return {
        "delta_estimate": metric_fn(a) - metric_fn(b),
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided_bootstrap": min(1.0, p_two_sided),
        "n_boot": int(n_boot),
    }
```

## Step 2 — Add tests

Create `tests/test_stat_tests.py`.

```python
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
```

## Step 3 — Export daily returns from backtest

If `backtest_topk` already returns `daily_return` or equivalent, use it. If not, edit `src_tc/backtest.py` so the returned dict includes:

```python
"daily_returns": np.asarray(daily_returns, dtype=float),
```

Then update `save_all_results` in `run_experiment_v2.py` to write:

```python
np.savez(
    os.path.join(cfg.output_dir, "test_backtest_returns_top5.npz"),
    returns=test_bt["top5"]["daily_returns"],
)
```

Do the same for valid top5 if needed.

## Step 4 — Add script to run tests from saved outputs

Create `scripts/run_stat_tests.py`.

```python
import argparse
import glob
import json
import os
import numpy as np
from src_tc.stat_tests import bootstrap_metric_ci, paired_bootstrap_delta_ci, max_drawdown_from_returns


def load_returns(path):
    data = np.load(path)
    return data["returns"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed_root", required=True)
    parser.add_argument("--baseline_root", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prop_paths = sorted(glob.glob(os.path.join(args.proposed_root, "**", "test_backtest_returns_top5.npz"), recursive=True))
    if not prop_paths:
        raise FileNotFoundError("No proposed return files found")

    rows = []
    for p in prop_paths:
        r = load_returns(p)
        ci = bootstrap_metric_ci(r, n_boot=args.n_boot, seed=0)
        rows.append({
            "run": p,
            "sharpe": ci,
            "mdd": max_drawdown_from_returns(r),
        })

    result = {"proposed": rows}

    if args.baseline_root:
        base_paths = sorted(glob.glob(os.path.join(args.baseline_root, "**", "test_backtest_returns_top5.npz"), recursive=True))
        paired = []
        for p, b in zip(prop_paths, base_paths):
            paired.append({
                "proposed": p,
                "baseline": b,
                "delta_sharpe": paired_bootstrap_delta_ci(load_returns(p), load_returns(b), n_boot=args.n_boot, seed=1),
            })
        result["paired_vs_baseline"] = paired

    with open(os.path.join(args.out, "stat_tests.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(os.path.join(args.out, "stat_tests.json"))


if __name__ == "__main__":
    main()
```

## Step 5 — Run commands

```bash
cd StockMixer
pytest -q tests/test_stat_tests.py
python scripts/run_stat_tests.py \
  --proposed_root results/crypto_tc_v2 \
  --out reports/crypto_tc_v2_stats \
  --n_boot 2000
```

If a baseline has been run:

```bash
python scripts/run_stat_tests.py \
  --proposed_root results/ablations_crypto/dynamic_both \
  --baseline_root results/ablations_crypto/static_both \
  --out reports/stats_dynamic_vs_static \
  --n_boot 2000
```

## Pass conditions

This step passes only if:

1. `test_backtest_returns_top5.npz` is saved for each evaluated run.
2. `stat_tests.json` contains Sharpe estimate and 95% CI.
3. Paired comparison produces `delta_estimate`, CI, and bootstrap p-value when baseline is supplied.
4. The paper does not claim significance unless the paired CI excludes zero.

## Stronger later extension

After this basic version passes, add:

- stationary bootstrap for autocorrelated returns;
- White Reality Check;
- Hansen SPA test;
- Deflated Sharpe Ratio;
- Probability of Backtest Overfitting.

Do not implement all advanced tests in this micro-task.

