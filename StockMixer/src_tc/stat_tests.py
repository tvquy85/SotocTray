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
