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
        "ann_return": float(equity[-1] ** (TRADING_DAYS / max(len(r), 1)) - 1.0) if len(r) > 0 else 0.0,
        "net_sharpe": float(np.sqrt(TRADING_DAYS) * r.mean() / (r.std(ddof=1) + 1e-12)) if len(r) > 1 else 0.0,
        "max_drawdown": float(drawdown.min()) if len(r) else 0.0,
        "avg_turnover": 0.0,
        "n_days": int(len(r)),
    }


def equal_weight_all(gt, mask):
    gt = np.asarray(gt, dtype=float)
    mask = np.asarray(mask, dtype=float)
    valid_count = mask.sum(axis=1)
    daily = np.where(valid_count > 0, (gt * mask).sum(axis=1) / np.maximum(valid_count, 1.0), 0.0)
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
