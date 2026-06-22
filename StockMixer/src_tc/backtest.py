import numpy as np


def topk_equal_weights(scores, valid, topk):
    w = np.zeros_like(scores, dtype=np.float64)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return w
    k = min(topk, len(valid_idx))
    local_scores = scores[valid_idx]
    chosen_local = np.argsort(local_scores)[-k:]
    chosen = valid_idx[chosen_local]
    w[chosen] = 1.0 / k
    return w


def summarize_returns(gross_returns, net_returns, turnovers, holdings, annualization=252):
    def ann_sharpe(x):
        return float(np.sqrt(annualization) * np.mean(x) / (np.std(x) + 1e-12))

    def sortino(x):
        downside = x[x < 0]
        denom = np.std(downside) if len(downside) > 1 else np.std(x)
        return float(np.sqrt(annualization) * np.mean(x) / (denom + 1e-12))

    def max_drawdown(returns):
        equity = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(equity)
        dd = equity / (peak + 1e-12) - 1.0
        return float(np.min(dd)), equity

    mdd, equity = max_drawdown(net_returns)
    ann_ret = float(np.prod(1.0 + net_returns) ** (annualization / max(len(net_returns), 1)) - 1.0)
    calmar = float(ann_ret / (abs(mdd) + 1e-12))

    return {
        'gross_mean_daily': float(np.mean(gross_returns)),
        'net_mean_daily': float(np.mean(net_returns)),
        'gross_sharpe': ann_sharpe(gross_returns),
        'net_sharpe': ann_sharpe(net_returns),
        'net_sortino': sortino(net_returns),
        'net_ann_return': ann_ret,
        'max_drawdown': mdd,
        'calmar': calmar,
        'avg_turnover': float(np.mean(turnovers)),
        'median_turnover': float(np.median(turnovers)),
        'avg_holdings': float(np.mean(holdings)),
        'final_equity': float(equity[-1]) if len(equity) else 1.0,
        'net_returns': net_returns,
        'gross_returns': gross_returns,
        'turnovers': turnovers,
        'equity': equity,
    }


def backtest_topk(prediction, ground_truth, mask, topk=5, cost_bps=10.0):
    assert prediction.shape == ground_truth.shape == mask.shape
    N, T = prediction.shape
    cost_rate = cost_bps / 10000.0
    prev_w = np.zeros(N, dtype=np.float64)

    gross_returns = []
    net_returns = []
    turnovers = []
    holdings = []

    for t in range(T):
        valid = mask[:, t] > 0.5
        w = topk_equal_weights(prediction[:, t], valid, topk)
        gross = float(np.sum(w * ground_truth[:, t]))
        turnover = float(np.sum(np.abs(w - prev_w)))
        net = gross - cost_rate * turnover

        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(turnover)
        holdings.append(int(np.sum(w > 0)))
        prev_w = w

    gross_returns = np.array(gross_returns, dtype=np.float64)
    net_returns = np.array(net_returns, dtype=np.float64)
    turnovers = np.array(turnovers, dtype=np.float64)

    return summarize_returns(gross_returns, net_returns, turnovers, holdings)
