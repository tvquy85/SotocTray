import numpy as np


def market_state_from_closes_v1(closes, close_col=-1, window=16, train_end_idx=756):
    """5-Dim Context Extraction"""
    close_px = closes[:, :, close_col]  # (N, T)
    rets = np.log(close_px[:, 1:] / close_px[:, :-1])  # (N, T-1)

    N, Tm1 = rets.shape
    W = window
    num_win = Tm1 - W + 1
    metrics = np.zeros((num_win, 5), dtype=np.float32)

    x = np.arange(W, dtype=np.float32)  # 0 ... 15 for slope calc
    var_x = x.var()

    for k in range(num_win):
        win = rets[:, k:k + W]  # (N, W)
        idx_series = win.mean(axis=0)  # equally-weighted index, (W,)

        # 1) mean return
        mean_ret = win.mean()

        # 2) slope (market momentum)
        cov_x = ((x - x.mean()) * (idx_series - idx_series.mean())).mean()
        slope = cov_x / var_x

        # 3) realised vol
        real_vol = idx_series.std()

        # 4) dispersion
        disp = win.std(axis=0).mean()

        # 5) PCA (Systemic Risk)
        win_centered = win - win.mean(axis=1, keepdims=True)
        gram = (win_centered.T @ win_centered) / W
        eigvals = np.linalg.eigvalsh(gram)
        total_var = win.var(axis=1).sum()
        pca_ratio = eigvals[-1] / (total_var + 1e-8)

        metrics[k] = [mean_ret, slope, real_vol, disp, pca_ratio]

    # Fix Leakage
    train_slice = metrics[:train_end_idx]
    min_vals = train_slice.min(axis=0)
    max_vals = train_slice.max(axis=0)
    
    diff = max_vals - min_vals
    diff[diff == 0] = 1.0
    metrics_norm = (metrics - min_vals) / diff

    return metrics_norm


def market_state_from_closes_v2(closes, close_col=-1, window=16, train_end_idx=756):
    """10-Dim Context Extraction"""
    close_px = closes[:, :, close_col]  # (N, T)
    rets = np.log(close_px[:, 1:] / close_px[:, :-1])  # (N, T-1)

    N, Tm1 = rets.shape
    W = window
    num_win = Tm1 - W + 1
    metrics = np.zeros((num_win, 10), dtype=np.float32)

    x = np.arange(W, dtype=np.float32)
    var_x = x.var()

    for k in range(num_win):
        win = rets[:, k:k + W]  # (N, W)
        idx_series = win.mean(axis=0)

        # 1) mean return
        mean_ret = win.mean()

        # 2) slope (market momentum)
        cov_x = ((x - x.mean()) * (idx_series - idx_series.mean())).mean()
        slope = cov_x / var_x

        # 3) realised vol
        real_vol = idx_series.std()

        # 4) dispersion
        disp = win.std(axis=0).mean()

        # 5) PCA (Systemic Risk)
        win_centered = win - win.mean(axis=1, keepdims=True)
        gram = (win_centered.T @ win_centered) / W
        eigvals = np.linalg.eigvalsh(gram)
        total_var = win.var(axis=1).sum()
        pca_ratio = eigvals[-1] / (total_var + 1e-8)

        # 6) Cross-Sectional Skewness (Tail Risk)
        win_mean = win.mean(axis=0, keepdims=True)
        win_std = win.std(axis=0, keepdims=True) + 1e-8
        skew_val = np.mean(((win - win_mean) / win_std)**3, axis=0).mean()

        # 7) Cross-Sectional Kurtosis (Fat-tail)
        kurt_val = np.mean(((win - win_mean) / win_std)**4, axis=0).mean() - 3.0

        # 8) Downside Volatility (Fear Gauge)
        downside_win = np.where(win < 0, win, 0)
        downside_vol = downside_win.std(axis=0).mean()

        # 9) Autocorrelation lag-1 (Market Efficiency / Trend)
        if W > 1:
            mu_idx = idx_series.mean()
            idx_centered = idx_series - mu_idx
            num = np.sum(idx_centered[:-1] * idx_centered[1:])
            den = np.sum(idx_centered**2) + 1e-8
            autocorr = num / den
        else:
            autocorr = 0.0

        # 10) Herd Behavior Index (Mean Pairwise Correlation)
        var_idx = idx_series.var()
        mean_var = win.var(axis=1).mean()
        mean_corr = (N * var_idx - mean_var) / ((N - 1) * mean_var + 1e-8)

        metrics[k] = [mean_ret, slope, real_vol, disp, pca_ratio, skew_val, kurt_val, downside_vol, autocorr, mean_corr]

    # Fix Leakage
    train_slice = metrics[:train_end_idx]
    min_vals = train_slice.min(axis=0)
    max_vals = train_slice.max(axis=0)
    
    diff = max_vals - min_vals
    diff[diff == 0] = 1.0
    metrics_norm = (metrics - min_vals) / diff

    return metrics_norm
