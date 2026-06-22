import numpy as np
from scipy.stats import spearmanr


def safe_pearson(x, y):
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x, y):
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    val = spearmanr(x, y).correlation
    return float(val) if val == val else np.nan


def evaluate_prediction_metrics(prediction, ground_truth, mask, precision_k=10):
    assert prediction.shape == ground_truth.shape == mask.shape
    T = prediction.shape[1]
    mse_num = np.sum(((prediction - ground_truth) * mask) ** 2)
    mse_den = np.sum(mask) + 1e-12

    ic_list = []
    ric_list = []
    prec_list = []

    for t in range(T):
        valid = mask[:, t] > 0.5
        if valid.sum() < max(precision_k, 2):
            continue
        p = prediction[valid, t]
        g = ground_truth[valid, t]

        ic_list.append(safe_pearson(p, g))
        ric_list.append(safe_spearman(p, g))

        top_idx = np.argsort(p)[-precision_k:]
        prec_list.append(float(np.mean(g[top_idx] > 0)))

    return {
        'mse': float(mse_num / mse_den),
        'IC': float(np.nanmean(ic_list)),
        'RIC': float(np.nanmean(ric_list)),
        'ICIR': float(np.nanmean(ic_list) / (np.nanstd(ic_list) + 1e-12)),
        'RICIR': float(np.nanmean(ric_list) / (np.nanstd(ric_list) + 1e-12)),
        f'prec_{precision_k}': float(np.nanmean(prec_list)),
    }
