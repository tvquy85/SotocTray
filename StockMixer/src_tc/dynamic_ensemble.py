import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src_tc.backtest import backtest_topk
from src_tc.metrics import evaluate_prediction_metrics

def load_seed_predictions(dataset, method, seeds):
    preds_list = []
    gt = None
    mask = None
    for s in seeds:
        base_dir = f'results/{dataset.lower()}_{method}_seed{s}'
        try:
            t_data = np.load(os.path.join(base_dir, 'test_predictions.npz'))
            preds_list.append(t_data['pred'])
            if gt is None:
                gt = t_data['gt']
                mask = t_data['mask']
        except FileNotFoundError:
            print(f"Warning: Missing data for {base_dir}")
            return None, None, None
            
    return np.array(preds_list), gt, mask

def calculate_daily_ic(pred, gt, mask):
    # pred: [T, N], gt: [T, N], mask: [T, N]
    T = pred.shape[0]
    ic_list = []
    for t in range(T):
        valid_idx = np.where(mask[t] > 0.5)[0]
        if len(valid_idx) < 2:
            ic_list.append(0.0)
            continue
        p = pred[t, valid_idx]
        g = gt[t, valid_idx]
        if np.std(p) < 1e-8 or np.std(g) < 1e-8:
            ic_list.append(0.0)
        else:
            rho, _ = spearmanr(p, g)
            ic_list.append(rho if not np.isnan(rho) else 0.0)
    return np.array(ic_list)

def main():
    dataset = 'NASDAQ'
    method = 'tc'
    seeds = [1, 2, 3]
    
    # preds: [num_seeds, T, N]
    preds, gt, mask = load_seed_predictions(dataset, method, seeds)
    if preds is None:
        print("Predictions not found.")
        return
        
    num_seeds, T, N = preds.shape
    
    # 1. Calculate daily IC for each seed
    daily_ic = np.zeros((num_seeds, T))
    for i in range(num_seeds):
        daily_ic[i] = calculate_daily_ic(preds[i], gt, mask)
        
    # 2. Rolling 20-day mean IC
    rolling_ic = np.zeros((num_seeds, T))
    W = 20
    for i in range(num_seeds):
        series = pd.Series(daily_ic[i])
        # Shift by 1 so we only use PAST information, not current day's GT!
        shifted_mean = series.rolling(window=W, min_periods=1).mean().shift(1).fillna(0.0).values
        rolling_ic[i] = shifted_mean
        
    # 3. Dynamic Attention (Softmax)
    tau = 0.05  # Temperature
    weights = np.zeros((num_seeds, T))
    for t in range(T):
        if t < 5:
            # Not enough history, uniform weight
            weights[:, t] = 1.0 / num_seeds
        else:
            scores = rolling_ic[:, t] / tau
            scores -= np.max(scores) # numerical stability
            exp_s = np.exp(scores)
            weights[:, t] = exp_s / np.sum(exp_s)
            
    # 4. Ensemble Predictions
    ensemble_preds = np.zeros((T, N))
    for t in range(T):
        for i in range(num_seeds):
            ensemble_preds[t] += weights[i, t] * preds[i, t]
            
    # Naive ensemble for comparison
    naive_preds = np.mean(preds, axis=0)
    
    # Evaluate
    naive_bt = backtest_topk(naive_preds, gt, mask, topk=5, cost_bps=10.0)
    dynamic_bt = backtest_topk(ensemble_preds, gt, mask, topk=5, cost_bps=10.0)
    
    print("--- Naive Ensemble ---")
    print(f"Net Sharpe: {naive_bt['net_sharpe']:.4f}, Turnover: {naive_bt['avg_turnover']:.4f}")
    
    print("\n--- Dynamic IC-Attention Ensemble ---")
    print(f"Net Sharpe: {dynamic_bt['net_sharpe']:.4f}, Turnover: {dynamic_bt['avg_turnover']:.4f}")
    print(f"Max Drawdown: {dynamic_bt['max_drawdown']:.4f}")

if __name__ == '__main__':
    main()
