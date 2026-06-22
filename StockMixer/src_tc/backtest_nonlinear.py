import os
import argparse
import numpy as np
import pandas as pd
from src_tc.backtest import summarize_returns, topk_equal_weights

def backtest_harsh_impact(prediction, ground_truth, mask, topk=5, linear_bps=10.0, impact_bps=50.0):
    assert prediction.shape == ground_truth.shape == mask.shape
    N, T = prediction.shape
    c_linear = linear_bps / 10000.0
    c_impact = impact_bps / 10000.0
    prev_w = np.zeros(N, dtype=np.float64)

    gross_returns = []
    net_returns = []
    turnovers = []
    holdings = []

    for t in range(T):
        valid = mask[:, t] > 0.5
        w = topk_equal_weights(prediction[:, t], valid, topk)
        gross = float(np.sum(w * ground_truth[:, t]))
        
        dw = np.abs(w - prev_w)
        total_turnover = float(np.sum(dw))
        
        # Cost = linear cost + quadratic impact penalty for large reallocations
        cost = np.sum(c_linear * dw + c_impact * (dw ** 2))
        
        net = gross - cost

        gross_returns.append(gross)
        net_returns.append(net)
        turnovers.append(total_turnover)
        holdings.append(int(np.sum(w > 0)))
        prev_w = w

    gross_returns = np.array(gross_returns, dtype=np.float64)
    net_returns = np.array(net_returns, dtype=np.float64)
    turnovers = np.array(turnovers, dtype=np.float64)

    return summarize_returns(gross_returns, net_returns, turnovers, holdings)

def load_seed_predictions(dataset, method, seeds):
    test_preds = []
    test_gt, test_mask = None, None
    for seed in seeds:
        if method == 'tc_stockmixer':
            base_dir = f"results/{dataset.lower()}_tc_seed{seed}"
        elif method == 'stockmixer_baseline':
            base_dir = f"results/{dataset.lower()}_baseline_seed{seed}"
        else:
            base_dir = f"results/{dataset.lower()}_ppo_seed{seed}"
            if not os.path.exists(base_dir):
                return None, None, None
                
        t_data = np.load(os.path.join(base_dir, 'test_predictions.npz'))
        test_preds.append(t_data['pred'])
        if test_gt is None:
            test_gt, test_mask = t_data['gt'], t_data['mask']
            
    return np.array(test_preds), test_gt, test_mask

def main():
    datasets = ['NASDAQ', 'SP500']
    methods = ['stockmixer_baseline', 'tc_stockmixer', 'ppo']
    seeds = [1, 2, 3]
    
    results = []
    
    for dataset in datasets:
        for method in methods:
            curr_seeds = [1] if method == 'stockmixer_baseline' else seeds
            test_preds, test_gt, test_mask = load_seed_predictions(dataset, method, curr_seeds)
            if test_preds is None:
                continue
                
            # For simplicity, we just use mean ensemble (kappa=0) to evaluate impact
            mu_test = np.mean(test_preds, axis=0)
            
            # Linear only (Standard 10 bps)
            bt_std = backtest_harsh_impact(mu_test, test_gt, test_mask, topk=5, linear_bps=10.0, impact_bps=0.0)
            
            # Non-linear (10 bps + 100 bps impact penalty)
            bt_nl = backtest_harsh_impact(mu_test, test_gt, test_mask, topk=5, linear_bps=10.0, impact_bps=100.0)
            
            results.append({
                'dataset': dataset,
                'method': method,
                'turnover': bt_std['avg_turnover'],
                'standard_net_sharpe': bt_std['net_sharpe'],
                'nonlinear_net_sharpe': bt_nl['net_sharpe'],
                'sharpe_drop': bt_std['net_sharpe'] - bt_nl['net_sharpe']
            })
            
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/nonlinear_impact_analysis.csv', index=False)
    print("\nSaved to reports/nonlinear_impact_analysis.csv")

if __name__ == '__main__':
    main()
