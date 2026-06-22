import numpy as np
import pandas as pd
import os
from src_tc.backtest_nonlinear import load_seed_predictions
from src_tc.backtest import backtest_topk

def calculate_regime_metrics(returns, regime_mask):
    if np.sum(regime_mask) == 0:
        return np.nan, np.nan
        
    regime_returns = returns[regime_mask]
    
    downside = regime_returns[regime_returns < 0]
    equity = np.cumprod(1 + regime_returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / (peak + 1e-12) - 1.0
    mdd = float(np.min(dd))
    
    ann_ret = float(np.mean(regime_returns) * 252)
    return ann_ret, mdd

def main():
    datasets = ['NASDAQ', 'SP500']
    methods = ['stockmixer_baseline', 'tc_stockmixer', 'ppo']
    
    results = []
    
    for dataset in datasets:
        test_preds, test_gt, test_mask = load_seed_predictions(dataset, 'stockmixer_baseline', [1])
        if test_gt is None:
            continue
            
        valid_counts = np.sum(test_mask > 0.5, axis=0)
        market_returns = np.sum(test_gt * (test_mask > 0.5), axis=0) / (valid_counts + 1e-8)
        
        rolling_ret = pd.Series(market_returns).rolling(60).sum().fillna(0).values
        bear_mask = rolling_ret < -0.05  # -5% over 60 days
        bull_mask = ~bear_mask
        
        print(f"\n--- Dataset {dataset}: Bear days = {np.sum(bear_mask)}, Bull days = {np.sum(bull_mask)} ---")
        
        for method in methods:
            curr_seeds = [1] if method == 'stockmixer_baseline' else [1, 2, 3]
            test_preds, _, _ = load_seed_predictions(dataset, method, curr_seeds)
            if test_preds is None:
                continue
                
            mu_test = np.mean(test_preds, axis=0)
            bt = backtest_topk(mu_test, test_gt, test_mask, topk=5, cost_bps=10.0)
            
            port_returns = bt['net_returns']
            
            bear_ann, bear_mdd = calculate_regime_metrics(port_returns, bear_mask)
            bull_ann, bull_mdd = calculate_regime_metrics(port_returns, bull_mask)
            
            results.append({
                'dataset': dataset,
                'method': method,
                'bull_ann_ret': bull_ann,
                'bull_mdd': bull_mdd,
                'bear_ann_ret': bear_ann,
                'bear_mdd': bear_mdd
            })
            
            print(f"[{method}]")
            print(f"  Bear -> Ann Ret: {bear_ann:.4f}, MDD: {bear_mdd:.4f}")
            print(f"  Bull -> Ann Ret: {bull_ann:.4f}, MDD: {bull_mdd:.4f}")

    df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/regime_analysis.csv', index=False)
    print("\nSaved to reports/regime_analysis.csv")

if __name__ == '__main__':
    main()
