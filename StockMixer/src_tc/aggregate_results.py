import argparse
import os
import numpy as np
import pandas as pd
from src_tc.metrics import evaluate_prediction_metrics
from src_tc.backtest import backtest_topk

def load_seed_predictions(dataset, method, seeds):
    valid_preds, test_preds = [], []
    valid_gt, test_gt = None, None
    valid_mask, test_mask = None, None
    
    for seed in seeds:
        if method == 'tc_stockmixer':
            base_dir = f"results/{dataset.lower()}_tc_seed{seed}"
        else:
            base_dir = f"results/{dataset.lower()}_baseline_seed{seed}"
            
        v_data = np.load(os.path.join(base_dir, 'valid_predictions.npz'))
        t_data = np.load(os.path.join(base_dir, 'test_predictions.npz'))
        
        valid_preds.append(v_data['pred'])
        test_preds.append(t_data['pred'])
        
        if valid_gt is None:
            valid_gt, valid_mask = v_data['gt'], v_data['mask']
            test_gt, test_mask = t_data['gt'], t_data['mask']
            
    return np.array(valid_preds), np.array(test_preds), valid_gt, valid_mask, test_gt, test_mask

def tune_kappa(valid_preds, valid_gt, valid_mask):
    mu = np.mean(valid_preds, axis=0)
    sigma = np.std(valid_preds, axis=0)
    
    best_kappa = 0.0
    best_score = -1e18
    
    for kappa in [0.0, 0.25, 0.5, 1.0]:
        adjusted_score = mu - kappa * sigma
        bt = backtest_topk(adjusted_score, valid_gt, valid_mask, topk=5, cost_bps=10.0)
        if bt['net_sharpe'] > best_score:
            best_score = bt['net_sharpe']
            best_kappa = kappa
            
    return best_kappa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    args = parser.parse_args()

    print(f"Aggregating {args.dataset} {args.method} for seeds {args.seeds}...")
    valid_preds, test_preds, valid_gt, valid_mask, test_gt, test_mask = load_seed_predictions(args.dataset, args.method, args.seeds)

    best_kappa = tune_kappa(valid_preds, valid_gt, valid_mask)
    print(f"Selected best kappa on validation: {best_kappa}")

    mu_test = np.mean(test_preds, axis=0)
    sigma_test = np.std(test_preds, axis=0)
    
    # 1. Ensemble without uncertainty gate: kappa = 0
    adj_score_0 = mu_test
    pm_0 = evaluate_prediction_metrics(adj_score_0, test_gt, test_mask)
    bt_0_top5 = backtest_topk(adj_score_0, test_gt, test_mask, topk=5, cost_bps=10.0)
    
    # 2. Ensemble with uncertainty gate: selected kappa
    adj_score_k = mu_test - best_kappa * sigma_test
    pm_k = evaluate_prediction_metrics(adj_score_k, test_gt, test_mask)
    bt_k_top5 = backtest_topk(adj_score_k, test_gt, test_mask, topk=5, cost_bps=10.0)

    os.makedirs('tables', exist_ok=True)
    out_file = f"tables/ensemble_{args.dataset}_{args.method}.csv"
    
    results = [
        {
            'dataset': args.dataset,
            'method': args.method,
            'kappa': 0.0,
            'IC': pm_0['IC'],
            'RIC': pm_0['RIC'],
            'top5_net_sharpe_10bps': bt_0_top5['net_sharpe'],
            'top5_avg_turnover': bt_0_top5['avg_turnover'],
            'top5_max_drawdown': bt_0_top5['max_drawdown']
        },
        {
            'dataset': args.dataset,
            'method': args.method,
            'kappa': best_kappa,
            'IC': pm_k['IC'],
            'RIC': pm_k['RIC'],
            'top5_net_sharpe_10bps': bt_k_top5['net_sharpe'],
            'top5_avg_turnover': bt_k_top5['avg_turnover'],
            'top5_max_drawdown': bt_k_top5['max_drawdown']
        }
    ]
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"Saved ensemble evaluation to {out_file}")

if __name__ == '__main__':
    main()
