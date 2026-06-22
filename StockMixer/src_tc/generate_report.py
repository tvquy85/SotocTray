import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    os.makedirs('tables', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    datasets = ['NASDAQ', 'SP500']
    methods = [('stockmixer_fixed', 'baseline'), ('tc_stockmixer', 'tc')]
    
    main_results = []
    
    for dataset in datasets:
        for method_name, method_dir_suffix in methods:
            for seed in [1, 2, 3]:
                d = f"results/{dataset.lower()}_{method_dir_suffix}_seed{seed}"
                m_file = os.path.join(d, 'metrics.json')
                if os.path.exists(m_file):
                    with open(m_file, 'r') as f:
                        m = json.load(f)
                    
                    row = {
                        'dataset': dataset,
                        'method': method_name,
                        'seed_or_ensemble': f'seed_{seed}',
                        'IC': m['test_prediction_metrics']['IC'],
                        'RIC': m['test_prediction_metrics']['RIC'],
                        'ICIR': m['test_prediction_metrics']['ICIR'],
                        'RICIR': m['test_prediction_metrics']['RICIR'],
                        'prec_10': m['test_prediction_metrics']['prec_10'],
                        'top5_gross_sharpe': m['test_backtest']['top5']['gross_sharpe'],
                        'top5_net_sharpe_10bps': m['test_backtest']['top5']['net_sharpe'],
                        'top5_sortino_10bps': m['test_backtest']['top5']['net_sortino'],
                        'top5_max_drawdown_10bps': m['test_backtest']['top5']['max_drawdown'],
                        'top5_avg_turnover': m['test_backtest']['top5']['avg_turnover'],
                        'top10_net_sharpe_10bps': m['test_backtest']['top10']['net_sharpe'],
                        'top10_avg_turnover': m['test_backtest']['top10']['avg_turnover'],
                    }
                    main_results.append(row)
        
        # Add ensemble results
        ens_file = f"tables/ensemble_{dataset}_tc_stockmixer.csv"
        if os.path.exists(ens_file):
            ens_df = pd.read_csv(ens_file)
            for _, r in ens_df.iterrows():
                row = {
                    'dataset': dataset,
                    'method': 'tc_stockmixer',
                    'seed_or_ensemble': f'ensemble_kappa_{r["kappa"]}',
                    'IC': r['IC'],
                    'RIC': r['RIC'],
                    'top5_net_sharpe_10bps': r['top5_net_sharpe_10bps'],
                    'top5_avg_turnover': r['top5_avg_turnover'],
                    'top5_max_drawdown_10bps': r['top5_max_drawdown']
                }
                main_results.append(row)
                
    if main_results:
        df = pd.DataFrame(main_results)
        df.to_csv('tables/main_results.csv', index=False)
        print("Generated tables/main_results.csv")
        
    # Generate Equity Curves
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        
        # Plot baseline seed 1
        b_file = f"results/{dataset.lower()}_baseline_seed1/test_equity_curve_top5.csv"
        if os.path.exists(b_file):
            b_df = pd.read_csv(b_file)
            plt.plot(b_df['test_equity'], label='Baseline (Seed 1)')
            
        # Plot TC seed 1
        t_file = f"results/{dataset.lower()}_tc_seed1/test_equity_curve_top5.csv"
        if os.path.exists(t_file):
            t_df = pd.read_csv(t_file)
            plt.plot(t_df['test_equity'], label='TC-StockMixer (Seed 1)')
            
        plt.title(f"{dataset} Equity Curve (Top 5, 10 bps)")
        plt.xlabel("Days")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"figures/equity_curve_{dataset}.png")
        plt.close()
        print(f"Generated figures/equity_curve_{dataset}.png")

if __name__ == '__main__':
    main()
