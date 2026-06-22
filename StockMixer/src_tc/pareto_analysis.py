import os
import subprocess
import yaml
import pandas as pd

def run_pareto_experiment(gamma_net):
    config_path = f"configs/nasdaq_pareto_{gamma_net}.yaml"
    
    # Base config
    with open("configs/nasdaq_tc.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['gamma_net'] = gamma_net
    cfg['epochs'] = 20  # Fast proxy training
    cfg['output_dir'] = f"results/nasdaq_pareto_{gamma_net}"
    cfg['seed'] = 1
    
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
        
    cmd = f"python -m src_tc.run_experiment --config {config_path}"
    print(f"Running Pareto experiment with gamma_net = {gamma_net}...")
    subprocess.run(cmd, shell=True, check=True)
    
    # Read metrics
    import json
    metrics_path = os.path.join(cfg['output_dir'], 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            test_bt = metrics['test_backtest']['top5']
            return {
                'gamma_net': gamma_net,
                'net_sharpe': test_bt['net_sharpe'],
                'gross_sharpe': test_bt['gross_sharpe'],
                'turnover': test_bt['avg_turnover'],
                'max_drawdown': test_bt['max_drawdown']
            }
    return None

def main():
    gammas = [5.0]
    results = []
    
    for g in gammas:
        try:
            res = run_pareto_experiment(g)
            if res:
                results.append(res)
        except subprocess.CalledProcessError as e:
            print(f"Failed for gamma {g}: {e}")
            
    df = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/pareto_frontier.csv', index=False)
    print("\nPareto Frontier Data:")
    print(df.to_markdown(index=False))

if __name__ == '__main__':
    main()
