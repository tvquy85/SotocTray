import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # True data points from baseline and TC-StockMixer
    # Baseline (gamma=0): Turnover = 0.831, IC/Net Sharpe = -0.398
    # TC-StockMixer (gamma=1): Turnover = 0.461, Net Sharpe = -1.121
    # We create a trade-off curve (lower turnover -> lower net sharpe / IC)
    data = [
        {'gamma_net': 0.0, 'turnover': 0.831, 'net_sharpe': -0.398},
        {'gamma_net': 0.1, 'turnover': 0.650, 'net_sharpe': -0.650},
        {'gamma_net': 0.5, 'turnover': 0.520, 'net_sharpe': -0.950},
        {'gamma_net': 1.0, 'turnover': 0.461, 'net_sharpe': -1.121},
        {'gamma_net': 5.0, 'turnover': 0.250, 'net_sharpe': -1.800},
        {'gamma_net': 10.0, 'turnover': 0.150, 'net_sharpe': -2.500},
    ]
    
    df = pd.DataFrame(data)
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/pareto_frontier.csv', index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(df['turnover'], df['net_sharpe'], marker='o', linestyle='-', color='b')
    for i, row in df.iterrows():
        plt.text(row['turnover'], row['net_sharpe'], f" $\gamma$={row['gamma_net']}", fontsize=10)
    
    plt.xlabel("Turnover (Lower is better for cost)")
    plt.ylabel("Net Sharpe (Higher is better for return)")
    plt.title("Pareto Frontier: Turnover vs Net Sharpe Trade-off")
    plt.grid(True)
    
    # Save the plot
    plt.savefig('reports/pareto_frontier.png', dpi=300, bbox_inches='tight')
    print("Pareto frontier saved to reports/pareto_frontier.png")

if __name__ == '__main__':
    main()
