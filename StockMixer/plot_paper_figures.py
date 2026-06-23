import matplotlib.pyplot as plt
import numpy as np
import os

# Create figures directory
os.makedirs("paper/figures", exist_ok=True)

# Data for Crypto Average (3 seeds, Top-10 Portfolio)
models = ['Baseline\nTC-StockMixer', 'Dynamic V1\n(Turnover Only)', 'Dynamic V2\n(Unified Regime-Aware)']
sharpe = [1.75, 3.15, 3.511] # Approx for baseline/V1, exact for V2
mdd = [-18.0, -17.0, -15.09] # in %
turnover = [0.57, 0.43, 0.419]

x = np.arange(len(models))
width = 0.25

# Use a premium style
plt.style.use('ggplot')
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Sharpe (Higher is better)
color1 = '#2b83ba'
bar1 = ax1.bar(x - width, sharpe, width, label='Net Sharpe Ratio', color=color1, edgecolor='black')
ax1.set_ylabel('Net Sharpe Ratio (Higher is Better)', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 4.5)

# Plot MDD (Closer to 0 is better)
ax2 = ax1.twinx()
color2 = '#d7191c'
bar2 = ax2.bar(x, mdd, width, label='Max Drawdown (%)', color=color2, edgecolor='black')
ax2.set_ylabel('Max Drawdown % (Closer to 0 is Better)', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-25, 0)

# Plot Turnover
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color3 = '#abdda4'
bar3 = ax3.bar(x + width, turnover, width, label='Turnover', color=color3, edgecolor='black')
ax3.set_ylabel('Turnover Rate (Lower is Better)', color='#4daf4a', fontsize=12, fontweight='bold')
ax3.tick_params(axis='y', labelcolor='#4daf4a')
ax3.set_ylim(0, 1.0)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
plt.title('Out-of-Distribution Robustness Evaluation (Crypto Market)', fontsize=14, fontweight='bold')

# Combine legends
bars = [bar1, bar2, bar3]
labels = [b.get_label() for b in bars]
ax1.legend(bars, labels, loc='upper left', framealpha=0.9)

# Add value labels
for bar in bar1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bar2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval - 1.0, f'{yval}%', ha='center', va='top', fontsize=10, fontweight='bold')
for bar in bar3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/crypto_robustness.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot generated successfully at paper/figures/crypto_robustness.png")
