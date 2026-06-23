import subprocess
import concurrent.futures

commands = [
    "python -u -m src_tc.rl_baseline_train --config configs/nasdaq_tc.yaml --seed 1 > ppo_nasdaq_seed1.log 2>&1",
    "python -u -m src_tc.rl_baseline_train --config configs/nasdaq_tc.yaml --seed 2 > ppo_nasdaq_seed2.log 2>&1",
    "python -u -m src_tc.rl_baseline_train --config configs/nasdaq_tc.yaml --seed 3 > ppo_nasdaq_seed3.log 2>&1",
    "python -u -m src_tc.rl_baseline_train --config configs/sp500_tc.yaml --seed 1 > ppo_sp500_seed1.log 2>&1",
    "python -u -m src_tc.rl_baseline_train --config configs/sp500_tc.yaml --seed 2 > ppo_sp500_seed2.log 2>&1",
    "python -u -m src_tc.rl_baseline_train --config configs/sp500_tc.yaml --seed 3 > ppo_sp500_seed3.log 2>&1",
]

def run_command(cmd):
    print(f"Starting: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    print(f"Finished: {cmd}")

if __name__ == "__main__":
    print("Running PPO baseline training with max 2 parallel workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(run_command, commands)
    print("All PPO training tasks finished!")
