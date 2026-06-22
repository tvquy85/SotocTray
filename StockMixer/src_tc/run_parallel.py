import subprocess
import concurrent.futures
import time

commands = [
    "python -m src_tc.run_experiment --config configs/nasdaq_baseline.yaml --seed 1",
    "python -m src_tc.run_experiment --config configs/sp500_baseline.yaml --seed 1",
    "python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed 1",
    "python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed 2",
    "python -m src_tc.run_experiment --config configs/nasdaq_tc.yaml --seed 3",
    "python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed 1",
    "python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed 2",
    "python -m src_tc.run_experiment --config configs/sp500_tc.yaml --seed 3",
]

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    duration = time.time() - t0
    print(f"Finished: {cmd} in {duration:.1f}s")

def main():
    print("Running 12 tasks with max 4 parallel workers...")
    # Use ThreadPoolExecutor because subprocess releases the GIL
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_cmd, commands)
    
    print("All training tasks finished. Running aggregations...")
    subprocess.run("python -m src_tc.aggregate_results --dataset NASDAQ --method tc_stockmixer --seeds 1 2 3", shell=True)
    subprocess.run("python -m src_tc.aggregate_results --dataset SP500 --method tc_stockmixer --seeds 1 2 3", shell=True)
    
    print("Generating report...")
    subprocess.run("python -m src_tc.generate_report", shell=True)
    print("Done!")

if __name__ == '__main__':
    main()
