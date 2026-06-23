import subprocess
import concurrent.futures
import time

commands = [
    "python -u scgran_train_tc.py --dataset NASDAQ --seed 1 > tc_scgran_nasdaq_seed1.log 2>&1",
    "python -u scgran_train_tc.py --dataset NASDAQ --seed 2 > tc_scgran_nasdaq_seed2.log 2>&1",
    "python -u scgran_train_tc.py --dataset NASDAQ --seed 3 > tc_scgran_nasdaq_seed3.log 2>&1",
    "python -u scgran_train_tc.py --dataset SP500 --seed 1 > tc_scgran_sp500_seed1.log 2>&1",
    "python -u scgran_train_tc.py --dataset SP500 --seed 2 > tc_scgran_sp500_seed2.log 2>&1",
    "python -u scgran_train_tc.py --dataset SP500 --seed 3 > tc_scgran_sp500_seed3.log 2>&1",
]

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    t0 = time.time()
    subprocess.run(cmd, shell=True, check=True)
    duration = time.time() - t0
    print(f"Finished: {cmd} in {duration:.1f}s")

def main():
    print("Running TC-SCGRAN training with max 2 parallel workers to avoid OOM...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(run_cmd, commands)
    
    print("All training tasks finished!")

if __name__ == '__main__':
    main()
