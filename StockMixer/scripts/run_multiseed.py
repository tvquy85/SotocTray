import argparse
import os
import subprocess
import sys
import yaml
from concurrent.futures import ProcessPoolExecutor


def run_seed(args_tuple):
    sys_exec, tmp_path, seed, dry_run = args_tuple
    cmd = [sys_exec, "-m", "src_tc.run_experiment_v2", "--config", tmp_path, "--seed", str(seed)]
    print("RUN", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", required=True, help="Comma list, e.g. 0,1,2,3,4")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base = yaml.safe_load(f)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    base_out = base["output_dir"]
    tag = args.tag or os.path.basename(args.config).replace(".yaml", "")

    tasks = []
    for seed in seeds:
        cfg = dict(base)
        cfg["seed"] = seed
        cfg["output_dir"] = os.path.join(base_out, tag, f"seed{seed}")
        tmp_path = f"configs/tmp_{tag}_seed{seed}.yaml"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        tasks.append((sys.executable, tmp_path, seed, args.dry_run))

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            executor.map(run_seed, tasks)
    else:
        for t in tasks:
            run_seed(t)


if __name__ == "__main__":
    main()
