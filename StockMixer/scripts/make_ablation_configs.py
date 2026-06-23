import argparse
import copy
import os
import yaml

MODES = [
    "no_penalty",
    "static_alpha_only",
    "static_lambda_only",
    "static_both",
    "dynamic_alpha_only",
    "dynamic_lambda_only",
    "dynamic_both",
    "random_context",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out_dir", default="configs/ablations_crypto")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.base, "r") as f:
        base = yaml.safe_load(f)

    for mode in MODES:
        cfg = copy.deepcopy(base)
        cfg["ablation_mode"] = mode
        cfg["output_dir"] = os.path.join("results", "ablations_crypto", mode)
        path = os.path.join(args.out_dir, f"{mode}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(path)


if __name__ == "__main__":
    main()
