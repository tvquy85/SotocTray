import argparse
import copy
import os
import yaml


def make_folds(valid_index, test_index, fold_count, step_size, min_train_end):
    folds = []
    for fold in range(fold_count):
        vi = valid_index + fold * step_size
        ti = test_index + fold * step_size
        if vi <= min_train_end:
            raise ValueError("valid_index must remain after training warm-up")
        folds.append((fold, vi, ti))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out_dir", default="configs/walkforward_crypto")
    parser.add_argument("--fold_count", type=int, default=3)
    parser.add_argument("--step_size", type=int, default=60)
    parser.add_argument("--min_train_end", type=int, default=200)
    args = parser.parse_args()

    with open(args.base, "r") as f:
        base = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    base_vi = int(base["valid_index"]) - (args.fold_count - 1) * args.step_size
    base_ti = int(base["test_index"]) - (args.fold_count - 1) * args.step_size

    folds = make_folds(
        base_vi,
        base_ti,
        args.fold_count,
        args.step_size,
        args.min_train_end,
    )

    for fold, vi, ti in folds:
        cfg = copy.deepcopy(base)
        cfg["valid_index"] = vi
        cfg["test_index"] = ti
        cfg["output_dir"] = os.path.join("results", "walkforward_crypto", f"fold{fold}")
        path = os.path.join(args.out_dir, f"fold{fold}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(path, vi, ti)


if __name__ == "__main__":
    main()
