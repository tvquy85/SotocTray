# 07 — Walk-Forward Validation on the Current Dataset

## Objective
Add walk-forward validation using the same existing dataset. This checks whether performance survives different train/valid/test windows instead of one fortunate split.

Do not add new markets in this step.

## Rationale
Financial time series are non-stationary. A single split can overstate performance. Walk-forward validation is required before claiming robustness.

## Files to add/edit

```text
StockMixer/scripts/make_walkforward_configs.py
StockMixer/tests/test_walkforward_config_generation.py
```

No model code should be modified in this task.

## Step 1 — Create config generator

Create `scripts/make_walkforward_configs.py`.

```python
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
    folds = make_folds(
        int(base["valid_index"]),
        int(base["test_index"]),
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
```

## Step 2 — Add unit test

Create `tests/test_walkforward_config_generation.py`.

```python
from scripts.make_walkforward_configs import make_folds


def test_make_folds_monotonic():
    folds = make_folds(valid_index=300, test_index=400, fold_count=3, step_size=50, min_train_end=200)
    assert folds == [(0, 300, 400), (1, 350, 450), (2, 400, 500)]


def test_make_folds_rejects_invalid_training_window():
    try:
        make_folds(valid_index=100, test_index=200, fold_count=1, step_size=50, min_train_end=200)
    except ValueError:
        return
    raise AssertionError("Expected invalid walk-forward split to fail")
```

## Step 3 — Generate current-data folds

Run:

```bash
cd StockMixer
pytest -q tests/test_walkforward_config_generation.py
python scripts/make_walkforward_configs.py \
  --base configs/crypto_tc_v2.yaml \
  --out_dir configs/walkforward_crypto \
  --fold_count 3 \
  --step_size 60
```

Inspect:

```bash
cat configs/walkforward_crypto/fold0.yaml | grep -E 'valid_index|test_index|output_dir'
cat configs/walkforward_crypto/fold1.yaml | grep -E 'valid_index|test_index|output_dir'
cat configs/walkforward_crypto/fold2.yaml | grep -E 'valid_index|test_index|output_dir'
```

## Step 4 — Run each fold with small seed count first

```bash
for f in configs/walkforward_crypto/fold*.yaml; do
  python scripts/run_multiseed.py --config "$f" --seeds 0,1,2 --tag wf3seed
done

python scripts/aggregate_multiseed.py --root results/walkforward_crypto --out reports/walkforward_crypto_3fold_3seed
```

After smoke validation, increase to 10 seeds:

```bash
for f in configs/walkforward_crypto/fold*.yaml; do
  python scripts/run_multiseed.py --config "$f" --seeds 0,1,2,3,4,5,6,7,8,9 --tag wf10seed
done

python scripts/aggregate_multiseed.py --root results/walkforward_crypto --out reports/walkforward_crypto_3fold_10seed
```

## Pass conditions

This step passes only if:

1. At least 3 walk-forward configs are generated.
2. Each fold has later `valid_index` and `test_index` than the previous fold.
3. No fold violates `assert_no_target_leakage` in `run_experiment_v2.py`.
4. The aggregated report contains at least 9 runs for smoke mode or 30 runs for full mode.
5. The mean test Net Sharpe remains positive in at least 2 of 3 folds.

## Failure handling

If performance collapses in later folds, do not hide it. Add a paper subsection:

```text
Limitations under temporal distribution shift
```

Then report which fold failed and inspect:

- volatility regime;
- asset universe changes;
- turnover spikes;
- alpha/lambda diagnostics;
- transaction cost sensitivity.

