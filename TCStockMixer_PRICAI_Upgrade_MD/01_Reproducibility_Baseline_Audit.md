# 01 — Reproducibility and Baseline Audit

## Objective

Tạo baseline artifact trước khi sửa model. Task này không cải thiện thuật toán; nó đóng băng config, split, shape dữ liệu và run output để so sánh sau mỗi nâng cấp.

## Allowed edit scope

- Add: `StockMixer/tests/`
- Add: `StockMixer/scripts/collect_baseline_audit.py`
- Do not modify model/loss/backtest.

## Step 1 — Add split tests

Create `StockMixer/tests/test_data_splits.py`:

```python
import pytest
from src_tc.data import make_offsets, assert_no_target_leakage


def test_make_offsets_non_empty_and_ordered():
    train, valid, test = make_offsets(
        valid_index=100, test_index=150, trade_dates=220, lookback=16, steps=1
    )
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0
    assert train.max() < valid.max() < test.max()


def test_no_target_leakage_train_valid_test():
    valid_index, test_index, trade_dates, lookback, steps = 100, 150, 220, 16, 1
    train, valid, test = make_offsets(valid_index, test_index, trade_dates, lookback, steps)
    assert_no_target_leakage(train, valid_index, test_index, lookback, steps, "train")
    assert_no_target_leakage(valid, valid_index, test_index, lookback, steps, "valid")
    assert_no_target_leakage(test, test_index, trade_dates, lookback, steps, "test")


def test_empty_split_raises_assertion():
    with pytest.raises(AssertionError):
        make_offsets(valid_index=10, test_index=15, trade_dates=20, lookback=16, steps=1)
```

Run:

```bash
cd StockMixer
python -m pytest tests/test_data_splits.py -q
```

Expected: `3 passed`.

## Step 2 — Add baseline audit script

Create `StockMixer/scripts/collect_baseline_audit.py`:

```python
import argparse
import json
from pathlib import Path
import torch

from src_tc.config import load_config
from src_tc.data import StockDatasetGPU, make_offsets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="audit/baseline_audit.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StockDatasetGPU(cfg, device)
    train, valid, test = make_offsets(
        cfg.valid_index, cfg.test_index, dataset.trade_dates, cfg.lookback_length, cfg.steps
    )

    audit = {
        "config_path": args.config,
        "config": vars(cfg),
        "device": str(device),
        "trade_dates": int(dataset.trade_dates),
        "eod_shape": list(dataset.eod.shape),
        "mask_shape": list(dataset.mask.shape),
        "gt_shape": list(dataset.gt.shape),
        "price_shape": list(dataset.price.shape),
        "offset_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "offset_ranges": {
            "train": [int(train.min()), int(train.max())],
            "valid": [int(valid.min()), int(valid.max())],
            "test": [int(test.min()), int(test.max())],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
```

Run:

```bash
cd StockMixer
python scripts/collect_baseline_audit.py \
  --config configs/crypto_tc_v2.yaml \
  --out audit/crypto_current_baseline_audit.json
```

Expected artifact:

```text
audit/crypto_current_baseline_audit.json
```

## Step 3 — Run current unmodified V2 once

```bash
cd StockMixer
python src_tc/run_experiment_v2.py --config configs/crypto_tc_v2.yaml --seed 12345678
```

Expected artifacts:

```text
results/crypto_tc_v2_seed12345678/metrics.json
results/crypto_tc_v2_seed12345678/history.csv
results/crypto_tc_v2_seed12345678/test_predictions.npz
results/crypto_tc_v2_seed12345678/config_used.yaml
```

## Final test case

```bash
cd StockMixer
python -m pytest tests/test_data_splits.py -q
python scripts/collect_baseline_audit.py --config configs/crypto_tc_v2.yaml --out audit/check.json
python - <<'PYTEST'
from pathlib import Path
import json
x = json.loads(Path('audit/check.json').read_text())
assert x['offset_counts']['train'] > 0
assert x['offset_counts']['valid'] > 0
assert x['offset_counts']['test'] > 0
print('AUDIT_PASS')
PYTEST
```

Pass condition: `AUDIT_PASS`.
