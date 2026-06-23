# 04 — Export Inference Diagnostics for Regime Interpretability

## Objective

Lưu time series của `alpha_t`, `lambda_t`, `valid_count`, và `market_return` trong validation/test. Đây là bằng chứng để vẽ figure: dynamic penalties có phản ứng với regime hay không.

## Allowed edit scope

- Modify: `StockMixer/src_tc/trainer_v2.py`
- Modify: `StockMixer/src_tc/run_experiment_v2.py`
- Add: `StockMixer/tests/test_inference_diagnostics.py`

## Step 1 — Extend `predict_over_offsets`

Change signature:

```python
def predict_over_offsets(model, dataset, offsets, cfg, return_diagnostics=False):
```

Before loop:

```python
diag_rows = []
```

Inside loop use mask-aware model call:

```python
out = model(data_batch, mask_batch)
prediction, alpha_t, lambda_t = out if isinstance(out, tuple) and len(out) == 3 else (out, None, None)
```

After appending prediction/gt/mask:

```python
if return_diagnostics:
    valid_bool = mask_batch.squeeze(-1) > 0.5
    valid_count = int(valid_bool.sum().detach().cpu().item())
    market_return = float(gt_batch.squeeze(-1)[valid_bool].mean().detach().cpu().item()) if valid_count > 0 else 0.0
    diag_rows.append({
        "offset": int(offset),
        "alpha_t": float(alpha_t.mean().detach().cpu().item()) if alpha_t is not None else float("nan"),
        "lambda_t": float(lambda_t.mean().detach().cpu().item()) if lambda_t is not None else float("nan"),
        "valid_count": valid_count,
        "market_return": market_return,
    })
```

At end:

```python
if return_diagnostics:
    return prediction_np, gt_np, mask_np, diag_rows
return prediction_np, gt_np, mask_np
```

## Step 2 — Update `evaluate_model`

Use:

```python
pred, gt, mask, diag_rows = predict_over_offsets(
    model, dataset, offsets, cfg, return_diagnostics=True
)
```

Return:

```python
return pred, gt, mask, pred_metrics, {'softmax': bt_softmax, 'top5': bt5, 'top10': bt10}, diag_rows
```

## Step 3 — Save diagnostics

In `run_experiment_v2.py`, update callers:

```python
_, _, _, valid_pred_metrics, valid_bt, _ = evaluate_model(model, dataset, valid_offsets, cfg)
valid_pred, valid_gt, valid_mask, valid_pm, valid_bt, valid_diag = evaluate_model(model, dataset, valid_offsets, cfg)
test_pred, test_gt, test_mask, test_pm, test_bt, test_diag = evaluate_model(model, dataset, test_offsets, cfg)
```

Update `save_all_results` signature to accept `valid_diag`, `test_diag`, then add:

```python
pd.DataFrame(valid_diag).to_csv(os.path.join(cfg.output_dir, 'valid_diagnostics.csv'), index=False)
pd.DataFrame(test_diag).to_csv(os.path.join(cfg.output_dir, 'test_diagnostics.csv'), index=False)
```

## Step 4 — Unit test

Create `StockMixer/tests/test_inference_diagnostics.py`:

```python
import numpy as np
import torch
from src_tc.trainer_v2 import predict_over_offsets


class DummyCfg:
    tau = 0.1


class DummyDataset:
    def get_batch(self, offset):
        x = torch.ones(4, 3, 2) * float(offset + 1)
        mask = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
        price = torch.ones(4, 1)
        gt = torch.tensor([[0.01], [0.02], [0.99], [-0.01]])
        return x, mask, price, gt


class DummyModel(torch.nn.Module):
    def forward(self, x, mask=None):
        pred = torch.arange(x.shape[0], dtype=torch.float32).view(-1, 1)
        return pred, torch.tensor([[0.7]]), torch.tensor([[1.2]])


def test_predict_over_offsets_returns_diagnostics():
    pred, gt, mask, diag = predict_over_offsets(
        DummyModel(), DummyDataset(), offsets=np.array([0, 1, 2]), cfg=DummyCfg(), return_diagnostics=True
    )
    assert pred.shape == (4, 3)
    assert len(diag) == 3
    assert diag[0]['valid_count'] == 3
    assert abs(diag[0]['alpha_t'] - 0.7) < 1e-6
    assert abs(diag[0]['lambda_t'] - 1.2) < 1e-6
```

Run:

```bash
cd StockMixer
python -m pytest tests/test_inference_diagnostics.py -q
python src_tc/run_experiment_v2.py --config configs/crypto_tc_v2.yaml --seed 12345678
python - <<'PYTEST'
import pandas as pd
from pathlib import Path
p = Path('results/crypto_tc_v2_seed12345678/test_diagnostics.csv')
assert p.exists()
df = pd.read_csv(p)
assert {'offset','alpha_t','lambda_t','valid_count','market_return'}.issubset(df.columns)
assert df['alpha_t'].between(0.1, 2.0).all()
assert df['lambda_t'].between(0.0, 2.0).all()
print('DIAGNOSTICS_PASS')
PYTEST
```
