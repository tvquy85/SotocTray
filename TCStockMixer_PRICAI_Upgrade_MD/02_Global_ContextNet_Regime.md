# 02 — Replace Per-Stock ContextNet with Global Market Regime Encoder

## Objective

Sửa mismatch giữa paper và code: `ContextNet` phải nhận global market state từ lookback window và mask valid assets. Sau task này, `alpha_t` và `lambda_t` là market-level scalars `[1, 1]` cho mỗi rebalance date.

## Allowed edit scope

- Modify: `StockMixer/src_tc/model_tc_v2.py`
- Modify: `StockMixer/src_tc/trainer_v2.py`
- Add: `StockMixer/tests/test_contextnet_global.py`
- Do not modify loss/backtest.

## Step 1 — Add `GlobalContextNet`

In `StockMixer/src_tc/model_tc_v2.py`, add this class before `StockMixerBackboneV2`:

```python
class GlobalContextNet(nn.Module):
    def __init__(self, time_steps, channels, hidden_dim=32,
                 alpha_min=0.1, alpha_max=2.0, lambda_min=0.0, lambda_max=2.0):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.norm = nn.LayerNorm([time_steps, channels])
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_steps * channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def compute_market_state(self, inputs, mask=None):
        # inputs: [num_assets, time_steps, channels]
        # mask:   [num_assets, 1]
        if mask is None:
            return inputs.mean(dim=0, keepdim=True)
        valid = (mask.squeeze(-1) > 0.5).float().view(-1, 1, 1)
        denom = valid.sum().clamp_min(1.0)
        return (inputs * valid).sum(dim=0, keepdim=True) / denom

    def forward(self, inputs, mask=None):
        market_state = self.compute_market_state(inputs, mask)
        ctx = self.net(self.norm(market_state))
        alpha_t = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(ctx[:, 0:1])
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * torch.sigmoid(ctx[:, 1:2])
        return alpha_t, lambda_t, market_state
```

Replace current `self.context_net = nn.Sequential(...)` with:

```python
self.context_net = GlobalContextNet(time_steps=time_steps, channels=channels)
```

## Step 2 — Update model forward

Change:

```python
def forward(self, inputs):
```

to:

```python
def forward(self, inputs, mask=None, return_context=False):
```

Replace current context block:

```python
market_state = inputs.mean(dim=2)
ctx_out = self.context_net(market_state)
alpha_t = 0.1 + 1.9 * torch.sigmoid(ctx_out[:, 0:1])
lambda_t = 0.0 + 2.0 * torch.sigmoid(ctx_out[:, 1:2])
return y + z, alpha_t, lambda_t
```

with:

```python
alpha_t, lambda_t, market_state = self.context_net(inputs, mask=mask)
score = y + z
if return_context:
    return score, alpha_t, lambda_t, {"market_state": market_state.detach()}
return score, alpha_t, lambda_t
```

## Step 3 — Pass mask into model

In `StockMixer/src_tc/trainer_v2.py`, replace every:

```python
out = model(data_batch)
```

with:

```python
out = model(data_batch, mask_batch)
```

Do this in both `train_one_epoch` and `predict_over_offsets`.

## Step 4 — Add tests

Create `StockMixer/tests/test_contextnet_global.py`:

```python
import torch
from src_tc.model_tc_v2 import GlobalContextNet


def test_global_context_outputs_scalar_penalties():
    torch.manual_seed(7)
    net = GlobalContextNet(time_steps=16, channels=5)
    x = torch.randn(10, 16, 5)
    mask = torch.ones(10, 1)
    alpha, lambd, state = net(x, mask)
    assert alpha.shape == (1, 1)
    assert lambd.shape == (1, 1)
    assert state.shape == (1, 16, 5)
    assert 0.1 <= float(alpha) <= 2.0
    assert 0.0 <= float(lambd) <= 2.0


def test_global_context_is_stock_permutation_invariant():
    torch.manual_seed(11)
    net = GlobalContextNet(time_steps=16, channels=5)
    x = torch.randn(12, 16, 5)
    mask = torch.ones(12, 1)
    perm = torch.randperm(12)
    a1, l1, s1 = net(x, mask)
    a2, l2, s2 = net(x[perm], mask[perm])
    assert torch.allclose(a1, a2, atol=1e-6)
    assert torch.allclose(l1, l2, atol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6)


def test_global_context_ignores_masked_asset_values():
    torch.manual_seed(13)
    net = GlobalContextNet(time_steps=16, channels=5)
    x = torch.randn(8, 16, 5)
    mask = torch.ones(8, 1)
    mask[-1] = 0.0
    a1, l1, s1 = net(x, mask)
    x2 = x.clone()
    x2[-1] = 1_000_000.0
    a2, l2, s2 = net(x2, mask)
    assert torch.allclose(a1, a2, atol=1e-6)
    assert torch.allclose(l1, l2, atol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6)
```

Run:

```bash
cd StockMixer
python -m pytest tests/test_contextnet_global.py -q
python src_tc/run_experiment_v2.py --config configs/crypto_tc_v2.yaml --seed 12345678
```

Pass condition: tests pass and experiment creates `metrics.json` without shape errors.
