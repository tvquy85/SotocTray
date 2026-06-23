# 05 — Ablation Controls for Dynamic NetRank

## Objective
Create a small, deterministic ablation interface so every claimed component can be isolated on the **current dataset** before scaling.

This task answers the reviewer question:

> Does the gain come from dynamic `alpha_t`, dynamic `lambda_t`, both, or unrelated implementation noise?

## Required ablation modes

Add exactly these modes first:

```text
no_penalty
static_alpha_only
static_lambda_only
static_both
dynamic_alpha_only
dynamic_lambda_only
dynamic_both
random_context
```

Do not add more modes in this step.

## Files to edit

```text
StockMixer/src_tc/config.py
StockMixer/src_tc/losses_v2.py
StockMixer/src_tc/trainer_v2.py
StockMixer/src_tc/model_tc_v2.py
StockMixer/configs/crypto_tc_v2.yaml
```

## Step 1 — Extend config safely

In `src_tc/config.py`, add optional fields with defaults at the end of `ExperimentConfig`.

```python
# Ablation controls for reviewer-grade experiments
ablation_mode: str = "dynamic_both"
static_alpha: float = 0.05
static_lambda: float = 0.05
random_context_seed: int = 12345
```

Add the same keys to `configs/crypto_tc_v2.yaml`:

```yaml
ablation_mode: dynamic_both
static_alpha: 0.05
static_lambda: 0.05
random_context_seed: 12345
```

## Step 2 — Add penalty resolver

In `src_tc/losses_v2.py`, add this helper above `compute_tc_loss_v2`.

```python
def resolve_penalty_coefficients(alpha_t, lambda_t, cfg, device):
    """Return scalar alpha_eff and lambda_eff for ablation-safe loss.

    Modes:
      no_penalty: disable both regularizers.
      static_*: use cfg.static_alpha / cfg.static_lambda.
      dynamic_*: use model-produced alpha_t / lambda_t.
      random_context: handled upstream by model output; here treated as dynamic_both.
    """
    mode = getattr(cfg, "ablation_mode", "dynamic_both")
    static_alpha = torch.tensor(float(getattr(cfg, "static_alpha", 0.05)), device=device)
    static_lambda = torch.tensor(float(getattr(cfg, "static_lambda", 0.05)), device=device)
    zero = torch.tensor(0.0, device=device)

    if alpha_t is None:
        alpha_dyn = static_alpha
    else:
        alpha_dyn = alpha_t.mean()

    if lambda_t is None:
        lambda_dyn = static_lambda
    else:
        lambda_dyn = lambda_t.mean()

    if mode == "no_penalty":
        return zero, zero
    if mode == "static_alpha_only":
        return static_alpha, zero
    if mode == "static_lambda_only":
        return zero, static_lambda
    if mode == "static_both":
        return static_alpha, static_lambda
    if mode == "dynamic_alpha_only":
        return alpha_dyn, zero
    if mode == "dynamic_lambda_only":
        return zero, lambda_dyn
    if mode in {"dynamic_both", "random_context"}:
        return alpha_dyn, lambda_dyn

    raise ValueError(f"Unknown ablation_mode={mode}")
```

Then replace existing direct uses of `alpha_t.mean()` and `lambda_t.mean()` inside `compute_tc_loss_v2` with:

```python
alpha_eff, lambda_eff = resolve_penalty_coefficients(alpha_t, lambda_t, cfg, prediction.device)

dynamic_turnover = turnover * alpha_eff
dynamic_downside = downside_penalty * lambda_eff
```

Also add these fields to `stats`:

```python
"alpha_eff": alpha_eff.detach(),
"lambda_eff": lambda_eff.detach(),
"turnover_penalty": dynamic_turnover.detach(),
"downside_penalty_weighted": dynamic_downside.detach(),
```

## Step 3 — Implement `random_context`

In `model_tc_v2.py`, do not change the default forward path. Add an optional argument:

```python
def forward(self, inputs, ablation_mode=None, random_context_seed=None):
```

After computing `alpha_t` and `lambda_t`, add:

```python
if ablation_mode == "random_context":
    # Deterministic random context control. Same shape, no information from market state.
    gen = torch.Generator(device=inputs.device)
    seed = 12345 if random_context_seed is None else int(random_context_seed)
    gen.manual_seed(seed)
    alpha_t = 0.01 + 0.19 * torch.rand(alpha_t.shape, device=inputs.device, generator=gen)
    lambda_t = 0.01 + 0.19 * torch.rand(lambda_t.shape, device=inputs.device, generator=gen)
```

In `trainer_v2.py`, change model calls from:

```python
out = model(data_batch)
```

to:

```python
out = model(
    data_batch,
    ablation_mode=getattr(cfg, "ablation_mode", "dynamic_both"),
    random_context_seed=getattr(cfg, "random_context_seed", 12345),
)
```

Do this in both train and predict paths.

## Step 4 — Create mode-specific config copies

Create `scripts/make_ablation_configs.py`:

```python
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
```

## Step 5 — Test cases

Create `tests/test_ablation_penalty_resolver.py`.

```python
from types import SimpleNamespace
import torch
from src_tc.losses_v2 import resolve_penalty_coefficients


def cfg(mode):
    return SimpleNamespace(ablation_mode=mode, static_alpha=0.07, static_lambda=0.11)


def test_penalty_modes_are_correct():
    device = torch.device("cpu")
    alpha = torch.tensor([[0.2], [0.4]], device=device)
    lamb = torch.tensor([[0.3], [0.5]], device=device)

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("no_penalty"), device)
    assert float(a) == 0.0 and float(l) == 0.0

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("static_both"), device)
    assert abs(float(a) - 0.07) < 1e-9
    assert abs(float(l) - 0.11) < 1e-9

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_both"), device)
    assert abs(float(a) - 0.3) < 1e-9
    assert abs(float(l) - 0.4) < 1e-9

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_alpha_only"), device)
    assert abs(float(a) - 0.3) < 1e-9
    assert float(l) == 0.0

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_lambda_only"), device)
    assert float(a) == 0.0
    assert abs(float(l) - 0.4) < 1e-9
```

Run:

```bash
cd StockMixer
pytest -q tests/test_ablation_penalty_resolver.py
python scripts/make_ablation_configs.py --base configs/crypto_tc_v2.yaml
ls configs/ablations_crypto/*.yaml
```

## Pass conditions

This step passes only if:

1. All resolver tests pass.
2. Exactly 8 ablation config files are produced.
3. Default behavior remains `dynamic_both`.
4. `random_context` uses the same output shape as normal context.
5. `history.csv` contains `alpha_eff`, `lambda_eff`, `turnover_penalty`, and `downside_penalty_weighted` after one short training run.

## Run a one-epoch smoke experiment

Temporarily override epochs for a smoke run:

```bash
python - <<'PY'
import yaml
p = 'configs/ablations_crypto/dynamic_both.yaml'
with open(p) as f:
    cfg = yaml.safe_load(f)
cfg['epochs'] = 1
cfg['output_dir'] = 'results/smoke_ablation_dynamic_both'
with open('configs/smoke_dynamic_both.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

python -m src_tc.run_experiment_v2 --config configs/smoke_dynamic_both.yaml --seed 0
```

## Reviewer-facing output

After full runs, produce a table:

```text
mode | Net Sharpe | MDD | turnover | Sortino | alpha_eff mean | lambda_eff mean
```

Expected acceptance logic:

- `dynamic_both` should beat `static_both` or at least dominate it on risk-adjusted utility.
- `dynamic_both` should beat `random_context`; otherwise ContextNet is not learning useful market context.
- `dynamic_alpha_only` should mainly reduce turnover.
- `dynamic_lambda_only` should mainly improve downside/MDD/Sortino.

