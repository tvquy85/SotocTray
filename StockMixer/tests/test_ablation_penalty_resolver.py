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
    assert abs(float(a) - 0.07) < 1e-6
    assert abs(float(l) - 0.11) < 1e-6

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_both"), device)
    assert abs(float(a) - 0.3) < 1e-6
    assert abs(float(l) - 0.4) < 1e-6

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_alpha_only"), device)
    assert abs(float(a) - 0.3) < 1e-6
    assert float(l) == 0.0

    a, l = resolve_penalty_coefficients(alpha, lamb, cfg("dynamic_lambda_only"), device)
    assert float(a) == 0.0
    assert abs(float(l) - 0.4) < 1e-6
