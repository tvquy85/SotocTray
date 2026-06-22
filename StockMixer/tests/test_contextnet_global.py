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
