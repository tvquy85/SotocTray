import torch
import torch.nn.functional as F


def masked_mse(pred, gt, mask):
    diff2 = ((pred - gt) * mask) ** 2
    return diff2.sum() / (mask.sum() + 1e-12)


def pairwise_rank_loss(scores, gt, mask, sample_size=None):
    scores = scores.squeeze(-1)
    gt = gt.squeeze(-1)
    mask = mask.squeeze(-1) > 0.5

    idx = torch.where(mask)[0]
    if idx.numel() < 2:
        return scores.new_tensor(0.0)

    if sample_size is not None and idx.numel() > sample_size:
        perm = torch.randperm(idx.numel(), device=idx.device)[:sample_size]
        idx = idx[perm]

    s = scores[idx]
    y = gt[idx]
    pred_diff = s[:, None] - s[None, :]
    gt_diff = y[:, None] - y[None, :]

    # Penalize opposite ordering. Equivalent spirit to StockMixer's pairwise rank loss.
    loss = F.relu(-(pred_diff * gt_diff))
    return loss.mean()


def masked_softmax(scores, mask, tau=0.1):
    scores = scores.squeeze(-1)
    mask_bool = mask.squeeze(-1) > 0.5
    scores = scores / max(tau, 1e-6)
    scores = scores.masked_fill(~mask_bool, -1e9)
    w = torch.softmax(scores, dim=0)
    w = torch.where(mask_bool, w, torch.zeros_like(w))
    w = w / (w.sum() + 1e-12)
    return w


def netrank_loss_v2(return_ratio, gt, mask, prev_w, tau, cost_bps, rho_concentration, alpha_t=None, lambda_t=None):
    w = masked_softmax(return_ratio, mask, tau=tau)
    future_ret = gt.squeeze(-1)

    gross = torch.sum(w * future_ret)
    if prev_w is None:
        turnover = torch.sum(torch.abs(w))
    else:
        turnover = torch.sum(torch.abs(w - prev_w.detach()))

    cost_rate = cost_bps / 10000.0
    
    # Calculate downside risk (squared downside returns)
    downside_return = torch.minimum(future_ret, torch.zeros_like(future_ret))
    downside_penalty = torch.sum(w * (downside_return ** 2))
    
    dynamic_turnover = turnover
    if alpha_t is not None:
        alpha_t = alpha_t.squeeze()
        dynamic_turnover = turnover * alpha_t.mean()
        
    dynamic_downside = downside_penalty
    if lambda_t is not None:
        lambda_t = lambda_t.squeeze()
        dynamic_downside = downside_penalty * lambda_t.mean()

    net = gross - cost_rate * dynamic_turnover - dynamic_downside
        
    concentration = torch.sum(w * w)

    loss = -net + rho_concentration * concentration
    stats = {
        'soft_gross': gross.detach(),
        'soft_turnover': turnover.detach(),
        'soft_downside': downside_penalty.detach(),
        'soft_net': net.detach(),
        'soft_concentration': concentration.detach(),
    }
    return loss, w, stats


def compute_tc_loss_v2(prediction, gt, base_price, mask, prev_w, cfg, alpha_t=None, lambda_t=None):
    return_ratio = (prediction - base_price) / (base_price + 1e-12)

    reg = masked_mse(return_ratio, gt, mask)
    rank = pairwise_rank_loss(return_ratio, gt, mask, sample_size=cfg.rank_sample_size)
    net_loss, w, net_stats = netrank_loss_v2(
        return_ratio=return_ratio,
        gt=gt,
        mask=mask,
        prev_w=prev_w,
        tau=cfg.tau,
        cost_bps=cfg.train_cost_bps,
        rho_concentration=cfg.rho_concentration,
        alpha_t=alpha_t,
        lambda_t=lambda_t
    )

    total = cfg.beta_reg * reg + cfg.alpha_rank * rank + cfg.gamma_net * net_loss
    stats = {
        'loss': total.detach(),
        'reg_loss': reg.detach(),
        'rank_loss': rank.detach(),
        'net_loss': net_loss.detach(),
        **net_stats,
    }
    return total, return_ratio, w, stats
