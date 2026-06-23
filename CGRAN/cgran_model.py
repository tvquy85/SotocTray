import torch
import torch.nn as nn
import torch.nn.functional as F

from gMLP import gMLP

acv = nn.Hardswish()

def get_cgran_loss(prediction, alpha_pred, ground_truth, base_price, mask, stock_num, args):
    """
    Computes MSE, Pairwise Rank Loss, and Residual Alpha Loss.
    """
    device = prediction.device
    all_one = torch.ones(stock_num, 1, dtype=torch.float32).to(device)
    
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    
    # Residual Alpha Loss (RA-Loss)
    epsilon = 1e-8
    market_return = torch.sum(ground_truth * mask, dim=0) / (torch.sum(mask, dim=0) + epsilon)
    market_return = market_return.unsqueeze(0).expand(stock_num, -1)
    alpha_target = ground_truth - market_return
    
    alpha_loss = F.mse_loss(alpha_pred * mask, alpha_target * mask)

    loss = reg_loss + args.alpha * rank_loss + args.lambda_alpha * alpha_loss
    
    return loss, reg_loss, rank_loss, alpha_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale1_mix_layer = Mixer2dTriU(time_step // 2, channel)

    def forward(self, inputs, x1):
        x = self.mix_layer(inputs)
        x1 = self.scale1_mix_layer(x1)
        return torch.cat([inputs, x, x1], dim=1)


class ContextGatedSpatialMixer(nn.Module):
    def __init__(self, time_step, stocks, hidden_dim=20, depth=2, ctx_dim=5):
        super(ContextGatedSpatialMixer, self).__init__()
        self.time_step = time_step
        self.gMlp = gMLP(time_step * 2 + time_step // 2, stocks, hidden_dim, depth, ctx_dim=ctx_dim)

    def forward(self, inputs, ctx):
        x = inputs.permute(1, 0)
        x = self.gMlp(x, ctx)
        x = x.permute(1, 0)
        return x


class CGRAN(nn.Module):
    def __init__(self, stocks, time_steps, channels, market, depth=2):
        super(CGRAN, self).__init__()
        self.mixer = MultTime2dMixer(time_steps, channels)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)
        self.scale1 = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        
        # Macro-Context Gated Spatial Mixing
        self.spatial_mixer = ContextGatedSpatialMixer(time_steps, stocks, market, depth, ctx_dim=market)
        self.spatial_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)
        
        # Multi-task Alpha Head
        self.alpha_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)

    def forward(self, inputs, ctx):
        x1 = inputs.permute(0, 2, 1)
        x1 = self.scale1(x1)
        x1 = x1.permute(0, 2, 1)

        y = self.mixer(inputs, x1)
        y = self.channel_fc(y).squeeze(-1)

        # Temporal representation
        temporal_pred = self.time_fc(y)
        
        # Context-Gated Spatial representation
        z = self.spatial_mixer(y, ctx)
        spatial_pred = self.spatial_fc(z)
        
        # Total Prediction
        total_pred = temporal_pred + spatial_pred
        
        # Residual Alpha Prediction from the gated cross-sectional features
        alpha_pred = self.alpha_fc(z)
        
        return total_pred, alpha_pred
