import torch
import torch.nn as nn
import torch.nn.functional as F

acv = nn.Hardswish()

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
            [nn.Linear(i + 1, 1) for i in range(time_step)]
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

class SpectralRegimeMixer(nn.Module):
    def __init__(self, time_steps, channels, latent_dim=32):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.freq_len = time_steps // 2 + 1
        
        # 1. Latent Regime Encoder (Takes market_x of shape [T, C] pooled -> [C])
        # We will just flatten T and C, or average over T. Let's average over T to get [C].
        self.regime_encoder = nn.Sequential(
            nn.Linear(channels, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 2. Spectral Filter Generator
        self.freq_filter = nn.Linear(latent_dim, self.freq_len * channels)
        
        # 3. Dynamic Loss Conditioners
        self.alpha_weight_gen = nn.Linear(latent_dim, 1)
        self.beta_weight_gen = nn.Linear(latent_dim, 1)

    def forward(self, x):
        """
        x: [N, T, C] - Stock Features
        """
        N, T, C = x.shape
        
        # --- A. Latent Regime Discovery ---
        # Derive broad market feature by averaging across stocks
        market_x = x.mean(dim=0) # [T, C]
        mkt_pooled = market_x.mean(dim=0) # [C]
        Z = self.regime_encoder(mkt_pooled) # [latent_dim]
        
        # --- B. Context-Aware Frequency Filtering ---
        # Transform to frequency domain across Time dimension (dim=1)
        X_freq = torch.fft.rfft(x, n=T, dim=1) # [N, Freq, C] (Complex)
        
        # Generate Frequency Filter from Latent Regime Z
        # Shape: [1, Freq, C] to broadcast across N stocks
        H_filter = torch.sigmoid(self.freq_filter(Z)).view(1, self.freq_len, C)
        
        # Modulate amplitudes (real filter applied to complex numbers)
        X_freq_filtered = X_freq * H_filter
        
        # Inverse FFT back to Time Domain
        x_filtered = torch.fft.irfft(X_freq_filtered, n=T, dim=1) # [N, T, C]
        
        # --- C. Dynamic Loss Weights ---
        lambda_alpha = torch.sigmoid(self.alpha_weight_gen(Z)) # [1]
        lambda_beta = torch.sigmoid(self.beta_weight_gen(Z))   # [1]
        
        return x_filtered, lambda_alpha, lambda_beta


class SCGRAN(nn.Module):
    def __init__(self, stocks, time_steps, channels, latent_dim=32):
        super(SCGRAN, self).__init__()
        
        # Spectral Regime Filtering
        self.spectral_mixer = SpectralRegimeMixer(time_steps, channels, latent_dim)
        
        # Temporal Mixing
        self.scale1 = nn.Conv1d(channels, channels, kernel_size=2, stride=2)
        self.mixer = MultTime2dMixer(time_steps, channels)
        self.channel_fc = nn.Linear(channels, 1)
        
        # Final Prediction Heads
        self.time_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)
        self.alpha_fc = nn.Linear(time_steps * 2 + time_steps // 2, 1)

    def forward(self, inputs):
        # 1. Prune high-frequency noise using Latent Regime
        x_filtered, lambda_alpha, lambda_beta = self.spectral_mixer(inputs)
        
        # 2. Extract Temporal Features
        x1 = x_filtered.permute(0, 2, 1)
        x1 = self.scale1(x1)
        x1 = x1.permute(0, 2, 1)

        y = self.mixer(x_filtered, x1)
        y = self.channel_fc(y).squeeze(-1)

        # 3. Predict Total Return (Beta + Alpha)
        total_pred = self.time_fc(y)
        
        # 4. Predict Idiosyncratic Return (Alpha)
        alpha_pred = self.alpha_fc(y)
        
        return total_pred, alpha_pred, lambda_alpha, lambda_beta


def get_scgran_loss(pred, alpha_pred, ground_truth, base_price, mask, stock_num, lambda_alpha, lambda_beta, args):
    device = pred.device
    all_one = torch.ones(stock_num, 1, dtype=torch.float32).to(device)
    
    return_ratio = torch.div(torch.sub(pred, base_price), base_price)
    
    # Base MSE Loss
    mse_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    
    # Pairwise Rank Loss (Beta focus)
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
    
    # Residual Alpha Loss
    epsilon = 1e-8
    market_return = torch.sum(ground_truth * mask, dim=0) / (torch.sum(mask, dim=0) + epsilon)
    market_return = market_return.unsqueeze(0).expand(stock_num, -1)
    alpha_target = ground_truth - market_return
    
    alpha_loss = F.mse_loss(alpha_pred * mask, alpha_target * mask)

    # Dynamic Weighting Conditioned on Regime
    # lambda_alpha and lambda_beta are output by the network (Range: 0 to 1)
    # We multiply them by the base hyperparameters defined in args
    dynamic_alpha_weight = (lambda_alpha.squeeze() * args.lambda_alpha)
    dynamic_beta_weight = (lambda_beta.squeeze() * args.alpha)
    
    loss = mse_loss + dynamic_alpha_weight * alpha_loss + dynamic_beta_weight * rank_loss
    
    return loss, mse_loss, rank_loss, alpha_loss, return_ratio
